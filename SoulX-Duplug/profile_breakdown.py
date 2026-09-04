"""SoulX-Duplug 单 chunk 耗时逐段拆解。

把 process() 一个 chunk 的耗时拆到各个组成部分，定位真正的瓶颈：

    process(chunk)
      └─ get_chunk()                 缓冲区切片，几乎不耗时
      └─ state_predict()
           └─ infer()
                ├─ _audio_to_tokens()   Whisper 特征提取 + GLM VQ 编码
                ├─ _tokens_to_embeds()  codebook 查表 + projector MLP
                ├─ _asr()               LLM forward #1  +  级联 ASR
                └─ _state_predict()     LLM forward #2

用 monkey patch 包裹各个方法来计时，不修改被测代码本身。

用法：
    python profile_breakdown.py                    # 默认 60 chunks
    python profile_breakdown.py --num-chunks 100
    python profile_breakdown.py --device cpu
"""

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import soxr
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


class Timer:
    """按名字累计各段耗时。"""

    def __init__(self, device):
        self.records = defaultdict(list)
        self.device = device

    def sync(self):
        """MPS/CUDA 是异步派发的，不同步会把耗时算到后面的调用上。"""
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def wrap(self, obj, method_name, label):
        """用计时逻辑包裹 obj 的某个方法。"""
        original = getattr(obj, method_name)

        def wrapped(*args, **kwargs):
            self.sync()
            t = time.perf_counter()
            result = original(*args, **kwargs)
            self.sync()
            self.records[label].append((time.perf_counter() - t) * 1000)
            return result

        setattr(obj, method_name, wrapped)
        return original

    def wrap_module(self, module, label):
        """给 nn.Module 挂前后钩子计时。

        nn.Module 的子模块不能用 setattr 替换成普通函数（PyTorch 会拦截），
        所以这里改用 forward pre/post hook。
        """
        state = {"t": 0.0}

        def pre_hook(mod, inputs):
            self.sync()
            state["t"] = time.perf_counter()

        def post_hook(mod, inputs, output):
            self.sync()
            self.records[label].append((time.perf_counter() - state["t"]) * 1000)

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)


def load_audio(wav_path, sample_rate=16000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def fmt_row(label, samples, total_mean, chunk_ms, calls_per_chunk):
    """输出一行统计。samples 为空时打印占位。"""
    if not samples:
        return f"  {label:<26} {'—':>9} {'—':>9} {'—':>9} {'—':>7} {'—':>8}"

    mean = statistics.mean(samples)
    # 摊到每个 chunk 上的平均贡献（考虑该段并非每 chunk 都触发）
    amortized = mean * calls_per_chunk
    share = amortized / total_mean * 100 if total_mean else 0
    return (
        f"  {label:<26} {mean:>7.1f}ms {statistics.median(samples):>7.1f}ms "
        f"{max(samples):>7.1f}ms {calls_per_chunk:>6.2f} {share:>6.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="单 chunk 耗时逐段拆解")
    parser.add_argument("--wav", default=os.path.join(REPO_DIR, "assets/tmp.wav"))
    parser.add_argument("--config", default=os.path.join(REPO_DIR, "config/config.yaml"))
    parser.add_argument("--num-chunks", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    cfg.infer_config.developer_mode = False
    if args.device:
        cfg.infer_config.device = args.device
    tmp_config = os.path.join(REPO_DIR, ".profile_config.yaml")
    OmegaConf.save(cfg, tmp_config)

    try:
        from service.model import load_turn_model

        print("=" * 78)
        print("SoulX-Duplug 单 chunk 耗时拆解")
        print("=" * 78)

        model = load_turn_model(config_path=tmp_config)
        chunk_size = model.config.infer_config.input.chunk_size
        sample_rate = model.config.infer_config.input.sample_rate
        chunk_ms = chunk_size / sample_rate * 1000

        audio = load_audio(args.wav, sample_rate)

        # 先跑预热，让 MPS 编译完 kernel，再挂计时钩子
        for i in range(args.warmup):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) == chunk_size:
                model.process(chunk)

        timer = Timer(model.device)
        timer.wrap(model, "_audio_to_tokens", "audio->tokens (GLM VQ)")
        timer.wrap(model, "_tokens_to_embeds", "tokens->embeds (proj)")
        timer.wrap(model, "_asr", "_asr (LLM#1 + ASR)")
        timer.wrap(model, "_state_predict", "_state_predict (LLM#2)")
        timer.wrap(model.cascade_asr, "recognize", "  └─ ASR only")

        # 单独计时 LLM 前向，用来把 _asr 里 LLM#1 和 ASR 的耗时分开
        timer.wrap_module(model.model.llm, "  └─ LLM forward (all)")
        # 再单独看 GLM VQ encoder 本身（_audio_to_tokens 里还含 Whisper 特征提取）
        timer.wrap_module(model.model.glm_tokenizer, "  └─ GLM VQ encoder")

        print(f"设备        : {model.device}")
        print(f"ASR         : {model.config.infer_config.asr.model_name}")
        print(f"chunk 预算  : {chunk_ms:.0f} ms")
        print(f"样本        : {args.num_chunks} chunks（已预热 {args.warmup}）")

        total = []
        offset = args.warmup
        for i in range(args.num_chunks):
            chunk = audio[(offset + i) * chunk_size : (offset + i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                break
            timer.sync()
            t = time.perf_counter()
            model.process(chunk)
            timer.sync()
            total.append((time.perf_counter() - t) * 1000)

        n = len(total)
        total_mean = statistics.mean(total)

        print("-" * 78)
        print(f"  {'组成部分':<24} {'均值':>9} {'中位':>9} {'最大':>9} {'次/chunk':>7} {'占比':>8}")
        print("-" * 78)

        for label in [
            "audio->tokens (GLM VQ)",
            "  └─ GLM VQ encoder",
            "tokens->embeds (proj)",
            "_asr (LLM#1 + ASR)",
            "  └─ ASR only",
            "  └─ LLM forward (all)",
            "_state_predict (LLM#2)",
        ]:
            s = timer.records.get(label, [])
            print(fmt_row(label, s, total_mean, chunk_ms, len(s) / n if n else 0))

        print("-" * 78)
        print(f"  {'单 chunk 总耗时':<24} {total_mean:>7.1f}ms "
              f"{statistics.median(total):>7.1f}ms {max(total):>7.1f}ms "
              f"{1.0:>6.2f} {100.0:>6.1f}%")
        print("-" * 78)

        # 汇总：LLM vs ASR vs 音频前端
        llm_all = timer.records.get("  └─ LLM forward (all)", [])
        asr_all = timer.records.get("  └─ ASR only", [])
        vq = timer.records.get("audio->tokens (GLM VQ)", [])
        proj = timer.records.get("tokens->embeds (proj)", [])

        def amort(samples):
            return statistics.mean(samples) * len(samples) / n if samples and n else 0.0

        llm_ms, asr_ms = amort(llm_all), amort(asr_all)
        vq_ms, proj_ms = amort(vq), amort(proj)
        other = total_mean - llm_ms - asr_ms - vq_ms - proj_ms

        print()
        print("按模块归并（已摊到每 chunk）：")
        print(f"  LLM 前向 (2 次/chunk)     {llm_ms:>7.1f}ms  {llm_ms / total_mean * 100:>5.1f}%"
              f"   [{len(llm_all) / n:.2f} 次/chunk]")
        print(f"  级联 ASR                  {asr_ms:>7.1f}ms  {asr_ms / total_mean * 100:>5.1f}%"
              f"   [{len(asr_all) / n:.2f} 次/chunk]")
        print(f"  音频 -> token (GLM VQ)    {vq_ms:>7.1f}ms  {vq_ms / total_mean * 100:>5.1f}%")
        print(f"  projector MLP             {proj_ms:>7.1f}ms  {proj_ms / total_mean * 100:>5.1f}%")
        print(f"  其余 (文本处理/拷贝等)     {other:>7.1f}ms  {other / total_mean * 100:>5.1f}%")
        print(f"  {'-' * 52}")
        print(f"  合计                      {total_mean:>7.1f}ms   100.0%"
              f"   (预算 {chunk_ms:.0f}ms, RTF {total_mean / chunk_ms:.2f})")
        print("=" * 78)

    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)


if __name__ == "__main__":
    main()
