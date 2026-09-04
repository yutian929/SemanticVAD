"""测量 ASR 延迟对整体 RTF 的影响。

ASR 处在严格的串行关键路径上（LLM forward -> ASR -> tokenize -> LLM forward），
所以它的耗时会直接加到每个 chunk 的预算里。本脚本用一个可注入固定延迟的
假 ASR 替换真实 ASR，用来回答：

  1. 如果 ASR 完全免费（0ms），RTF 能到多少？——即优化 ASR 的收益上限
  2. 换成网络 API（每次多出 RTT）之后，RTF 会变成多少？

用法：
    python benchmark_asr_impact.py                        # 扫 0/50/100/150/200ms
    python benchmark_asr_impact.py --delays 0,80,150
    python benchmark_asr_impact.py --num-chunks 60
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import soundfile as sf
import soxr

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


class FakeASR:
    """返回固定文本的假 ASR，用于把 ASR 耗时控制成一个已知常量。

    目的是隔离变量：真实 ASR 的耗时会随音频内容波动，而这里只关心
    "ASR 花了多久" 对整体 RTF 的影响，所以文本内容保持不变。
    """

    def __init__(self, delay_ms, text="你好我想问一下这个模型"):
        self.delay_sec = delay_ms / 1000.0
        self.text = text
        self.call_count = 0

    def recognize(self, audio_chunk, sample_rate=16000):
        self.call_count += 1
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        return self.text


def load_audio(wav_path, sample_rate=16000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def run_once(model, audio, chunk_size, num_chunks, warmup):
    """跑一轮，返回 (统计用延迟列表, ASR 调用次数)。"""
    model.reset()
    model.cascade_asr.call_count = 0

    latencies = []
    for i in range(num_chunks + warmup):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) < chunk_size:
            break
        t = time.time()
        model.process(chunk)
        elapsed_ms = (time.time() - t) * 1000
        if i >= warmup:
            latencies.append(elapsed_ms)

    return latencies, model.cascade_asr.call_count


def main():
    parser = argparse.ArgumentParser(description="ASR 延迟对 RTF 的影响")
    parser.add_argument("--wav", default=os.path.join(REPO_DIR, "assets/tmp.wav"))
    parser.add_argument("--config", default=os.path.join(REPO_DIR, "config/config.yaml"))
    parser.add_argument("--num-chunks", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--delays",
        default="0,50,100,150,200",
        help="要测试的 ASR 延迟（毫秒），逗号分隔",
    )
    args = parser.parse_args()

    delays = [int(x) for x in args.delays.split(",")]

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    cfg.infer_config.developer_mode = False
    tmp_config = os.path.join(REPO_DIR, ".asr_impact_config.yaml")
    OmegaConf.save(cfg, tmp_config)

    try:
        from service.model import load_turn_model

        print("=" * 70)
        print("ASR 延迟对整体 RTF 的影响")
        print("=" * 70)

        model = load_turn_model(config_path=tmp_config)
        chunk_size = model.config.infer_config.input.chunk_size
        sample_rate = model.config.infer_config.input.sample_rate
        chunk_ms = chunk_size / sample_rate * 1000

        audio = load_audio(args.wav, sample_rate)

        print(f"设备        : {model.device}")
        print(f"chunk 预算  : {chunk_ms:.0f} ms")
        print(f"统计样本    : {args.num_chunks} chunks（预热 {args.warmup}）")
        print()
        print("说明：用固定延迟的假 ASR 替换真实 ASR，隔离出 ASR 耗时这一个变量。")
        print("      delay=0 即 ASR 完全免费，代表优化 ASR 的收益上限。")
        print("-" * 70)
        print(f"{'ASR延迟':>8} | {'平均':>8} | {'P50':>8} | {'P90':>8} | {'RTF':>6} | {'ASR调用':>7} | 实时?")
        print("-" * 70)

        results = []
        for delay_ms in delays:
            model.cascade_asr = FakeASR(delay_ms)
            latencies, asr_calls = run_once(
                model, audio, chunk_size, args.num_chunks, args.warmup
            )
            if not latencies:
                continue

            mean_ms = statistics.mean(latencies)
            rtf = mean_ms / chunk_ms
            verdict = "是" if rtf < 1 else "否"
            results.append((delay_ms, mean_ms, rtf))

            print(
                f"{delay_ms:>6}ms | {mean_ms:>7.1f}ms | "
                f"{statistics.median(latencies):>7.1f}ms | "
                f"{sorted(latencies)[int(len(latencies) * 0.9)]:>7.1f}ms | "
                f"{rtf:>6.2f} | {asr_calls:>7} | {verdict}"
            )

        print("-" * 70)

        if results:
            base_delay, base_mean, base_rtf = results[0]
            # 用最小延迟那档反推出非 ASR 部分的固定开销
            asr_call_ratio = None
            if len(results) >= 2:
                d0, m0, _ = results[0]
                d1, m1, _ = results[-1]
                if d1 != d0:
                    # 每 chunk 平均承担的 ASR 次数（<1 说明只有部分 chunk 触发 ASR）
                    asr_call_ratio = (m1 - m0) / (d1 - d0)

            print()
            print("解读：")
            print(f"  非 ASR 固定开销（GLM tokenizer + 2x LLM forward）≈ {base_mean:.0f}ms")
            if asr_call_ratio is not None:
                print(f"  ASR 平均触发率 ≈ {asr_call_ratio:.2f} 次/chunk（静音段不触发）")
                budget_left = chunk_ms - base_mean
                if asr_call_ratio > 0:
                    max_affordable = budget_left / asr_call_ratio
                    print(f"  要达到实时，单次 ASR 必须快于 ≈ {max_affordable:.0f}ms")
                    print(f"  （{chunk_ms:.0f}ms 预算 - {base_mean:.0f}ms 固定开销，再摊到 {asr_call_ratio:.2f} 次调用上）")
            print()

    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)


if __name__ == "__main__":
    main()
