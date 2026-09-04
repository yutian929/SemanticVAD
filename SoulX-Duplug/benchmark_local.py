"""SoulX-Duplug 本地推理延迟基准测试。

按 chunk（默认 160ms）流式喂入音频，统计每个 chunk 的推理耗时，并给出
RTF（Real-Time Factor）等实时性指标。RTF < 1 表示能跟上实时音频。

用法：
    python benchmark_local.py                          # 用 assets/tmp.wav，跑 100 个 chunk
    python benchmark_local.py --wav path/to.wav
    python benchmark_local.py --num-chunks 200 --warmup 5
    python benchmark_local.py --device cpu             # 覆盖 config 里的设备
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


def percentile(values, p):
    """取第 p 百分位（p 为 0~100）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p / 100), len(ordered) - 1)
    return ordered[idx]


def load_audio(wav_path, sample_rate=16000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="SoulX-Duplug 本地推理延迟基准")
    parser.add_argument(
        "--wav", default=os.path.join(REPO_DIR, "assets/tmp.wav"), help="测试音频路径"
    )
    parser.add_argument(
        "--config", default=os.path.join(REPO_DIR, "config/config.yaml")
    )
    parser.add_argument("--num-chunks", type=int, default=100, help="统计的 chunk 数")
    parser.add_argument(
        "--warmup", type=int, default=5, help="预热 chunk 数（不计入统计）"
    )
    parser.add_argument(
        "--device", default=None, help="覆盖配置中的设备：cuda / mps / cpu / auto"
    )
    args = parser.parse_args()

    # 关掉 developer_mode 的逐帧日志，避免 print 影响计时
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    cfg.infer_config.developer_mode = False
    if args.device:
        cfg.infer_config.device = args.device

    tmp_config = os.path.join(REPO_DIR, ".benchmark_config.yaml")
    OmegaConf.save(cfg, tmp_config)

    try:
        from service.model import load_turn_model

        print("=" * 62)
        print("SoulX-Duplug 本地推理延迟基准")
        print("=" * 62)

        t0 = time.time()
        model = load_turn_model(config_path=tmp_config)
        load_sec = time.time() - t0

        chunk_size = model.config.infer_config.input.chunk_size
        sample_rate = model.config.infer_config.input.sample_rate
        chunk_ms = chunk_size / sample_rate * 1000

        audio = load_audio(args.wav, sample_rate)
        total_chunks = len(audio) // chunk_size
        n_run = min(args.num_chunks + args.warmup, total_chunks)

        print(f"设备          : {model.device}")
        print(f"ASR           : {model.config.infer_config.asr.model_name}")
        print(f"模型加载耗时  : {load_sec:.1f}s")
        print(f"音频          : {args.wav}")
        print(f"音频时长      : {len(audio) / sample_rate:.1f}s（{total_chunks} chunks）")
        print(f"chunk 大小    : {chunk_size} samples = {chunk_ms:.0f}ms")
        print(f"预热/统计     : {args.warmup} / {n_run - args.warmup} chunks")
        print("-" * 62)

        latencies = []
        state_counter = {}

        for i in range(n_run):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                break

            t = time.time()
            result = model.process(chunk)
            elapsed_ms = (time.time() - t) * 1000

            state = result.get("state", "?")
            if i < args.warmup:
                print(f"  [预热 {i + 1}/{args.warmup}] {elapsed_ms:8.1f}ms  {state}")
                continue

            latencies.append(elapsed_ms)
            state_counter[state] = state_counter.get(state, 0) + 1

            # 只打印超时的 chunk，避免刷屏
            if elapsed_ms > chunk_ms:
                print(
                    f"  chunk {i:>4}  {elapsed_ms:8.1f}ms  {state:<8} "
                    f"<-- 超出实时预算 {chunk_ms:.0f}ms"
                )

        if not latencies:
            print("没有采集到有效样本，请检查音频长度或 --num-chunks。")
            return

        mean_ms = statistics.mean(latencies)
        rtf = mean_ms / chunk_ms
        over = sum(1 for x in latencies if x > chunk_ms)

        print("-" * 62)
        print(f"样本数        : {len(latencies)}")
        print(f"平均延迟      : {mean_ms:8.1f} ms")
        print(f"中位数 (P50)  : {statistics.median(latencies):8.1f} ms")
        print(f"P90           : {percentile(latencies, 90):8.1f} ms")
        print(f"P99           : {percentile(latencies, 99):8.1f} ms")
        print(f"最小 / 最大   : {min(latencies):8.1f} / {max(latencies):.1f} ms")
        print(f"实时预算      : {chunk_ms:8.0f} ms per chunk")
        print(f"超预算 chunk  : {over} / {len(latencies)} ({over / len(latencies) * 100:.1f}%)")
        print("-" * 62)
        print(f"RTF           : {rtf:.3f}  （<1 表示可实时）")
        if rtf < 1:
            print(f"结论          : 可以实时运行，平均留有 {chunk_ms - mean_ms:.0f}ms 余量")
        else:
            print(f"结论          : 达不到实时，平均慢 {mean_ms - chunk_ms:.0f}ms/chunk，约需加速 {rtf:.1f}x")
        print(f"状态分布      : {state_counter}")
        print("=" * 62)

    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)


if __name__ == "__main__":
    main()
