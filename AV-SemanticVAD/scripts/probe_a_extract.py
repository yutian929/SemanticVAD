#!/usr/bin/env python
"""P0.3 探针 A — 抽取 X2-Turn 在"话轮判决点"的 hidden 表征。

数据: SoulX-Duplug-Eval / Easy-Turn-Testset-en（318 complete + 299 incomplete，
金标签，Apache-2.0）。每条是一段 utterance，标签=该句在结尾处是否语义完整。

做法（复刻 inference.py 的前向，但额外取 hidden；不改任何第三方代码）:
  encode → generate(ASR 对齐序列) → 对 base_model.model 单独前向取 last_hidden_state
  → 逐帧索引 prediction_index = prefix_length + frame_index − 1（inference.py:165）
  → 取"最后一帧"(判决点) 的 hidden，以及全帧 mean-pool
  → 顺带用 vad_lm_head 读该帧 6 类 turn 概率（frozen turn head 零训练基线）

对应 implementation-plan.md §P0.3。输出 npz 供 probe_a_train.py 训分类器。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio

from voxtral_realtime.transformers.inference import _encode_audio, STREAMING_PAD_ID
from voxtral_realtime.transformers.modeling import (
    TURN_CLASS_IDS, TURN_CLASS_NAMES, load_mtp_checkpoint,
)

REPO = Path("/home/thundrzhang/SemanticVAD")
MODEL_DIR = REPO / "X2-Turn/models/X2-Turn-4B-0812"
DATA = REPO / "data/SoulX-Duplug-Eval/Easy-Turn-Testset-en"
OUT_DIR = REPO / "AV-SemanticVAD/results/probe_a"


def list_samples():
    samples = []
    for cat, label in (("complete", 1), ("incomplete", 0)):
        for wav in sorted((DATA / cat).glob("*.wav"),
                          key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            samples.append({"wav": str(wav), "label": label, "cat": cat,
                            "id": f"{cat}/{wav.stem}"})
    return samples


@torch.inference_mode()
def extract_one(model, processor, wav_path, delay_ms, audio_len_per_tok):
    audio = Audio.from_file(str(wav_path), strict=False)
    sr = int(processor.feature_extractor.sampling_rate)
    audio.resample(sr)
    device = next(model.parameters()).device

    encoded = _encode_audio(processor, audio, delay_ms)
    input_ids = encoded.input_ids.to(device)
    input_features = encoded.input_features.to(device=device, dtype=model.dtype)
    num_delay_tokens = int(encoded.num_delay_tokens)
    prefix_length = int(input_ids.shape[1])
    num_audio_tokens = math.ceil(input_features.shape[-1] / audio_len_per_tok)
    frame_count = max(num_audio_tokens - prefix_length, 0)
    if frame_count <= 0:
        return None

    generated = model.generate(
        input_ids=input_ids, input_features=input_features,
        num_delay_tokens=num_delay_tokens, do_sample=False, num_beams=1,
        max_new_tokens=max(frame_count, 1),
    )[0]
    aligned = generated.unsqueeze(0)
    if aligned.shape[1] < num_audio_tokens:
        pad = torch.full((1, num_audio_tokens - aligned.shape[1]), STREAMING_PAD_ID,
                         dtype=aligned.dtype, device=device)
        aligned = torch.cat([aligned, pad], dim=1)
    else:
        aligned = aligned[:, :num_audio_tokens]

    # hidden 来自 base_model.model（正是 modeling.py:127-128 / 159-160 的做法）
    outputs = model.base_model.model(
        input_ids=aligned, input_features=input_features,
        num_delay_tokens=num_delay_tokens,
    )
    hidden = outputs.last_hidden_state[0]  # [seq, H]

    frame_idx = [prefix_length + f - 1 for f in range(frame_count)]
    frame_idx = [i for i in frame_idx if 0 <= i < hidden.shape[0]]
    last_i = frame_idx[-1]
    feat_last = hidden[last_i].float().cpu().numpy()
    feat_mean = hidden[frame_idx].float().mean(0).cpu().numpy()

    # frozen turn head 在判决点的 6 类概率（零训练基线参考）
    vad_last = model.vad_lm_head(hidden[last_i].to(model.vad_lm_head.weight.dtype))
    turn_probs = torch.softmax(vad_last[list(TURN_CLASS_IDS)].float(), -1).cpu().numpy()

    return {"feat_last": feat_last, "feat_mean": feat_mean,
            "turn_probs": turn_probs, "frame_count": frame_count,
            "n_audio_tokens": num_audio_tokens, "prefix_length": prefix_length}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0=all；>0 只跑前 N（计时用）")
    ap.add_argument("--shard", default="0/1", help="i/n 分片，多卡并行用")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    i, n = (int(x) for x in args.shard.split("/"))

    print(f"[extract] device={device} shard={args.shard} limit={args.limit}")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = load_mtp_checkpoint(str(MODEL_DIR), device=device, dtype=dtype)
    model.eval()
    frame_ms = 80
    delay_ms = int(getattr(model.config, "default_num_delay_tokens", 6)) * frame_ms
    alpt = int(getattr(model.config, "audio_length_per_tok", 8))

    samples = list_samples()
    samples = [s for k, s in enumerate(samples) if k % n == i]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[extract] {len(samples)} samples this shard")

    feats_last, feats_mean, turn_probs, labels, ids, meta = [], [], [], [], [], []
    t0 = time.time()
    for k, s in enumerate(samples):
        r = extract_one(model, processor, s["wav"], delay_ms, alpt)
        if r is None:
            print(f"  skip (0 frames): {s['id']}"); continue
        feats_last.append(r["feat_last"]); feats_mean.append(r["feat_mean"])
        turn_probs.append(r["turn_probs"]); labels.append(s["label"]); ids.append(s["id"])
        meta.append({"frame_count": r["frame_count"], "cat": s["cat"]})
        if (k + 1) % 25 == 0 or k == 0:
            dt = time.time() - t0
            print(f"  {k+1}/{len(samples)}  {dt:.0f}s  ({dt/(k+1):.2f}s/it)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.tag or f"shard{i}of{n}"
    out = OUT_DIR / f"features_{suffix}.npz"
    np.savez(out,
             feat_last=np.stack(feats_last), feat_mean=np.stack(feats_mean),
             turn_probs=np.stack(turn_probs), labels=np.array(labels),
             ids=np.array(ids), frame_counts=np.array([m["frame_count"] for m in meta]),
             turn_class_names=np.array(list(TURN_CLASS_NAMES)))
    print(f"[extract] wrote {out}  ({len(labels)} samples, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
