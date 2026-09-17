#!/usr/bin/env python
"""P0.2 — 加载 base、复现帧级输出、固化 golden、解锁 V3 / V7。

对应 implementation-plan.md §P0.2（P0.2.2–P0.2.6）。全程"调用"上游，不改第三方目录。

用法:
    python AV-SemanticVAD/scripts/run_backbone.py [--device cuda:0]

产物:
    AV-SemanticVAD/tests/golden/backbone.json   固定样本的帧级回归基线 (P0.2.4)
    AV-SemanticVAD/results/p0_2_backbone.json    V3/V7 报告 + 摘要
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint
from voxtral_realtime.transformers.modeling import TURN_CLASS_IDS, TURN_CLASS_NAMES

REPO = Path("/home/thundrzhang/SemanticVAD")
MODEL_DIR = REPO / "X2-Turn/models/X2-Turn-4B-0812"
SAMPLE = REPO / "X2-Turn/turn-demo/assets/sample_en.wav"
GOLDEN = REPO / "AV-SemanticVAD/tests/golden/backbone.json"
REPORT = REPO / "AV-SemanticVAD/results/p0_2_backbone.json"


def find_decoder_layer0(model) -> dict:
    """V3 (P0.2.5): 定位 decoder layer 0 的模块路径 —— 视觉融合的 hook 挂点。"""
    layer0_paths, container_paths = [], []
    for name, _ in model.named_modules():
        if re.search(r"(^|\.)layers\.0$", name):
            layer0_paths.append(name)
        if re.search(r"(^|\.)layers$", name):
            container_paths.append(name)
    # 逐帧读出的 hidden 来自 base_model.model(...).last_hidden_state（modeling.py:127-128）
    inner = model.base_model.model
    inner_type = type(inner).__name__
    return {
        "hidden_source": "model.base_model.model(...).last_hidden_state",
        "inner_model_type": inner_type,
        "layers_containers": container_paths,
        "decoder_layer0_paths": layer0_paths,
    }


def frames_to_jsonable(frames, keep_probs=False):
    out = []
    for f in frames:
        d = {"index": f.index, "start_ms": f.start_ms, "end_ms": f.end_ms,
             "label": f.label, "confidence": round(f.confidence, 4)}
        if keep_probs:
            d["probabilities"] = {k: round(v, 4) for k, v in f.probabilities.items()}
        out.append(d)
    return out


def turn_events(frames):
    """非平凡（非 uncertain/最高频类）帧的紧凑序列，便于对比 delay 影响。"""
    return [(f.index, f.label, round(f.confidence, 3))
            for f in frames if f.label != TURN_CLASS_NAMES[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    report = {"model_dir": str(MODEL_DIR), "device": device, "dtype": str(dtype)}

    # ---- P0.2.2 加载 base ----
    print(f"[P0.2.2] loading base on {device} ({dtype}) ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = load_mtp_checkpoint(str(MODEL_DIR), device=device, dtype=dtype)
    model.eval()
    load_s = time.time() - t0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        vram_gb = torch.cuda.memory_allocated(device) / 1e9
    else:
        vram_gb = None
    print(f"  loaded in {load_s:.1f}s, VRAM allocated ≈ {vram_gb:.2f} GB"
          if vram_gb else f"  loaded in {load_s:.1f}s (cpu)")
    report["load"] = {"seconds": round(load_s, 1), "vram_gb": round(vram_gb, 2) if vram_gb else None}
    report["turn_class_ids"] = list(TURN_CLASS_IDS)
    report["turn_class_names"] = list(TURN_CLASS_NAMES)

    # ---- P0.2.5 V3: decoder layer 0 hook 挂点 ----
    v3 = find_decoder_layer0(model)
    report["V3_hook"] = v3
    print(f"[P0.2.5/V3] inner model = {v3['inner_model_type']}")
    print(f"           layers containers = {v3['layers_containers']}")
    print(f"           decoder layer0 = {v3['decoder_layer0_paths'][:6]}")

    # ---- P0.2.3 复现帧级输出（默认 delay）----
    print(f"[P0.2.3] infer_asr_turn on {SAMPLE.name} (default delay) ...")
    t0 = time.time()
    res = infer_asr_turn(model, processor, str(SAMPLE))
    infer_s = time.time() - t0
    print(f"  frame_ms = {res.frame_ms}  (期望 80)")
    print(f"  n_frames = {len(res.turn_frames)}  (3.4s 音频期望 ≈53)")
    print(f"  transcript = {res.transcript!r}")
    print(f"  infer {infer_s:.1f}s")
    report["P0_2_3"] = {
        "frame_ms": res.frame_ms,
        "n_frames": len(res.turn_frames),
        "transcript": res.transcript,
        "infer_seconds": round(infer_s, 1),
        "nontrivial_events": turn_events(res.turn_frames),
    }
    assert res.frame_ms == 80, f"frame_ms={res.frame_ms}, 期望 80 —— 视听对齐前提被破坏!"

    # ---- P0.2.4 固化 golden ----
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    golden = {
        "source_audio": str(SAMPLE.relative_to(REPO)),
        "model_dir": str(MODEL_DIR.relative_to(REPO)),
        "frame_ms": res.frame_ms,
        "n_frames": len(res.turn_frames),
        "transcript": res.transcript,
        "generated_token_ids": res.generated_token_ids,
        "frames": frames_to_jsonable(res.turn_frames, keep_probs=True),
    }
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=2))
    print(f"[P0.2.4] golden -> {GOLDEN.relative_to(REPO)}")

    # ---- P0.2.6 V7: turn head 是否吃 delay tokens ----
    print("[P0.2.6/V7] comparing delay_ms = 80 vs default(480) ...")
    v7 = {}
    for delay in (80, 480):
        r = infer_asr_turn(model, processor, str(SAMPLE), delay_ms=delay)
        v7[str(delay)] = {
            "n_frames": len(r.turn_frames),
            "transcript": r.transcript,
            "nontrivial_events": turn_events(r.turn_frames),
        }
        print(f"   delay={delay}ms: n_frames={len(r.turn_frames)}, "
              f"events={v7[str(delay)]['nontrivial_events'][:8]}")
    same = v7["80"]["nontrivial_events"] == v7["480"]["nontrivial_events"]
    v7["events_identical_80_vs_480"] = same
    v7["interpretation"] = (
        "turn 帧时序对 delay_ms 不敏感 → 视觉前视约束可放松到 80ms" if same
        else "turn 帧时序随 delay_ms 变化 → turn head 消费 delay tokens，前视不能免费放松"
    )
    report["V7"] = v7
    print(f"   → identical={same}: {v7['interpretation']}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[done] report -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
