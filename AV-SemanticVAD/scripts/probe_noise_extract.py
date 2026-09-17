#!/usr/bin/env python
"""噪声鲁棒性 gate — 在竞争说话人噪声下重跑纯音频探针（决定视觉有无主场）。

数据：Easy-Turn-en（干净时探针 A feat_last AUC≈0.99）。
噪声：竞争说话人（从其它 Easy-Turn 片段随机取一段混入，按目标 SNR 缩放）——
      即 AV-Dialog 里视觉唯一稳定有效的"干扰说话人"场景。也支持 white。
时长不变 → 若 AUC 随 SNR 下降，是声学-语义信息被毁，而非长度伪迹。

复用探针 A 的抽取（末帧 hidden + turn_probs）。输出 npz 供 probe_noise_eval.py。
"""
from __future__ import annotations

import argparse, random, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor
from voxtral_realtime.transformers.modeling import TURN_CLASS_NAMES, load_mtp_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_a_extract import list_samples, extract_one

REPO = Path("/home/thundrzhang/SemanticVAD")
MODEL_DIR = REPO / "X2-Turn/models/X2-Turn-4B-0812"
NOISE_DIR = REPO / "data/SoulX-Duplug-Eval/_noise"
OUT_DIR = REPO / "AV-SemanticVAD/results/probe_noise"


def mix_at_snr(sig, noise, snr_db, rng):
    if len(noise) < len(sig):
        reps = int(np.ceil(len(sig) / len(noise)))
        noise = np.tile(noise, reps)
    start = rng.randint(0, max(0, len(noise) - len(sig)))
    noise = noise[start:start + len(sig)]
    if len(noise) < len(sig):  # 兜底：保证与 sig 等长
        noise = np.pad(noise, (0, len(sig) - len(noise)))
    ps = float(np.mean(sig ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    mixed = sig + scale * noise
    peak = np.max(np.abs(mixed)) + 1e-9
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def make_noisy(samples, snr_db, noise_type, pool, seed=0):
    outdir = NOISE_DIR / f"{noise_type}_snr{snr_db}"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    out_items = []
    for s in samples:
        dst = outdir / f"{s['cat']}_{Path(s['wav']).stem}.wav"
        if not dst.exists():
            sig, sr = sf.read(s["wav"])
            sig = np.asarray(sig, dtype=np.float32)
            if sig.ndim > 1:
                sig = sig[:, 0]
            if noise_type == "babble":
                other = s["wav"]
                while other == s["wav"]:
                    other = rng.choice(pool)
                nz, _ = sf.read(other)
                nz = np.asarray(nz, dtype=np.float32)
                if nz.ndim > 1:
                    nz = nz[:, 0]
            else:  # white
                nz = np.random.default_rng(seed + hash(s["id"]) % 100000).standard_normal(len(sig)).astype(np.float32)
            mixed = mix_at_snr(sig, nz, snr_db, rng)
            sf.write(dst, mixed, sr)
        out_items.append({**s, "wav": str(dst)})
    return out_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--noise", default="babble", choices=["babble", "white"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    i, n = (int(x) for x in a.shard.split("/"))

    all_samples = list_samples()
    pool = [s["wav"] for s in all_samples]
    shard = [s for k, s in enumerate(all_samples) if k % n == i]
    if a.limit:
        shard = shard[: a.limit]
    print(f"[noise] {a.noise} SNR={a.snr}dB shard {a.shard}: {len(shard)} clips — preparing ...")
    samples = make_noisy(shard, a.snr, a.noise, pool=pool)

    proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = load_mtp_checkpoint(str(MODEL_DIR), device=dev, dtype=dt).eval()
    delay_ms = int(getattr(model.config, "default_num_delay_tokens", 6)) * 80
    alpt = int(getattr(model.config, "audio_length_per_tok", 8))

    fl, tp, lab, ids, fc = [], [], [], [], []
    t0 = time.time()
    for k, s in enumerate(samples):
        r = extract_one(model, proc, s["wav"], delay_ms, alpt)
        if r is None:
            continue
        fl.append(r["feat_last"]); tp.append(r["turn_probs"])
        lab.append(s["label"]); ids.append(s["id"]); fc.append(r["frame_count"])
        if (k + 1) % 50 == 0 or k == 0:
            print(f"  {k+1}/{len(samples)} {time.time()-t0:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.noise}_snr{a.snr}_shard{i}of{n}.npz"
    np.savez(out, feat_last=np.stack(fl), turn_probs=np.stack(tp), labels=np.array(lab),
             ids=np.array(ids), frame_counts=np.array(fc),
             turn_class_names=np.array(list(TURN_CLASS_NAMES)))
    print(f"[noise] wrote {out} ({len(lab)}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
