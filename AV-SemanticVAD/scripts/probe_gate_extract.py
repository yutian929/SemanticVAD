#!/usr/bin/env python
"""Gate 实验 v2 — 硬停顿纯音频探针（修正版）。

v1 的两个错误（见 gate_report.json 的教训）：
 (a) 长度伪迹：incomplete 截到句中停顿(短)、complete 是整句(长)，length-only AUC 就 0.99。
 (b) 冻结头退化：读"截断片段的末帧"落在**尾部静音**里，turn head 恒 idle=1.0，无信息。

修正：
 - **判决帧 = 语音结束帧**（停顿前最后一个词的结束时刻 / turn-taking 前末词结束），
   不是尾部静音帧。模型在该帧有自然前视(480ms)看进静音、但截断在停顿结束以免看到后续。
 - 主指标改为**无混淆的按类统计**：该帧上 turn head 各类概率、turn_end 率、
   以及"看起来像说完了(idle/turn_end)"的比例——incomplete vs complete 分别看。
 - 判别 AUC 仍报，但配严格长度匹配，且不作为主判据。
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio
from voxtral_realtime.transformers.inference import _encode_audio, STREAMING_PAD_ID
from voxtral_realtime.transformers.modeling import TURN_CLASS_IDS, TURN_CLASS_NAMES, load_mtp_checkpoint

REPO = Path("/home/thundrzhang/SemanticVAD")
MODEL_DIR = REPO / "X2-Turn/models/X2-Turn-4B-0812"
FDB = REPO / "data/SoulX-Duplug-Eval/Full-Duplex-Bench-zh"
TRUNC = REPO / "data/SoulX-Duplug-Eval/_gate_trunc2"
OUT_DIR = REPO / "AV-SemanticVAD/results/probe_gate"
FRAME_MS = 80


def build_items():
    """返回 [(wav_trunc, label, id, decision_t, clip_end_t)]。
    decision_t = 语音结束(判决点); clip_end_t = 截断点(含供前视的静音，不含后续语音)。"""
    TRUNC.mkdir(parents=True, exist_ok=True)
    items = []
    for d in sorted((FDB / "pause_handling").iterdir(), key=lambda p: p.name):
        if not d.is_dir(): continue
        pa = json.load(open(d / "pause.json"))
        tr = json.load(open(d / "transcription.json"))
        if not pa or not tr: continue
        p_start, p_end = float(pa[0]["timestamp"][0]), float(pa[0]["timestamp"][1])
        # 语音结束帧 = 停顿开始（停顿前最后一词的结束）
        decision_t = p_start
        clip_end = p_end                      # 截到停顿结束（有静音供前视，无后续语音）
        out = TRUNC / f"pause_{d.name}.wav"
        _trunc(d / "input.wav", out, clip_end)
        items.append({"wav": str(out), "label": 0, "id": f"pause/{d.name}",
                      "decision_t": decision_t, "clip_end": clip_end})
    for d in sorted((FDB / "turn_taking").iterdir(), key=lambda p: p.name):
        if not d.is_dir(): continue
        tr = json.load(open(d / "transcription.json"))
        if not tr: continue
        decision_t = float(tr[-1]["timestamp"][1])   # 末词结束
        clip_end = decision_t + 0.70                  # 留 0.7s 尾静音供前视（与停顿相当）
        out = TRUNC / f"turn_{d.name}.wav"
        _trunc(d / "input.wav", out, clip_end)
        items.append({"wav": str(out), "label": 1, "id": f"turn/{d.name}",
                      "decision_t": decision_t, "clip_end": clip_end})
    return items


def _trunc(src, dst, end_s):
    if dst.exists(): return
    wav, sr = sf.read(src)
    n = min(len(wav), int(round(end_s * sr)))
    sf.write(dst, wav[:n], sr)


@torch.inference_mode()
def extract_at(model, processor, wav, decision_t, delay_ms, alpt):
    audio = Audio.from_file(str(wav), strict=False)
    sr = int(processor.feature_extractor.sampling_rate); audio.resample(sr)
    dev = next(model.parameters()).device
    enc = _encode_audio(processor, audio, delay_ms)
    input_ids = enc.input_ids.to(dev)
    feats = enc.input_features.to(device=dev, dtype=model.dtype)
    ndt = int(enc.num_delay_tokens)
    prefix = int(input_ids.shape[1])
    n_audio = math.ceil(feats.shape[-1] / alpt)
    fcount = max(n_audio - prefix, 0)
    if fcount <= 0: return None
    gen = model.generate(input_ids=input_ids, input_features=feats, num_delay_tokens=ndt,
                         do_sample=False, num_beams=1, max_new_tokens=max(fcount, 1))[0]
    aligned = gen.unsqueeze(0)
    if aligned.shape[1] < n_audio:
        pad = torch.full((1, n_audio - aligned.shape[1]), STREAMING_PAD_ID, dtype=aligned.dtype, device=dev)
        aligned = torch.cat([aligned, pad], 1)
    else:
        aligned = aligned[:, :n_audio]
    out = model.base_model.model(input_ids=aligned, input_features=feats, num_delay_tokens=ndt)
    hid = out.last_hidden_state[0]
    # 判决帧：decision_t 对应的帧 f，读 hidden[prefix + f - 1]
    f = int(round(decision_t * 1000 / FRAME_MS))
    f = max(0, min(f, fcount - 1))
    idx = prefix + f - 1
    idx = max(0, min(idx, hid.shape[0] - 1))
    feat = hid[idx].float().cpu().numpy()
    vad = model.vad_lm_head(hid[idx].to(model.vad_lm_head.weight.dtype))
    probs = torch.softmax(vad[list(TURN_CLASS_IDS)].float(), -1).cpu().numpy()
    return {"feat": feat, "turn_probs": probs, "frame_count": fcount, "decision_frame": f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    i, n = (int(x) for x in a.shard.split("/"))
    items = build_items()
    print(f"[gate2] {len(items)} clips (inc={sum(x['label']==0 for x in items)}, com={sum(x['label']==1 for x in items)})")
    items = [x for k, x in enumerate(items) if k % n == i]
    if a.limit: items = items[:a.limit]
    proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = load_mtp_checkpoint(str(MODEL_DIR), device=dev, dtype=dt); model.eval()
    delay_ms = int(getattr(model.config, "default_num_delay_tokens", 6)) * FRAME_MS
    alpt = int(getattr(model.config, "audio_length_per_tok", 8))
    feat, tp, lab, ids, fc, dfr = [], [], [], [], [], []
    t0 = time.time()
    for k, s in enumerate(items):
        r = extract_at(model, proc, s["wav"], s["decision_t"], delay_ms, alpt)
        if r is None:
            print("skip", s["id"]); continue
        feat.append(r["feat"]); tp.append(r["turn_probs"]); lab.append(s["label"])
        ids.append(s["id"]); fc.append(r["frame_count"]); dfr.append(r["decision_frame"])
        if (k+1) % 25 == 0 or k == 0:
            print(f"  {k+1}/{len(items)} {time.time()-t0:.0f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"v2_shard{i}of{n}.npz"
    np.savez(out, feat=np.stack(feat), turn_probs=np.stack(tp), labels=np.array(lab),
             ids=np.array(ids), frame_counts=np.array(fc), decision_frames=np.array(dfr),
             turn_class_names=np.array(list(TURN_CLASS_NAMES)))
    print(f"[gate2] wrote {out} ({len(lab)}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
