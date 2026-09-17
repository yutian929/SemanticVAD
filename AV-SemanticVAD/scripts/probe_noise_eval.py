#!/usr/bin/env python
"""噪声鲁棒性 gate 评估 — 画出 feat_last AUC 随 SNR 的退化曲线。

对比：clean（探针 A，results/probe_a）vs 竞争说话人 SNR {10,5,0}（results/probe_noise）。
重点看**剔除长度后**的 AUC：clean≈0.77，若随 SNR 降到 ~0.5 → 音频语义判断被噪声毁 →
视觉有潜在主场（必要条件满足）。full feat_last AUC 因长度地板(~0.86)不会跌到底，仅作参考。
"""
from __future__ import annotations

import glob, json, re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path("/home/thundrzhang/SemanticVAD")
PROBE_A = REPO / "AV-SemanticVAD/results/probe_a"
PROBE_N = REPO / "AV-SemanticVAD/results/probe_noise"
REPORT = PROBE_N / "noise_report.json"


def load(shards):
    if not shards:
        return None
    P = {k: [] for k in ("feat_last", "labels", "frame_counts")}
    for s in shards:
        d = np.load(s, allow_pickle=True)
        for k in P:
            P[k].append(d[k])
    return {k: np.concatenate(v) for k, v in P.items()}


def auc(X, y, C=1.0):
    if X.ndim == 1:
        X = X[:, None]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def best(X, y):
    return max(auc(X, y, C=c) for c in (0.01, 0.1, 1.0))


def evaluate(d):
    y = d["labels"].astype(int)
    FL, fc = d["feat_last"], d["frame_counts"].astype(float)
    resid = FL - LinearRegression().fit(fc[:, None], FL).predict(fc[:, None])
    return {"n": int(len(y)), "feat_last_auc": best(FL, y),
            "length_only_auc": auc(fc, y),
            "length_regressed_out_auc": best(resid, y)}


def main():
    conds = {}
    clean = load(sorted(glob.glob(str(PROBE_A / "features_shard*of*.npz"))))
    if clean:
        conds["clean"] = evaluate(clean)
    # 噪声条件
    snrs = {}
    for f in glob.glob(str(PROBE_N / "babble_snr*_shard*of*.npz")):
        m = re.search(r"babble_snr([0-9.]+)_shard", f)
        snrs.setdefault(float(m.group(1)), []).append(f)
    for snr in sorted(snrs, reverse=True):
        d = load(sorted(snrs[snr]))
        if d is not None:
            conds[f"babble_snr{snr:g}"] = evaluate(d)

    print(f"{'condition':16s} {'N':>4s} {'feat_last':>10s} {'len-only':>9s} {'len-removed':>12s}")
    for name, r in conds.items():
        print(f"{name:16s} {r['n']:>4d} {r['feat_last_auc']:>10.3f} "
              f"{r['length_only_auc']:>9.3f} {r['length_regressed_out_auc']:>12.3f}")

    # 判读：剔除长度后的 AUC 随噪声是否明显下降
    verdict = "（噪声条件尚未跑完）"
    if "clean" in conds and any(k.startswith("babble") for k in conds):
        clean_r = conds["clean"]["length_regressed_out_auc"]
        worst = min(conds[k]["length_regressed_out_auc"]
                    for k in conds if k.startswith("babble"))
        drop = clean_r - worst
        if worst < 0.62 and drop > 0.10:
            verdict = (f"音频语义判断在噪声下明显退化（剔除长度 AUC {clean_r:.2f}→{worst:.2f}）"
                       f"→ **必要条件满足**：视觉有潜在主场，值得取视觉数据验证能否补回。")
        else:
            verdict = (f"音频在噪声下仍稳（剔除长度 AUC {clean_r:.2f}→{worst:.2f}，跌 {drop:+.2f}）"
                       f"→ 视觉主场证据不足，倾向诚实转 dataset+negative result。")
    print(f"\n判读：{verdict}")
    REPORT.write_text(json.dumps({"conditions": conds, "verdict": verdict},
                                 ensure_ascii=False, indent=2))
    print(f"报告 -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
