#!/usr/bin/env python
"""P0.3 探针 A — 在 X2-Turn hidden 上训线性探针，出 AUC，并对照混淆因素。

读 probe_a_extract.py 产出的 npz（可多分片），用分层 5 折交叉验证训 logistic
regression，报告各特征集的 ROC-AUC：

  feat_last   判决点(末帧) hidden [3072]        —— 探针 A 主指标
  feat_mean   全帧 mean-pool hidden [3072]
  turn_probs  frozen turn head 6 类概率 [6]      —— 零训练基线（模型现成能力）
  turn_end    仅 turn_end 概率 [1]               —— 单特征零训练基线
  length      仅 frame_count [1]                 —— ★ 时长混淆基线（incomplete 更短）

判据（implementation-plan.md §P0.3）:
  >0.85 音频基线很强→必须分层汇报 ｜ 0.70–0.85 有空间→主线推进 ｜ <0.70 信息不足
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path("/home/thundrzhang/SemanticVAD")
PROBE_DIR = REPO / "AV-SemanticVAD/results/probe_a"
REPORT = PROBE_DIR / "probe_a_report.json"
SEED = 0


def load_all():
    shards = sorted(glob.glob(str(PROBE_DIR / "features_shard*of*.npz")))
    if not shards:
        raise SystemExit(f"no feature shards in {PROBE_DIR}")
    parts = {k: [] for k in ("feat_last", "feat_mean", "turn_probs",
                             "labels", "ids", "frame_counts")}
    names = None
    for s in shards:
        d = np.load(s, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])
        names = list(d["turn_class_names"])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    print(f"loaded {len(shards)} shards, N={len(data['labels'])} "
          f"(complete={int((data['labels']==1).sum())}, "
          f"incomplete={int((data['labels']==0).sum())})")
    return data, names


def auc_cv(X, y, C=1.0):
    if X.ndim == 1:
        X = X[:, None]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, proba)),
            "ap": float(average_precision_score(y, proba))}


def main():
    data, turn_names = load_all()
    y = data["labels"].astype(int)
    turn_end_i = turn_names.index("turn_end")
    uncertain_i = turn_names.index("uncertain")

    feature_sets = {
        "feat_last": data["feat_last"],
        "feat_mean": data["feat_mean"],
        "turn_probs": data["turn_probs"],
        "turn_end_prob": data["turn_probs"][:, turn_end_i],
        "uncertain_prob": data["turn_probs"][:, uncertain_i],
        "length_framecount": data["frame_counts"].astype(float),
    }

    print("\n=== 探针 A: 交叉验证 AUC ===")
    results = {}
    # feat_* 是 3072 维、N<d，扫几个 C 取最优
    for name, X in feature_sets.items():
        if X.ndim == 2 and X.shape[1] > 100:
            best = max((auc_cv(X, y, C=c) | {"C": c} for c in (0.01, 0.1, 1.0)),
                       key=lambda r: r["auc"])
        else:
            best = auc_cv(X, y) | {"C": 1.0}
        results[name] = best
        print(f"  {name:20s}  AUC={best['auc']:.4f}  AP={best['ap']:.4f}  (C={best['C']})")

    main_auc = results["feat_last"]["auc"]
    length_auc = results["length_framecount"]["auc"]
    if main_auc > 0.85:
        verdict = (">0.85 → 音频基线很强，视觉空间小；必须改走分层汇报"
                   "（停顿类型 × 噪声条件），不报全集 F1")
    elif main_auc >= 0.70:
        verdict = "0.70–0.85 → 有真实空间，按主线推进"
    else:
        verdict = "<0.70 → hidden 里话轮信息不足，考虑解冻 vad_lm_head 或加大 LoRA"

    length_note = (
        f"⚠️ 时长混淆：length-only AUC={length_auc:.3f}。"
        + ("与 feat_last 接近，需警惕探针在借时长而非语义（做长度分层/回归剔除后复核）。"
           if length_auc >= main_auc - 0.05 else
           f"feat_last({main_auc:.3f}) 明显高于时长基线，hidden 携带的完整性信息超出单纯时长。")
    )

    report = {
        "n_samples": int(len(y)),
        "n_complete": int((y == 1).sum()),
        "n_incomplete": int((y == 0).sum()),
        "results": results,
        "main_metric": "feat_last",
        "main_auc": main_auc,
        "verdict": verdict,
        "length_confound": length_note,
        "turn_class_names": turn_names,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n主指标 feat_last AUC = {main_auc:.4f}")
    print(f"判据: {verdict}")
    print(length_note)
    print(f"\n报告 -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
