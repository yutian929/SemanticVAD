#!/usr/bin/env python
"""Gate 实验评估 — 硬停顿上纯音频的完整性可分性 + 冻结模型是否被停顿骗到。

对比锚点：探针 A 在 Easy-Turn（简单孤立句）feat_last AUC = 0.994。

输出两组证据：
 (1) 判别 AUC（complete vs incomplete）+ 时长混淆控制（长度匹配 / 回归剔除），与 0.994 对比。
 (2) ★ 无混淆的直接证据：冻结 turn head 在"句中思考停顿(incomplete)"点是否预测 turn_end
     （= 会不会错误抢话）。按类分别统计，天然不受长度混淆影响。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path("/home/thundrzhang/SemanticVAD")
GATE = REPO / "AV-SemanticVAD/results/probe_gate"
REPORT = GATE / "gate_report.json"
EASY_TURN_AUC = 0.9944  # 探针 A 锚点
SEED = 0


def load():
    shards = sorted(glob.glob(str(GATE / "features_shard*of*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {GATE}")
    keys = ("feat_last", "feat_mean", "turn_probs", "labels", "ids", "frame_counts", "cats")
    P = {k: [] for k in keys}
    names = None
    for s in shards:
        d = np.load(s, allow_pickle=True)
        for k in keys:
            P[k].append(d[k])
        names = list(d["turn_class_names"])
    return {k: np.concatenate(v) for k, v in P.items()}, names


def auc_cv(X, y, C=1.0):
    if X.ndim == 1:
        X = X[:, None]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def best_auc(X, y):
    return max(auc_cv(X, y, C=c) for c in (0.01, 0.1, 1.0))


def main():
    d, names = load()
    y = d["labels"].astype(int)
    FL, fc, tp = d["feat_last"], d["frame_counts"].astype(float), d["turn_probs"]
    turn_end_i = names.index("turn_end")
    print(f"N={len(y)}  complete={int((y==1).sum())}  incomplete={int((y==0).sum())}")

    # (1) 判别 AUC + 长度控制
    auc_last = best_auc(FL, y)
    auc_len = auc_cv(fc, y)
    # 长度匹配子集
    q_lo, q_hi = np.percentile(fc[y == 0], 25), np.percentile(fc[y == 1], 75)
    band = (fc >= q_lo) & (fc <= q_hi)
    matched = None
    if (y[band] == 0).sum() > 20 and (y[band] == 1).sum() > 20:
        matched = {"band": [float(q_lo), float(q_hi)], "n": int(band.sum()),
                   "feat_last_auc": best_auc(FL[band], y[band]),
                   "length_auc": auc_cv(fc[band], y[band])}
    # 回归剔除长度
    resid = FL - LinearRegression().fit(fc[:, None], FL).predict(fc[:, None])
    auc_resid = best_auc(resid, y)

    # (2) 冻结 turn head 是否被停顿骗到（无混淆）
    pte = tp[:, turn_end_i]  # P(turn_end)
    argmax_is_turnend = (tp.argmax(1) == turn_end_i)
    inc = y == 0  # incomplete（句中思考停顿：正确应“别抢话”）
    com = y == 1
    frozen = {
        "incomplete_pause": {
            "n": int(inc.sum()),
            "mean_P_turn_end": float(pte[inc].mean()),
            "interrupt_rate_argmax_turn_end": float(argmax_is_turnend[inc].mean()),
        },
        "complete_end": {
            "n": int(com.sum()),
            "mean_P_turn_end": float(pte[com].mean()),
            "turn_end_rate_argmax": float(argmax_is_turnend[com].mean()),
        },
    }

    drop = EASY_TURN_AUC - auc_last
    gate_pass = auc_last < 0.85 or (auc_resid < 0.80)
    verdict = (
        f"GATE {'PASS' if gate_pass else 'FAIL'}：硬停顿 feat_last AUC={auc_last:.3f}"
        f"（Easy-Turn 0.994，跌 {drop:+.3f}）；剔除长度残差 AUC={auc_resid:.3f}。"
        + ("音频在硬停顿上明显变差 → pivot 有实证地基。"
           if gate_pass else
           "音频在硬停顿上仍强 → benchmark 论点削弱，需重想切片。")
    )

    report = {
        "anchor_easy_turn_feat_last_auc": EASY_TURN_AUC,
        "n": int(len(y)),
        "discrimination": {
            "feat_last_auc": auc_last,
            "length_only_auc": auc_len,
            "length_matched": matched,
            "length_regressed_out_auc": auc_resid,
            "drop_vs_easy_turn": drop,
        },
        "frozen_turn_head": frozen,
        "gate_pass": bool(gate_pass),
        "verdict": verdict,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== (1) 判别 AUC（complete vs incomplete）===")
    print(f"  feat_last            = {auc_last:.4f}   (Easy-Turn 0.994, 跌 {drop:+.3f})")
    print(f"  length-only          = {auc_len:.4f}")
    if matched:
        print(f"  feat_last(长度匹配)  = {matched['feat_last_auc']:.4f}  "
              f"(length {matched['length_auc']:.4f}, N={matched['n']})")
    print(f"  feat_last(剔除长度)  = {auc_resid:.4f}")
    print("\n=== (2) 冻结 turn head 在停顿点是否被骗（无混淆）===")
    print(f"  incomplete 停顿点: mean P(turn_end)={frozen['incomplete_pause']['mean_P_turn_end']:.3f}, "
          f"误抢话率(argmax=turn_end)={frozen['incomplete_pause']['interrupt_rate_argmax_turn_end']:.3f}")
    print(f"  complete 说完点:  mean P(turn_end)={frozen['complete_end']['mean_P_turn_end']:.3f}, "
          f"turn_end 率={frozen['complete_end']['turn_end_rate_argmax']:.3f}")
    print(f"\n{verdict}")
    print(f"报告 -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
