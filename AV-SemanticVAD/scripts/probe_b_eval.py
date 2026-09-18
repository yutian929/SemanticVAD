#!/usr/bin/env python
"""探针 B 评测：视觉特征 → 预测 complete/incomplete，报 AUC + 混淆基线。

对照（对齐探针 A 方法）：
- visual_all（73 维池化视觉，主指标）
- 混淆基线：pause_dur（时长）、face_valid_ratio（人脸有效率，检测是否泄露标签）
- chance
CV：分层 5 折 logistic regression；另报 leave-one-meeting-out（防同人泄露）。

用法：python probe_b_eval.py ES2002a ES2002b ES2002c ES2002d
"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut

def load(meetings):
    Xs, ys, durs, vrs, grp = [], [], [], [], []
    for gi, m in enumerate(meetings):
        d = np.load(f"AV-SemanticVAD/results/probe_b/{m}_visual.npz", allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"]); durs.append(d["dur"])
        vrs.append(d["valid_ratio"]); grp += [gi] * len(d["y"])
    return (np.concatenate(Xs), np.concatenate(ys).astype(int),
            np.concatenate(durs), np.concatenate(vrs), np.array(grp))

def auc_cv(X, y, groups=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if groups is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    else:
        s = cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=groups, scoring="roc_auc")
    return s.mean(), s.std()

def main():
    meetings = sys.argv[1:] or ["ES2002a", "ES2002b", "ES2002c", "ES2002d"]
    X, y, dur, vr, grp = load(meetings)
    print(f"# meetings={meetings}")
    print(f"# N={len(y)}  complete={int((y==1).sum())} incomplete={int((y==0).sum())}  "
          f"baseline(majority)={max(y.mean(),1-y.mean()):.3f}")
    print(f"# face_valid_ratio: median={np.median(vr):.2f} min={vr.min():.2f}")
    print()
    for name, feat in [("visual_all(73d)", X),
                       ("pause_dur(1d)", dur.reshape(-1,1)),
                       ("face_valid_ratio(1d)", vr.reshape(-1,1))]:
        m, s = auc_cv(feat, y)
        print(f"  {name:22s} 5fold AUC = {m:.3f} ± {s:.3f}")
    # leave-one-meeting-out（同会议同人，跨会议更严）
    m, s = auc_cv(X, y, groups=grp)
    print(f"  {'visual_all LOMO':22s}  AUC = {m:.3f} ± {s:.3f}   (leave-one-meeting-out)")

if __name__ == "__main__":
    main()
