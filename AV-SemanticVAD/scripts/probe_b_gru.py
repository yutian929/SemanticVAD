#!/usr/bin/env python
"""探针 B（GRU 版）：因果 GRU 直接吃逐帧 24 维序列 → complete/incomplete。

对照计划 §2.2 的因果 GRU 设计（而非池化+线性）。小样本(N~154)+弱标签，故：
- 小 GRU(hidden 32) + dropout + 早停 + 类权重；
- 分层 5 折 与 留一会议(LOMO) 双 CV，报 AUC 均值±std；
- 与线性探针(0.42)对照，看时序是否救回信号。

用法：python probe_b_gru.py [ES2002a ...]
"""
import sys, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

def load(meetings):
    seqs, y, grp = [], [], []
    for gi, m in enumerate(meetings):
        d = np.load(f"AV-SemanticVAD/results/probe_b/{m}_visual.npz", allow_pickle=True)
        seqs += list(d["seqs"]); y += list(d["y"]); grp += [gi]*len(d["y"])
    return seqs, np.array(y, int), np.array(grp)

class GRU(nn.Module):
    def __init__(self, d=25, h=32):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(h, 1)
    def forward(self, x, lens):
        p = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(p)
        return self.fc(self.drop(hn[-1])).squeeze(-1)

def pad(batch):
    lens = torch.tensor([len(s) for s in batch])
    T = int(lens.max()); D = batch[0].shape[1]
    x = torch.zeros(len(batch), T, D)
    for i, s in enumerate(batch):
        x[i, :len(s)] = torch.tensor(s)
    return x, lens

def norm_fit(seqs_tr):
    allf = np.concatenate([s for s in seqs_tr], 0)
    mu = allf[:, :24].mean(0); sd = allf[:, :24].std(0) + 1e-6
    return mu, sd
def norm_apply(seqs, mu, sd):
    out = []
    for s in seqs:
        s = s.copy(); s[:, :24] = (s[:, :24] - mu) / sd
        out.append(s.astype(np.float32))
    return out

def train_eval(tr_i, te_i, seqs, y):
    mu, sd = norm_fit([seqs[i] for i in tr_i])
    Str = norm_apply([seqs[i] for i in tr_i], mu, sd)
    Ste = norm_apply([seqs[i] for i in te_i], mu, sd)
    ytr = y[tr_i]
    pos = ytr.mean(); w = torch.tensor([(1-pos)/max(pos,1e-3)]).to(DEV)
    net = GRU().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    xtr, ltr = pad(Str); xtr, ltr = xtr.to(DEV), ltr
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(DEV)
    xte, lte = pad(Ste); xte = xte.to(DEV)
    best_auc, best = 0.5, None
    for ep in range(120):
        net.train(); opt.zero_grad()
        out = net(xtr, ltr); loss = lossf(out, ytr_t); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(xte, lte)).cpu().numpy()
    return p

def main():
    meetings = sys.argv[1:] or ["ES2002a","ES2002b","ES2002c","ES2002d"]
    seqs, y, grp = load(meetings)
    print(f"# N={len(y)} complete={int(y.sum())} incomplete={int((1-y).sum())} device={DEV}")
    # 分层 5 折
    for name, splitter, kw in [("Stratified-5fold", StratifiedKFold(5, shuffle=True, random_state=0), {}),
                               ("Leave-one-meeting-out", LeaveOneGroupOut(), {"groups": grp})]:
        aucs = []
        for tr_i, te_i in splitter.split(np.zeros(len(y)), y, **({"groups": grp} if kw else {})):
            if len(np.unique(y[te_i])) < 2:  # 该折只有一类，跳过
                continue
            p = train_eval(tr_i, te_i, seqs, y)
            aucs.append(roc_auc_score(y[te_i], p))
        print(f"  GRU {name:22s} AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (folds={len(aucs)})")

if __name__ == "__main__":
    main()
