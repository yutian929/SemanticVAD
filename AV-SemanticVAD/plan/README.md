# plan/ — 导航（先读这一页）

> **给下一个接手的人 / Claude agent**：这一页让你一眼知道文件都是什么、按什么顺序读、现在到哪一步。

## 当前状态（2026-09-14）

- **进度**：项目 **Phase 0（前置探针与环境）已完成**；Phase 1（数据集）尚未开始。
- **关键结论（探针实测）**：
  1. 干净条件下 X2-Turn 纯音频判 complete/incomplete **AUC≈0.99** → 音频近天花板，视觉在干净语义线上无空间。
  2. 词法明显的句中停顿：音频**没被骗**（强 LM 能读懂）。
  3. **竞争说话人噪声下音频完整性信号崩到近随机（0.77→0.47）** → **音频在声学退化下失效，视觉有潜在主场**。
- **当前 framing**：不是"视觉帮语义判断"，而是 **"音频被干扰、完整性信号塌掉时，视觉维持话轮判断"**。
- **下一步**：取**带视频 AV 数据**，验证噪声下**视觉能否补回**音频丢失的信号（sufficiency）。未验前，不宣称贡献 C2 成立、也不宣称项目失败。
- **数据决策（2026-09-16）**：主训练数据用 **MM-F2F（英文，非中文；只发标注+YouTube链接，媒体须自下）**，**英文一条线先跑通**；中文暂无现成大规模 AV 源，靠自采/Full-Duplex-Bench-zh，后定。详见 `implementation-plan.md` 顶部 changelog。

## 文件地图与阅读顺序

| 顺序 | 文件 | 是什么 | 权威性 |
|---|---|---|---|
| 0 | [`../../START-HERE.md`](../../START-HERE.md) | 30 秒看懂项目在做什么 | 入口 |
| 1 | **本文件** | plan/ 导航 + 当前状态 | 导航 |
| 2 | [`implementation-plan.md`](implementation-plan.md) | **唯一权威计划**：Phase 0–5 任务 / Gate / 指标。顶部有**执行 changelog** | ★ 权威（任务/指标） |
| 3 | [`implementation-execution/`](implementation-execution/) | **执行记录**（按项目 phase 分）：实际做了什么、得到什么结果 | 事实记录 |
| 4 | [`architecture.md`](architecture.md) | 架构与设计论证（采纳/否决的路线） | ★ 权威（架构） |
| 5 | [`benchmark-proposal.md`](benchmark-proposal.md) | benchmark 设计草案（**提案 / 待验证 / 非权威**） | 提案 |

> **文档权威性约定**（冲突时以此判断，且**只改权威、不留两份**）：
> 任务/Gate/指标 → `implementation-plan.md`；架构/论证 → `architecture.md`；
> 代码位置行号 → [`../docs/code-anchors.md`](../docs/code-anchors.md)；文献数字 → [`../research/evidence-table.md`](../research/evidence-table.md)。

## plan/ 与 execution/ 的关系

- `implementation-plan.md` = **该做什么**（Phase 0–5，权威）。
- `implementation-execution/phaseN.md` = **实际做了什么、结论**（按项目 phase 分文件）。今天所有工作都属 **Phase 0**，见 `implementation-execution/phase0.md`。
- 未来 Phase 1–5 各自新增 `phaseN.md`。

## 环境 / 复现速查

- 本机 Python：`/home/thundrzhang/miniconda3/envs/x2-turn/bin/python`（conda env `x2-turn`，2×H20）。
- 权重软链：`X2-Turn/models/X2-Turn-4B-0812`（→ `/home/thundrzhang/X2-Turn/models`）。
- 探针脚本：`../scripts/`；结果：`../results/`；执行细节见 `implementation-execution/phase0.md`。
