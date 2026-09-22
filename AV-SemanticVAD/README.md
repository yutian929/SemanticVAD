# AV-SemanticVAD — 多人 per-face 视听语义 VAD

> 目标会议 **CVPR 2027** ｜ 算力 双卡 H20（96 GB/卡）｜ 骨干 **X2-Turn-4B-0812**（Voxtral 流式版）

在多人同框、流式（80 ms/chunk）设定下，对画面内**每个人**逐 chunk 联合输出
`{在不在说 · ASR · complete/incomplete · 是否在对系统说}`。

---

## 权威文档

| 文件 | 管什么 | 何时读 |
|---|---|---|
| [`plan/cvpr-framing.md`](plan/cvpr-framing.md) | **★ 对外定位的唯一权威**：new setting、C1/C2/C3、措辞纪律、竞品与文献核实、骨干选型 | 一直 |
| [`plan/architecture.md`](plan/architecture.md) | **★ 模型契约的唯一权威**：输入 / 中间流程 / 输出、训练、单测 | 写模型前 |
| [`docs/code-anchors.md`](docs/code-anchors.md) | 第三方参考代码位置（含行号） | 写代码时 |
| [`docs/server-setup.md`](docs/server-setup.md) | GPU 环境、权重、磁盘、已知坑 | 上服务器时 |

**事实记录（非计划）**：[`plan/implementation-execution/`](plan/implementation-execution/) —— 探针怎么跑的、
得到什么数、文献核查结论。其中的**计划与架构段落已作废**，以上面两份为准。

**文献**：[`research/evidence-table.md`](research/evidence-table.md)（证据速查）、
`research/visual-semantic-vad-survey.md`（综述）、`research/methodology.md`（检索方法论）。

---

## 已实测的四个探针（架构的全部实证地基）

| 探针 | 结论 |
|---|---|
| A · 干净单人，X2-Turn hidden 线性探针 | 完整性 **AUC 0.99** → 音频近天花板，且**线性可读** |
| 噪声 gate · 竞争说话人 SNR 10/5/0 | 剔时长后 **0.77 → 0.47** → 混合/重叠下音频**塌** |
| B · 视觉 → 完整性（AMI，73 维 / 因果 GRU） | **0.42 ≈ 随机** → 视觉**不能**直接读完整性 |
| B · 视觉 → 说话人身份（sanity） | acc **0.649** vs 随机 0.25 → 视觉**能**做归属 |

→ **架构第一原则：视觉做归属与朝向，音频做内容与完整性。** 原始产物在 [`results/`](results/)。

---

## 目录

| 路径 | 内容 |
|---|---|
| `plan/` | 两份权威文档 + 执行记录 |
| `scripts/` | 探针脚本（`probe_a_*` / `probe_b_*` / `probe_gate_*` / `probe_noise_*`） |
| `results/` | 探针产物（特征 npz、报告 json） |
| `tests/` | 单测 |
| `annotate/` · `avsvad/` · `configs/` | 标注工具、包骨架、配置（多为占位） |
| `models/` | `face_landmarker.task`（MediaPipe） |

**不改三个只读第三方目录**（`X2-Turn/`、`SoulX-Duplug/`、`SoulX-Duplug-training/`），见 [`../THIRD_PARTY.md`](../THIRD_PARTY.md)。
