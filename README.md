# SemanticVAD — 视听语义 VAD 研究工作区

> **主项目**：[`AV-SemanticVAD/`](AV-SemanticVAD/) —— 多人同框、流式、per-face 的语义完整性判决。
> 目标会议 **CVPR 2027**。
>
> **资源前提**：人力、物力、算力充足，**数据不是问题**——数据可自采自标到所需规模，不列为风险。

**一句话**：在多人同框、流式（80 ms/chunk）的设定下，对画面内**每个人**逐 chunk 联合判
`{在不在说 · 说了什么(ASR) · 说完没(语义完整性) · 是否在对系统说}`；
视觉负责**归属 + 朝向**，音频负责**内容与完整性**。

## 权威文档（只有这两份，冲突以它们为准）

| 文件 | 管什么 |
|---|---|
| [`AV-SemanticVAD/plan/cvpr-framing.md`](AV-SemanticVAD/plan/cvpr-framing.md) | **对外定位**：new setting、三点贡献、竞品口径、骨干选型、文献核实记录 |
| [`AV-SemanticVAD/plan/architecture.md`](AV-SemanticVAD/plan/architecture.md) | **模型契约**：输入 / 中间流程 / 输出、训练、单测、风险 |

其余目录：`AV-SemanticVAD/plan/implementation-execution/`（探针与调研的**事实记录**，其中的计划部分已作废）、
`AV-SemanticVAD/docs/`（服务器部署、第三方代码锚点）、`AV-SemanticVAD/research/`（文献证据）、
`AV-SemanticVAD/results/`（探针原始产物）。

> **2026-09-21 清理**：旧的 `START-HERE.md`、`plan/README.md`、`implementation-plan.md`、
> `architecture-legacy-v4.md`、`architecture-diagram.html`、`benchmark-proposal.md` 已删除
> （它们停留在 IROS / 冻结骨干 / 单说话人的旧方向）。需要时从 git 历史 `ac5dee1` 取回。

## 目录构成

| 路径 | 角色 | 说明 |
|---|---|---|
| [`AV-SemanticVAD/`](AV-SemanticVAD/) | **主项目** | 计划、调研、脚本、结果 |
| `data/` | 数据 | AMI（ES2002a–d，4 路 Closeup + 标注）、MM-F2F（仅交接说明） |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 溯源清单 | 三份第三方代码的 SHA / 许可 / 只读约定 |
| [`X2-Turn/`](X2-Turn/) | 第三方（只读） | **骨干权重来源**（X2-Turn-4B-0812），上游 `@8992c7c` |
| [`SoulX-Duplug/`](SoulX-Duplug/) | 第三方（只读） | 范式参考（推理侧），`main @45bd237` |
| [`SoulX-Duplug-training/`](SoulX-Duplug-training/) | 第三方（只读） | 范式参考（训练侧），`training-code @928b065` |

三份第三方代码均为 **Apache-2.0**，`LICENSE` 已随代码保留；已直接纳入仓库，不使用 submodule。
