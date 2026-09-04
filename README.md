# SemanticVAD — 视听语义 VAD 研究工作区

> **主项目**：[`AV-SemanticVAD/`](AV-SemanticVAD/) —— 融合视觉模态的语义 VAD，目标会议 **IROS**
>
> **一句话**：把 complete/incomplete 判决从纯音频扩展到视听 ——
> 建语料（C1）、改模型（C2）、上真机（C3）。

**首次接手本项目，请从 [`START-HERE.md`](START-HERE.md) 开始。**

---

## 目录构成

| 路径 | 角色 | 说明 |
|---|---|---|
| [`START-HERE.md`](START-HERE.md) | **入口** | 阅读顺序、当前状态、下一步动作 |
| [`AV-SemanticVAD/`](AV-SemanticVAD/) | **主项目** | 计划、调研、代码（待实现） |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 溯源清单 | 三份第三方代码的 SHA / 许可 / 只读约定 |
| [`X2-Turn/`](X2-Turn/) | 第三方（只读） | **base 权重来源**，`@53d3b9a` |
| [`SoulX-Duplug/`](SoulX-Duplug/) | 第三方（只读） | 范式参考（推理侧），`main @45bd237` |
| [`SoulX-Duplug-training/`](SoulX-Duplug-training/) | 第三方（只读） | 范式参考（训练侧），`training-code @928b065` |

三份第三方代码均为 **Apache-2.0**，`LICENSE` 已随代码保留。
**已直接纳入仓库，不使用 git submodule** —— `git clone` 一次即可。

---

## 四者的关系

```
X2-Turn-4B-0812 权重  ──►  全部冻结，作为 base
                              （其 hidden 已含话轮表征，故不用裸 Voxtral）
SoulX-Duplug          ──►  「冻结前端 + 可训 Projector + LoRA」范式
                              （同任务的成功先例，含显式 complete/incomplete 实现）
SoulX-Duplug-training ──►  训练循环与官方数据格式
                              （finetune.py / example_data_fisher.jsonl）
AV-SemanticVAD        ──►  在 base 上侧接一路视觉，输出逐帧 complete/incomplete
```

> ⚠️ `SoulX-Duplug/` 与 `SoulX-Duplug-training/` **是互补的两棵树，不是包含关系**。
> 上游把推理与训练放在不同分支：前者有 `service/model.py`（complete/incomplete 判决），
> 后者删掉了它但加入 `finetune.py` 等训练工具。**两个都要留。**

---

## 快速开始（GPU 服务器）

```bash
git clone https://github.com/yutian929/SemanticVAD.git
cd SemanticVAD
cat START-HERE.md
```

完整部署步骤（环境、权重下载、磁盘估算）见
[`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md)。

---

## 文档权威性约定

**为避免多处描述冲突，每类信息只有一个权威来源：**

| 信息 | 权威来源 | 其他文档 |
|---|---|---|
| 任务清单、Gate、指标定义 | `AV-SemanticVAD/plan/implementation-plan.md` | 只链接 |
| 架构与设计论证 | `AV-SemanticVAD/plan/architecture.md` | 只链接 |
| 环境、权重下载、已知坑 | `AV-SemanticVAD/docs/server-setup.md` | 只链接 |
| **第三方代码位置与行号** | `AV-SemanticVAD/docs/code-anchors.md` | 只链接 |
| 第三方来源 SHA 与许可 | `THIRD_PARTY.md` | 只链接 |
| 文献证据与数字 | `AV-SemanticVAD/research/evidence-table.md` | 只链接 |

**冲突时以权威来源为准。** 修改后请同步更新引用它的文档。
