# SemanticVAD — 视听语义 VAD 研究工作区

> **主项目**：[`AV-SemanticVAD/`](AV-SemanticVAD/) —— 融合视觉模态的语义 VAD（目标会议 **IROS**）
>
> 本仓库是一个**自包含工作区**：主项目与三份第三方代码树全部在库内，
> **不使用 git submodule** —— `git clone` 一次即可开始工作。

---

## 一、目录构成

| 路径 | 角色 | 说明 |
|---|---|---|
| **[`AV-SemanticVAD/`](AV-SemanticVAD/)** | **主项目** | 计划、调研、代码（待实现） |
| [`X2-Turn/`](X2-Turn/) | **base 权重来源** | 第三方，只读。`@53d3b9a` (2026-08-31) |
| [`SoulX-Duplug/`](SoulX-Duplug/) | **范式参考**（推理服务） | 第三方，只读。`main @45bd237` |
| [`SoulX-Duplug-training/`](SoulX-Duplug-training/) | **训练代码参考** | 第三方，只读。`training-code @928b065` |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | **溯源清单** | 三份代码的来源 SHA / 许可 / 修改政策 / 更新方法 |

三份第三方代码均为 **Apache-2.0**，`LICENSE` 已随代码保留。

**四者的关系**（详见 `AV-SemanticVAD/plan/architecture.md`）：

```
X2-Turn-4B-0812 权重  ──►  全部冻结，作为 base
                              （其 hidden 已含话轮表征，故不用裸 Voxtral）
SoulX-Duplug          ──►  提供「冻结前端 + 可训 Projector + LoRA」范式
                              （同任务的成功先例，含显式 complete/incomplete）
SoulX-Duplug-training ──►  提供训练循环与数据格式（finetune.py / example_data_*.jsonl）
AV-SemanticVAD        ──►  在 base 上侧接一路视觉，输出逐帧 complete/incomplete
```

> ⚠️ **`SoulX-Duplug-training/` 不是 `SoulX-Duplug/` 的超集，而是另一棵树** ——
> 上游把推理服务和训练代码放在两个分支上，前者删掉了 `service/model.py` 等推理代码，
> 加入了 `finetune.py`、`utils/ema/` 等训练工具。两者互补，都要留。

---

## 二、在 GPU 服务器上从零拉起

### 2.1 克隆

```bash
git clone https://github.com/yutian929/SemanticVAD.git
cd SemanticVAD
```

**一条命令即可，没有子模块。** 克隆后约 12 MB（权重不在库内）。

私有仓库的凭据配置见 [`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md) §1.1。

### 2.2 权重下载（**不在库内**，7.2 GB 已被 gitignore）

```bash
# base 权重
huggingface-cli download x-square-robot/X2-Turn-4B-0812 \
    --local-dir X2-Turn/models/X2-Turn-4B-0812

# 范式参考权重（可选，仅在需要复现 SoulX 时）
huggingface-cli download Soul-AILab/SoulX-Duplug-0.6B \
    --local-dir SoulX-Duplug/pretrained_models/SoulX-Duplug-0.6B

# 评测集
huggingface-cli download Soul-AILab/SoulX-Duplug-Eval --repo-type dataset \
    --local-dir data/SoulX-Duplug-Eval
```

### 2.3 环境

X2-Turn 已自带安装脚本（2026-08-31 上游新增）：

```bash
cd X2-Turn && bash install.sh
```

详细步骤与 H20 相关配置见 [`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md)。

---

## 三、第三方代码的只读约定与更新

**`X2-Turn/`、`SoulX-Duplug/`、`SoulX-Duplug-training/` 三个目录一律只读。**

我们的改造走**包装**而非 fork：新增代码全在 `AV-SemanticVAD/avsvad/` 下，
通过 `avsvad/upstream.py` 引用上游模块。这样才能在论文里诚实声明「未修改基座代码」。

**如何与上游对比或更新**：见 [`THIRD_PARTY.md`](THIRD_PARTY.md) §4。
简言之 —— 在仓库外单独 clone 一份上游做 `diff -r`，确认后再替换，
**替换完必须回去更新 `THIRD_PARTY.md` 的 SHA**。

---

## 四、当前状态（2026-09-04）

| 项 | 状态 |
|---|---|
| 调研 | ✅ 完成（`AV-SemanticVAD/research/`） |
| 架构设计 | ✅ v4（`plan/architecture.md`） |
| 实施计划 | ✅ v3.2（`plan/implementation-plan.md`） |
| **代码实现** | ⬜ **未开始** —— 目录骨架已建，从 Phase 0 起步 |
| 算力 | **双卡 H20（96 GB/卡）** |

**第三方代码基线版本**（完整信息见 [`THIRD_PARTY.md`](THIRD_PARTY.md)）：

| 目录 | 分支 | 提交 | 日期 |
|---|---|---|---|
| X2-Turn | `main` | `53d3b9a` | 2026-08-31 |
| SoulX-Duplug | `main` | `45bd237` | 2026-08-24 |
| SoulX-Duplug-training | `training-code` | `928b065` | 2026-07-17 |

---

## 五、下一步

按 [`AV-SemanticVAD/plan/implementation-plan.md`](AV-SemanticVAD/plan/implementation-plan.md) 的「立即执行的三件事」：

1. **W1 D1**：提交伦理审批 + 核对 MM-F2F 再分发条款（最长串行依赖）
2. **W1**：配环境、载 X2-Turn 权重、查 V1/V2/V3
3. **W2**：跑**探针 A** 与**探针 B** —— 两个 AUC 数字决定整篇论文的 framing

> ⚠️ **不要跳过探针直接进入训练。** 现有全部证据都指向「视觉增益来自噪声鲁棒性，
> 不来自语义判断」。若探针 A 显示 X2-Turn hidden 的线性探针已达 0.90+，
> 必须先切换论文 framing 再投入训练。
