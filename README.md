# SemanticVAD — 视听语义 VAD 研究工作区

> **主项目**：[`AV-SemanticVAD/`](AV-SemanticVAD/) —— 融合视觉模态的语义 VAD（目标会议 **IROS**）
>
> 本仓库是一个**工作区聚合仓库**：主项目代码文档在库内，两个上游基座以 **git submodule** 形式引用。

---

## 一、目录构成

| 路径 | 角色 | 形态 | 说明 |
|---|---|---|---|
| **[`AV-SemanticVAD/`](AV-SemanticVAD/)** | **主项目** | 库内目录 | 计划、调研、代码（待实现） |
| [`X2-Turn/`](X2-Turn/) | **base 权重来源** | submodule | `yutian929/X2-Turn`（fork，跟踪 `X-Square-Robot/X2-Turn`） |
| [`SoulX-Duplug/`](SoulX-Duplug/) | **范式参考** | submodule | `yutian929/SoulX-Duplug`（fork，upstream `Soul-AILab/SoulX-Duplug`） |


**三者的关系**（详见 `AV-SemanticVAD/plan/architecture.md`）：

```
X2-Turn-4B-0812 权重  ──►  全部冻结，作为 base
                              （其 hidden 已含话轮表征，故不用裸 Voxtral）
SoulX-Duplug          ──►  提供「冻结前端 + 可训 Projector + LoRA」范式
                              （同任务的成功先例，含显式 complete/incomplete）
AV-SemanticVAD        ──►  在 base 上侧接一路视觉，输出逐帧 complete/incomplete
```

---

## 二、在 GPU 服务器上从零拉起

### 2.1 克隆（含子模块）

```bash
git clone --recurse-submodules https://github.com/yutian929/SemanticVAD.git
cd SemanticVAD
```

**已演练验证**：子模块会自动检出正确提交，克隆后约 **9.3 MB**（权重不在库内）。

若已经克隆但漏了 `--recurse-submodules`：

```bash
git submodule update --init --recursive
```

私有仓库的凭据配置见 [`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md) §1.1。

### 2.2 恢复 SoulX 的官方上游远程（含 `training-code` 分支）

子模块只记录 `origin`（fork）。**官方上游的 `training-code` 分支是 Phase 3 写 Trainer 的参考**，需手动加回：

```bash
cd SoulX-Duplug
git remote add upstream https://github.com/Soul-AILab/SoulX-Duplug.git
git fetch upstream
git branch -r | grep upstream    # 应看到 upstream/training-code
cd ..
```

上游三个分支：

| 分支 | 内容 | 我们的用途 |
|---|---|---|
| `main` | 推理服务 | 范式的结构参考（`model/model.py`） |
| **`training-code`** | **训练循环** | **Phase 3 自写 Trainer 的直接参考** |
| `dialogue-system` | 对话系统 | 备查 |

### 2.3 权重下载（**不在库内**，7.2 GB 已被 gitignore）

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

### 2.4 环境

X2-Turn 已自带安装脚本（2026-08-31 上游新增）：

```bash
cd X2-Turn && bash install.sh
```

详细步骤与 H20 相关配置见 [`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md)。

---

## 三、更新子模块到上游最新

```bash
# 拉取两个子模块各自 origin 的最新
git submodule update --remote --merge

# 确认后在父仓库提交新的子模块指针
git add X2-Turn SoulX-Duplug
git commit -m "chore: 更新子模块到上游最新"
```

**X2-Turn 同步真上游**（fork 落后 X-Square-Robot 时）：

```bash
cd X2-Turn
git remote add upstream https://github.com/X-Square-Robot/X2-Turn.git   # 首次
git fetch upstream && git merge upstream/main
```

---

## 四、当前状态（2026-09-04）

| 项 | 状态 |
|---|---|
| 调研 | ✅ 完成（`AV-SemanticVAD/research/`） |
| 架构设计 | ✅ v4（`plan/architecture.md`） |
| 实施计划 | ✅ v3.1（`plan/implementation-plan.md`） |
| **代码实现** | ⬜ **未开始** —— 目录骨架已建，从 Phase 0 起步 |
| 算力 | **双卡 H20（96 GB/卡）** |

**子模块基线版本**：

| 子模块 | 提交 | 日期 |
|---|---|---|
| X2-Turn | `53d3b9a` | 2026-08-31 |
| SoulX-Duplug | `45bd237` | 2026-08-24 |

---

## 五、下一步

按 [`AV-SemanticVAD/plan/implementation-plan.md`](AV-SemanticVAD/plan/implementation-plan.md) 的「立即执行的三件事」：

1. **W1 D1**：提交伦理审批 + 核对 MM-F2F 再分发条款（最长串行依赖）
2. **W1**：配环境、载 X2-Turn 权重、查 V1/V2/V3
3. **W2**：跑**探针 A** 与**探针 B** —— 两个 AUC 数字决定整篇论文的 framing

> ⚠️ **不要跳过探针直接进入训练。** 现有全部证据都指向「视觉增益来自噪声鲁棒性，
> 不来自语义判断」。若探针 A 显示 X2-Turn hidden 的线性探针已达 0.90+，
> 必须先切换论文 framing 再投入训练。
