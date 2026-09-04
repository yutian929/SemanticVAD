# GPU 服务器部署清单（双卡 H20）

> 目标：在有卡的服务器上把 Phase 0 跑起来。
> 对应 [`../plan/implementation-plan.md`](../plan/implementation-plan.md) §Phase 0。

---

## 0. 算力前提与由此确定的配置

| 项 | 值 |
|---|---|
| GPU | **2 × H20，96 GB/卡，共 192 GB** |
| 精度 | bf16（H20 原生支持） |

### 显存预算（单卡即可完成全部训练）

| 组成 | 估算 |
|---|---|
| 冻结 X2-Turn-4B bf16 前向 | ≈ 8 GB |
| LoRA r=32 参数 + 优化器状态 | ≈ 4 GB |
| VisualProjector + Encoder + 优化器 | < 1 GB |
| 激活（batch 8，含梯度检查点） | ≈ 3 GB |
| **合计** | **≈ 16 GB / 96 GB** |

### ★ 由算力宽裕度确定的三项决策

| # | 决策 | 原因 |
|---|---|---|
| 1 | **LoRA 直接锁 `r=32, α=64`** | 不再需要「按数据量升降档」的条件判断（原计划受单卡 24 GB 限制） |
| 2 | **新增臂 C**（相加 + LoRA + **解冻 `vad_lm_head`**） | 作为上界对照，成本极低（只多解冻 H×6 小头） |
| 3 | **噪声增广改为训练时即时合成** | 省磁盘，且允许更多 augmentation 变体 |

### 两卡怎么用

**首选：不引入 DDP。**

```
GPU 0  ──  训练主进程
GPU 1  ──  并行跑探针 / 评测 / 数据流水线
```

理由：batch=8 单卡显存占用仅 16 GB，DDP 带来的复杂度换不到收益。
仅当 batch 需推到 32+（例如音频 context 很长）时才启用：

```bash
torchrun --nproc_per_node 2 -m avsvad.train.cli ...
```

HF `Trainer` 会自动处理，无需改代码。

---

## 1. 克隆

```bash
git clone https://github.com/yutian929/SemanticVAD.git
cd SemanticVAD
```

**一条命令即可 —— 没有子模块。** 三份第三方代码（`X2-Turn/`、`SoulX-Duplug/`、
`SoulX-Duplug-training/`）已直接纳入仓库，克隆后约 12 MB（权重不在库内）。

溯源信息（来源 SHA / 许可 / 只读约定）见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md)。

### 1.1 私有仓库的凭据

仓库若为 private，服务器上需配一次凭据（**只需配这一个仓库**）。
**推荐用 PAT + credential store**，避免每次输入：

```bash
git config --global credential.helper store
# 首次 clone 时输入 GitHub 用户名 + Personal Access Token（不是密码）
# PAT 生成: https://github.com/settings/tokens  勾选 repo 权限
```

或用 SSH（需先把服务器公钥加到 GitHub）：

```bash
ssh-keygen -t ed25519 -C "h20-server"
cat ~/.ssh/id_ed25519.pub     # 复制到 https://github.com/settings/keys
# 然后把 URL 换成 SSH
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

### 1.2 三个第三方目录一律只读

| 目录 | 内容 | 版本 |
|---|---|---|
| `X2-Turn/` | base 权重来源 | `@53d3b9a` (2026-08-31) |
| `SoulX-Duplug/` | 范式参考（**推理服务**） | `main @45bd237` |
| **`SoulX-Duplug-training/`** | **训练代码参考** | `training-code @928b065` |

> ⚠️ **后两者不是包含关系，是互补的两棵树。**
> 上游把推理和训练放在不同分支上：`main` 有 `service/model.py`（含 complete/incomplete 判决），
> `training-code` 删掉了它但加入了 `finetune.py`、`utils/ema/` 等训练工具。

我们的改造走**包装**不 fork：新增代码全在 `AV-SemanticVAD/avsvad/` 下，
经 `avsvad/upstream.py` 引用上游模块。**不要在上述三个目录内改代码**，
否则将无法在论文里声明「未修改基座代码」，也无法与上游做 diff。

对比/更新上游的方法见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §4。

---

## 2. 环境

### 2.1 X2-Turn 自带脚本（推荐先试）

```bash
cd X2-Turn && bash install.sh
```

该脚本为上游 2026-08-31 新增（237 行），会处理依赖安装。

### 2.2 若需手工建环境

```bash
conda create -n avsvad python=3.11 -y
conda activate avsvad

# PyTorch —— 按服务器实际 CUDA 版本选择，先确认：
nvidia-smi | head -3
# H20 需 CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# X2-Turn 依赖
cd X2-Turn && pip install -e . && cd ..

# 我们新增的依赖
pip install peft transformers accelerate datasets \
            mediapipe opencv-python \
            scikit-learn pandas pyarrow \
            librosa soundfile \
            wandb
```

> ⚠️ `mediapipe` 在部分服务器发行版上需 `libgl1`：
> `apt-get install -y libgl1 libglib2.0-0`

---

## 3. 权重与数据（均不在 git 内）

```bash
export HF_HOME=/path/to/large/disk/hf_cache   # 建议指向大盘

# ① base 权重（必需）
huggingface-cli download x-square-robot/X2-Turn-4B-0812 \
    --local-dir X2-Turn/models/X2-Turn-4B-0812

# ② 跨域回归检查用评测集（Phase 4）
huggingface-cli download Soul-AILab/SoulX-Duplug-Eval --repo-type dataset \
    --local-dir data/SoulX-Duplug-Eval

# ③ 范式参考权重（可选）
huggingface-cli download Soul-AILab/SoulX-Duplug-0.6B \
    --local-dir SoulX-Duplug/pretrained_models/SoulX-Duplug-0.6B

# ④ 主训练语料 MM-F2F —— ⚠️ 先核对再分发条款（计划单 §1.1.1）
git clone https://github.com/Linyx1125/MM-F2F
```

**磁盘估算**：X2-Turn 权重 ~8 GB ＋ MM-F2F 210 h 视听语料（数百 GB）＋ 中间特征。
**建议预留 ≥1 TB。**

---

## 4. Phase 0 验收清单

对应计划单 §Phase 0，**这是服务器上要做的第一批事**。

### 4.1 环境与基线复现

| ☐ | 任务 | 验收 |
|---|---|---|
| ☐ | 0.1.1 环境配置 | `X2-Turn/turn-demo` 跑通 |
| ☐ | 0.1.2 加载 base | bf16 载入，显存 ≈8 GB |
| ☐ | 0.1.3 复现帧级输出 | `infer_asr_turn` 输出 80 ms/帧；3.4 s 音频 ≈53 帧 |
| ☐ | 0.1.4 固化 golden | `tests/golden/backbone.json` |

### 4.2 解锁待核实项（**纯读取，最先做**）

| ☐ | 项 | 方法 |
|---|---|---|
| ☐ | **V1** `hidden_size`/`vocab_size`/层数 | 读 `config.json` |
| ☐ | **V2** id 41+ 是否空闲 | 查 Tekken 词表 41–50 |
| ☐ | **V3** decoder layer 0 模块路径 | `print(model)` |
| ☐ | **V7** turn head 是否吃 delay tokens | 对比不同 `delay_ms` 下 turn 帧时序 |

### 4.3 ★ 两个探针（决定论文 framing）

| ☐ | 探针 | 判据 |
|---|---|---|
| ☐ | **A**：X2-Turn hidden 线性探针 → complete/incomplete | **>0.85 → 必须改走分层汇报**；0.70–0.85 按主线；<0.70 考虑加大 LoRA |
| ☐ | **B**：MediaPipe 24 维 → 小 GRU → complete/incomplete | **>0.62 核心卖点**；0.55–0.62 定位噪声鲁棒；≈0.50 转 negative result |

**Gate（W2）**：两个 AUC 数字出炉（无论结果如何）；V1/V2/V3 有明确答案。

> **不要跳过探针直接训练。** 若探针 A 已达 0.90+，视觉可捞空间极小
> （参照 Kurata'23 的 +3.0 是建立在 wav2vec2-base 弱基线上的，见 `plan/architecture.md` §B.1）。

---

## 5. 关键代码锚点（服务器上要读的文件）

### X2-Turn（base）

| 内容 | 位置 |
|---|---|
| `VoxtralMTP` 双头结构 | `X2-Turn/.../transformers/modeling.py:15-69,114-188` |
| 检查点加载 | 同文件 `:228-264` |
| `TURN_CLASS_IDS`（id 35–40） | `X2-Turn/.../inference.py:86` |
| hidden 读取偏移 `prefix_length + i − 1` | `X2-Turn/.../inference.py:165` |
| 冻结开关 `train_vad_head_only` | `modeling.py` |
| 规则控制器注入点 | `server.py:102, 115-120` |

### SoulX-Duplug（范式 · 推理侧）

| 内容 | 位置 |
|---|---|
| `EncoderProjector` | `SoulX-Duplug/model/model.py:17-35` |
| `token_samples = int(0.08 * 16000)` | `model/model.py:50` |
| WhisperVQ 冻结加载 | `model/model.py:53-58` |
| projector freeze 开关 | `model/model.py:60-71` |
| `forward()` audio/text embedding 混合 | `model/model.py:145-166` |
| **complete/incomplete 判决** | **`service/model.py:708-733`** |
| 状态 token id | `config/config.py:27-35` |
| 远场 RMS 门控 | `service/model.py:252-256` |

### ★ SoulX-Duplug-training（范式 · 训练侧，Phase 3 必读）

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| **训练入口** | `SoulX-Duplug-training/finetune.py` | 同范式同任务的训练循环，写 Trainer 前必读 |
| **数据格式样例** | `SoulX-Duplug-training/example_data_fisher.jsonl` | **官方 jsonl 格式**，Phase 1.6 字段设计应对照它，降低返工 |
| 启动参数 | `launch.sh` | lr / batch / 阶段划分的实际取值 |
| EMA 实现 | `utils/ema/`（三种：原生 / lightning / nemo） | 若需 EMA 可直接借 |
| 数据调度 | `utils/epoch_shuffle.py`、`utils/dynamic_train.py` | 变长样本的组批策略 |
| 学习率调度 | `utils/sparkvox/utils/scheduler.py` | — |
| 权重导出 | `scripts/export_weights.py` | Phase 3 出 checkpoint 时参考 |

> **Phase 3.5.1 的建议顺序**：先读 `example_data_fisher.jsonl` 定字段 →
> 再读 `finetune.py` 定训练循环 → 最后才动手写 `avsvad/train/trainer.py`。

---

## 6. 已知坑

| 坑 | 说明 |
|---|---|
| **帧对齐错 1 帧** | = 全局 80 ms 系统性偏差，且表现为「视觉略有帮助」，**极难察觉**。必须写脉冲响应单测（计划单 §2.7） |
| **`−1` 偏移** | `hidden_states[prefix_length + i − 1]`，next-token 语义，写错则全局错位（V4） |
| `mediapipe` 缺 `libgl1` | `apt-get install -y libgl1 libglib2.0-0` |
| **误改第三方目录** | 三个目录已删内层 `.git`，改了之后**不会有任何 git 提示**，且与上游 diff 会永久混入我们的改动。改造一律走 `avsvad/` 包装（§1.2） |
| **误以为 training 是 main 的子集** | 两者是**互补的两棵树**，`main` 有推理代码、`training-code` 有训练代码，缺一不可（§1.2） |
| HF 下载慢 | `export HF_ENDPOINT=https://hf-mirror.com` |
