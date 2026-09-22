# 模型架构 v6 —— forward 契约（输入 · 中间流程 · 输出）

> **本文件 = 模型契约的唯一权威。** 对外定位（new setting / 三点贡献 / 竞品口径）以
> [`cvpr-framing.md`](cvpr-framing.md) 为准，本文件不重复、也不得与之冲突。
> 旧的 v4/v5 架构文档、旧实施计划、旧 benchmark 提案已于 2026-09-21 **删除**（可从 git 历史取回）。
>
> **骨干已定** ✅ = **X2-Turn（Voxtral 流式版）**，Route A 热启动（cvpr-framing §8）。
>
> 标记：✅ 已定 ｜ 💡 建议待拍板 ｜ ⏳ 待实验定 ｜ ⚠️ 风险
>
> **最后更新**：2026-09-21

---

## 0. 一页纸总览

```
                        ┌─── 前置冻结模块（不参与梯度）───┐
  单目 RGB (多人同框) ──►│  人脸检测 + 跟踪 → K_t 条轨迹   │──► 每脸: 脸/嘴 crop + 头姿/注视
                        └──────────────────────────────┘
                                    │
                                    ▼  每脸每 80ms 一组 visual token（投影到 3072）
                          ┌──────────────────────┐
                          │  视觉塔（共享权重）   │
                          └──────────┬───────────┘
                                     │ 零初始化门控 cross-attn，逐脸注入
  多人混合音频 16kHz ──►[Whisper enc]──►[adapter 4×]──►┌──────────────────────────┐
                                    12.5Hz / 80ms      │  Voxtral LLM 26 层 d=3072 │
                                    (热启动·因果)       │  batch 维 = K_t 张脸      │
                                                       └────────────┬─────────────┘
                                                                    ▼  H_k[t]  每脸独立 hidden
                                           ┌────────────────────────────────────────┐
                                           │ forward 只吐 logits（每脸 × 每帧 × 3 组）│
                                           │  asr_logits   [B,K,T,131072]           │
                                           │  state_logits [B,K,T,3]                │
                                           │  addr_logits  [B,K,T,2]                │
                                           └────────────────────────────────────────┘
                                                        ← 模型到此为止；解码属系统层（§5）
```

---

## 1. 输入契约 ✅

### 1.1 音频

| 项 | 值 | 依据 |
|---|---|---|
| 形态 | **多人混合**单通道波形，16 kHz | cvpr-framing §1（不用麦阵 / DoA） |
| 前端 | Whisper-large-v3 编码器，`num_mel_bins=128`，`hidden=1280`，32 层 | `config.json: audio_config` |
| 降采样 | adapter `downsample_factor=4` → LLM token 流 | `config.json: downsample_factor` |
| **帧率** | **12.5 Hz = 80 ms / token** | `audio_length_per_tok=8` × `hop_length=160` ÷ 16000 = 80 ms（`inference.py:118-119` 就是这样算的） |
| LLM 维度 | `hidden_size=3072`，26 层，`vocab=131072` | `config.json: text_config` |
| **固有前瞻延迟** | `default_num_delay_tokens=6` → **480 ms** | `inference.py:122`；⚠️ 见 §6.2，视觉必须对齐同一延迟约定 |

### 1.2 视频

| 项 | 值 |
|---|---|
| 形态 | **单目 RGB**，多人同框；假设人一定在画面内 |
| 前置模块 | **人脸检测 + 跟踪**（冻结、不训）→ 第 t 帧给出 `K_t` 条稳定轨迹 |
| 每条轨迹产出 | ① 脸/嘴 ROI crop；② **头姿 + 注视**低维向量（addressee 轴必需，探针 B 用过 FaceLandmarker 的 3 维头姿） |
| 时间对齐 | 视频帧率重采样到 **12.5 Hz**（25 fps → 每 80 ms 格取末帧；50 fps → 池化） |
| `K_t` | **变长**，允许进出画面；`K_t = 0` 时模型退化为纯音频骨干（§6.3） |

**为什么 crop 而不是整帧**：读出是 per-face 的，视觉塔必须拿到"这张脸"的像素。
v5 曾写"整帧 RGB 编码器内部检测"，此处**改为前置显式检测跟踪** —— 理由：
(a) omni 类整帧 ViT 给不了 per-face token（cvpr-framing §8.3 第 5 条）；
(b) 检测跟踪做成冻结前置模块可插拔、可单独评测、不吃训练预算。

---

## 2. 中间流程

### 2.1 视觉塔 💡（选型待拍板，接口已定）

```
crop_k[t]  ─►[视觉编码器(预训练唇动/脸动塔)]─► e_k[t] ∈ R^d_v
pose_k[t]  ─►[小 MLP]──────────────────────► g_k[t] ∈ R^d_g
                       concat → Linear(d_v+d_g → 3072) → v_k[t]   （每脸每 80ms 一个 token）
```

- 编码器候选：**AV-HuBERT 视觉前端**（唇动，96×96 ROI，自监督预训练）vs **Light-ASD / TalkNet 骨干**（ASD 任务预训练，天然擅长"谁在说"）。💡 **建议 AV-HuBERT 前端**——它输出的是序列表征而非二分类 logit，更适合当 cross-attn 的 query 源。
- 注视/头姿走**独立支路**且不能被唇动塔吞掉：addressee 轴只靠它（cvpr-framing §3.2）。
- 每脸每帧 **1 个 token** 起步；若容量不足再升到 n 个（n 是超参，不改契约）。

### 2.2 per-face 条件化 —— ★ 本架构的核心机制 💡

**面孔 k 的 visual token 序列，通过零初始化门控 cross-attention 注入 LLM。**

```
LLM 第 ℓ 层（每 N 层插一个，N≈4）:
    h ← h + tanh(α_ℓ) · CrossAttn(Q = h, K = V = v_k[≤t])      α_ℓ 初始化为 0
```

- **因果**：帧 t 只看 `v_k[≤t]`（与音频侧同一延迟约定，§6.2）。
- **零初始化**：`α=0` 时整个模型**逐比特等于纯 X2-Turn** → 这是 §6.1 必测的恒等性单测，也是"视觉失效可降级"的结构保证。
- **batch 维 = K_t 张脸**：同一段混合音频的 `input_features` 在 K 条流上完全相同，差别只在
  (a) 注入的 `v_k`，(b) 各自的 text 流。一次 batched forward 出 K 套 hidden `H_k[t]`。
- **新增参数**只有：视觉塔 + 门控 cross-attn + 三个头；骨干走 LoRA。

> **⚠️ 新颖性口径（不可犯）**：这个机制 = **AV-TSE 的表征空间版本**，机制**不新**
> （CueNet / Plug-and-Steer / USEF-TSE，cvpr-framing §7）。
> 论文里**绝不能**写"首个视觉引导 per-face 提取/路由"；卖点钉在**输出是 per-face 语义状态而非波形**。

### 2.3 ★ 唯一需要你拍板的岔路：A 还是 B

|  | **A · 共享单次前向 + 读出**（v5 原案） | **B · per-face 条件化流**（本文件建议） |
|---|---|---|
| 骨干前向次数 | **1×** | **K×**（batch 维，一次 batched forward） |
| 视觉进入位置 | 骨干**之后**，cross-attn 读出 `H[t]` | 骨干**之内**，逐层门控注入 |
| per-face ASR | ❌ 只有一条混合流的 ASR | ✅ 每脸一条 text 流 |
| per-face 完整性从哪读 | **混合 hidden** —— 正是噪声 gate 塌到 **0.47** 的那个东西 ⚠️ | 已被视觉条件化的 `H_k`，信息未被混合前向抹平 |
| 成本（K=4, 4B, 12.5Hz） | 1 份 | 4 份（batched，2×H20 可承） |

**💡 建议 = B 为主模型，A 作为消融臂。** 三条理由：

1. **A 给不了 per-face ASR**，而"重叠鲁棒的 per-speaker 转写"是 new setting 承诺的三大能力之一。
2. **A 的完整性读出恰好走在探针警告的那条路上**：混合前向的 `H[t]` 若已退化，cross-attn 只能**路由**、
   不能**无中生有**（phase2.md §2.2 的原话）。B 把视觉放进骨干**之前/之内**，让条件化影响表征本身。
3. **A 是 B 的严格子集**（把门控关掉、改成事后读出即得 A）→ 做 B **白送** A 的消融对比；
   反过来做 A 则永远补不出 B。这条对 C3 的"同构消融"要求直接有利。

**⏳ 由实验最终裁定**：重叠段恢复实验（cvpr-framing §5 backlog 7）应**同时**跑 A 和 B 两臂 ——
它既是 C2 强弱的判据，也是这条岔路的判据。若 B 相对 A 在重叠档没有显著增益，则 B 的 K× 成本不值，退回 A。

---

## 3. 输出契约 ✅（forward 只吐 logits）

对每张脸 `k ∈ [1..K_t]`、每个 80 ms 帧 `t`：

| 输出 | 形状 | 实现 | 热启动 |
|---|---|---|---|
| `asr_logits` | `[B, K, T, 131072]` | 复用 `base_model.lm_head` | ✅ 直接继承 X2-Turn |
| `state_logits` | `[B, K, T, 3]` | **新** `Linear(3072, 3, bias=False)` | ✅ 从 `vad_lm_head` 的第 **35 / 37 / 38** 行拷贝（见下） |
| `addr_logits` | `[B, K, T, 2]` | **新** `Linear(3072, 2, bias=False)` | ❌ 随机初始化（无对应先验） |

### 3.1 状态三态 ✅

`{0: 静默 · 1: 说话-incomplete · 2: 说话-complete}` —— active 与 completeness 天然耦合，单头即可。

**热启动技巧**：X2-Turn 的 `vad_lm_head` 是 `lm_head` 的整词表拷贝，只有 id 35–40 有意义
（`modeling.py:15-23`：`idle/noidle/speaking/turn_end/backchannel/uncertain`）。
我们的三态与其中三类语义对齐 → 用对应行初始化：

```
state_head.weight[0] ← vad_lm_head.weight[35]   # 静默         ← idle
state_head.weight[1] ← vad_lm_head.weight[37]   # 说话-incomplete ← speaking
state_head.weight[2] ← vad_lm_head.weight[38]   # 说话-complete   ← turn_end
```

→ day-0 就不是随机头。**探针 A 的 0.99 说明完整性在 hidden 里线性可读**，这个三态小头足够。

### 3.2 判别式，不走生成式 ✅（cvpr-framing §3.2 backlog 1 → 本文件拍板）

状态**不**占词表槽位、**不**进 text 流。依据：探针 A 线性可读 → 判别式够；低延迟、不 generate、
K 个头天然并行、评测直接出 AUC；视觉失效时可独立降级。
（X2-Turn 自己走的是"整词表 + 保留 id"的路子，那是为了 checkpoint 兼容标准 LM 工具链 ——
我们不再守冻结范式，没有这个约束。）

### 3.3 addressee ✅（提级进核心）

per-face 二分类 `{对系统说 · 不对系统说}`，与状态正交。
**⚠️ 与 v5 的冲突点**：v5 明确"删去 addressee"，cvpr-framing §1/§3.2 今天把它提级进核心 setting。
**本文件以 cvpr-framing 为准：addressee 是第 4 个 per-face 输出。**
数据代价（需第一人称/机器人视角语料，AMI 给不了）见 cvpr-framing §2 C1 的 ⚠️。

### 3.4 系统层解码（不属模型 forward）

`{state_logits 三态 softmax}` × `{addr_logits}` × `{asr_logits 自回归解出的文本}`
→ "该不该现在回应他"。阈值、迟滞、uncertain 兜底都在系统层，模型不管。

---

## 4. 训练契约

### 4.1 阶段
- **S1**：冻骨干（含音频编码器），只训 **视觉塔 + 门控 cross-attn + 三个头**。门控从 0 长起来。
  验收：归属/状态可读、ASR 不坏（恒等性单测仍过）。
- **S2**：骨干开 **LoRA r=32**（数据量足则全微调），一次前向联合训。

### 4.2 损失
```
L = L_asr + λ_s · Σ_k L_state(k) + λ_a · Σ_k L_addr(k)
```
- `L_state` 类加权：**incomplete 被误判成 complete（= 抢话）代价最高**。
- 视觉 dropout `p≈0.3`（整段置零）→ 把"视觉失效可降级"从结构保证升级成统计保证。

### 4.3 标签形态（数据集接口，C1 必须产出这个）
每段视频、每 80 ms 帧、每条人脸轨迹：
`{track_id, state ∈ {0,1,2}, addressee ∈ {0,1}, text（该脸该帧的转写 token）}`。
⏳ 数据方案（自建 vs 扩展 AVCocktail）见 cvpr-framing §5 backlog 5。

---

## 5. 必测单测 ✅

1. **恒等性**：所有门控 `α=0` 且无视觉 → 输出与纯 X2-Turn **逐比特相同**。
2. **帧对齐脉冲响应**：在第 n 帧注入脉冲，确认响应出现在第 n 帧而非 n±1。
   ⚠️ X2-Turn 的读出偏移是 `prediction_index = prefix_length + frame_index - 1`（`inference.py:165`，next-token 语义），写错全局错位 80 ms。
3. **零初始化**：新头/新门控在 step 0 的梯度与输出符合预期。
4. **变长 K**：`K=0 / 1 / 4 / 8` 与人脸中途进出，形状与 mask 正确。

---

## 6. 已知风险与边界

### 6.1 重叠段 ⏳
两人重度重叠时混音表征退化（噪声 gate 0.47）。B 路线**假设**条件化能在表征层把目标捞回，
这是**经验问题**，由恢复实验量化。低置信 → 系统层输出 uncertain。

### 6.2 ⚠️ 延迟对齐（最容易写错的地方）
X2-Turn 默认 `num_delay_tokens=6` = **480 ms** 前瞻：text 流位置 `p` 对应的音频是 `p-6` 帧。
视觉 token 注入时必须**换算到同一时间基**，否则视听错位 480 ms 且不会报错、只会静默掉点。
实现时以 `inference.py:133-135` 的 `prefix_length / frame_count` 为准。

### 6.3 降级
`K_t = 0`、黑暗、无脸、跟踪丢失 → 该脸的 `v_k` 置零 → 门控路径无贡献 → 退化为纯音频骨干行为。

### 6.4 成本
B 路线是 K× 骨干计算。K=4、4B、batched decode @12.5 Hz 在 2×H20 上可行；
`K≥8` 需要评估是否降级为"状态头跑全部 K 脸（共享前向），ASR 流只对 active 脸起"的混合策略
（= cvpr-framing §3.2 的"读出/ASR 解耦"，作为**部署期优化**而非论文主模型）。

---

## 7. 相对 v5 的变化清单

| 项 | v5 | v6 | 原因 |
|---|---|---|---|
| addressee | 删去 | **第 4 个 per-face 输出** | cvpr-framing §1 提级 |
| 视觉输入 | 整帧 RGB，编码器内部检测 | **前置冻结检测跟踪 → per-face crop** | 整帧 ViT 给不了 per-face token |
| 视觉进入位置 | 骨干**之后** cross-attn 读出 | **骨干之内**逐层零初始化门控注入（B）；A 作消融 | A 的完整性读自混合 hidden，撞 0.47 |
| 状态读出 | 判别式 / 生成式未定 | **判别式**，且从 `vad_lm_head` 行热启动 | 探针 A 线性可读 |
| 粒度 | K 套 vs 单流未定 | **K 套（batch 维）**，ASR 解耦作部署优化 | per-face ASR 是 setting 承诺的能力 |
| 骨干 | Voxtral/X2-Turn 未细化 | **X2-Turn-4B-0812**，参数已核 | cvpr-framing §8 |

---

## 8. 仍待决 ⏳

1. **A / B 岔路**的最终裁定 → 重叠段恢复实验（两臂同跑）。
2. **视觉编码器选型**：AV-HuBERT 前端 vs Light-ASD/TalkNet；检测跟踪器选型。
3. 每脸每帧 visual token 数 `n`、门控插入间隔 `N`。
4. LoRA r=32 vs 全微调 —— 等数据量定。
5. 数据集方案与 addressee 标签定义（cvpr-framing §5 backlog 5 / 9）。

---

## 附录 · 复用的代码锚点

前缀 `X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/`，只读；详见 [`../docs/code-anchors.md`](../docs/code-anchors.md)。

- `modeling.py:15-23` `TURN_CLASS_IDS/NAMES` —— §3.1 热启动取的就是这里的 35/37/38。
- `modeling.py:37-188` `VoxtralMTP` —— 共享骨干 + 多头范式，我们推广为 per-face 多头。
- `modeling.py:60-69` —— `vad_lm_head` 拷贝 `lm_head` 的手法。
- `modeling.py:125-145` `train_vad_head_only` —— S1"冻骨干只训新头"的现成实现。
- `inference.py:86` `_predict_turn` —— 判别式读出的写法。
- `inference.py:118-122` —— 80 ms 帧长与 480 ms 延迟的计算（§1.1 / §6.2）。
- `inference.py:165` —— `prediction_index = prefix_length + frame_index - 1` 帧对齐偏移。
