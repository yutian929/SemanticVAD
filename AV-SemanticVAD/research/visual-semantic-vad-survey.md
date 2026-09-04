# 视觉引导的语义 VAD：深度调研报告

**主题**：是否存在能同时输入视频流（或人脸特征流）与音频流，判断说话人「话是否说完 / 语义是否完整 / 现在能不能接话」的模型？

**基线参照**：`X2-Turn`（arXiv:2608.10878）与 `SoulX-Duplug`（arXiv:2603.14877），当前纯音频语义 VAD 的公开 SOTA，均位于本仓库同级目录。

**检索方式**：arXiv API 结构化检索 + ar5iv/ACL Anthology/ISCA Archive 全文核验 + 工业方案调研 + 本地代码审计。方法论与可信度评级见 [`methodology.md`](methodology.md)。

---

## 0. 执行摘要（TL;DR）

### 0.1 直接回答

**存在「视频+音频 → 话轮边界判断」的模型，但不存在「视觉引导的 Semantic VAD」。**

这不是文字游戏，而是任务定义上的实质差异：

| | 现有视觉话轮工作的输出 | X2-Turn / SoulX-Duplug 的输出 |
|---|---|---|
| 标签 | `shift`/`hold`、`<SOT>`/`<SOB>`、`KEEP`/`TURN`/`BACKCHANNEL` | `complete` / `incomplete`（+`idle`/`nonidle`/`backchannel`） |
| 问的问题 | **"现在该谁说话 / 我该不该开口"** | **"他这句话说完了吗、语义闭合了吗"** |
| 决策依据 | 时序节奏 + 韵律 +（部分）词法 | **句法/语义完整性**为第一性依据 |
| 失败代价 | 抢话 / 冷场 | 打断未说完的句子 / 等待已说完的句子 |

**没有任何一篇工作把视觉特征接入一个逐帧输出「语义完整 / 语义不完整」判决的流式头。** 这是本方向真实存在的空白。

### 0.2 最接近的五项工作

按"接近程度"排序（详见 §3）：

1. **AV-Dialog**（arXiv:2511.11124，UW + Meta AI，2025-11）— **最接近**。首个视听输入的口语对话框架，摘要明确声称 *"semantically grounded turn-boundary detection"*。40ms chunk 流式，AV-HuBERT 唇部特征 + DAC 声学 token + Llama3-8B，双输出流（流式 AVSR + 话轮事件）。**但**其话轮标签是 `<SOT>`/`<SOB>`/`<EMP>`，没有 complete/incomplete 判决；且视觉增益几乎全部来自噪声鲁棒性（干净条件仅 +1.3%）。
2. **MM-F2F**（arXiv:2505.12654，ACL 2025，厦门大学）— **唯一同时具备语义（文本）与视觉模态**的话轮预测框架。GPT-2 + HuBERT + VideoMAE(人脸)，词帧级三分类。**且其 Limitations 明确承认在"语义不完整 + 停顿思考"场景失败** —— 这正是语义 VAD 的核心难例。
3. **MM-VAP 系列**（O'Connor Russell & Harte，ACL Findings 2025 / Interspeech 2025）— 多模态 VAP，量化了视觉对 hold/shift 的贡献（79%→83~84%）与噪声下的救援作用（52%→72%）。**无语义层建模**。
4. **MiniCPM-o 4.5 / Omni-Flow**（arXiv:2604.27393，OpenBMB）— **架构上**视觉 token 参与 `[listen]/[speak]` 控制决策，并明确声称"减少对外部 VAD 模块的依赖"。**但**时间粒度 1.0s（比 SoulX 粗 6.25 倍、比 X2-Turn 粗 12.5 倍），且全双工评测只在**无音频**的 LiveSports-3K-CC 上做，视觉引导的话轮能力从未被评测。
5. **Kurata et al.**（Interspeech 2023，早稻田大学）— 标题即"利用视觉线索进行**话语结束预测**"，三模态（视觉+声学+言语），AUC 0.896→0.920。**这是最早、也是标题最贴题的工作**，但规模小、非流式 LLM 范式、且被 MM-F2F 复现为仅 0.720 准确率。

### 0.3 最重要的一个反直觉发现

**在干净音频条件下，视觉对话轮判断的增益接近于零；在部分评测中甚至是负的。**

| 证据 | 结论 |
|---|---|
| AV-Dialog Table 3 | 干净条件 73.2% → 74.5%，**仅 +1.3%**；噪声 +8.1%；干扰说话人 +13.0% |
| MM-F2F Table 3 | Text+Audio 0.811 → +Video 0.823，**仅 +1.2%**；Video 单模态仅 0.559 |
| MM-VAP (Interspeech'25) | 干净 84%，10dB 音乐噪声崩到 52%，多模态救回 72% |
| **VideoFDB**（NVIDIA，arXiv:2605.30256） | **加入视频流使 7 个系统中 6 个的时序对齐率下降 0–5 个百分点**；MiniCPM-o 4.5 的 Conversational Flow 在 AV 模式下 3.54，**低于**纯音频模式 3.76 |

**解读**：视觉在现有范式里的作用是 **"目标说话人跟踪 / 噪声鲁棒性 / 附和检测"**，而**不是**"语义完整性判断"。视觉信息（唇动、表情、注视）本身不携带句法完整性信息——它携带的是"谁在说""说得多用力""有没有情绪"。

**这对本项目意味着两件事**：
- **风险**：如果只是把视觉特征拼到 X2-Turn 上，在干净近场音频的评测集上很可能得到 +0~1% 的增益，不足以支撑一篇论文或一个产品卖点。
- **机会**：真正的科学问题是 **"视觉能否提供音频与文本都无法提供的、关于说话人 *打算继续说* 的意图证据"** —— 即 turn-holding intent（思考型停顿时的注视回避、手势保持、口型未闭合）。这在 §5 展开。

---

## 1. 术语与任务定义：先把四个任务分清楚

本领域文献混用术语极为严重，是导致"以为有人做过 / 以为没人做过"两种误判的根源。必须先厘清。

| # | 任务 | 输入 | 输出 | 决策依据 | 代表工作 |
|---|---|---|---|---|---|
| T1 | **VAD**（Voice Activity Detection） | 音频帧 | 有声 / 无声 | 能量、频谱 | WebRTC-VAD, Silero-VAD |
| T2 | **EPD / Endpointing** | 音频流 | 说话结束点 | 静音时长阈值 + 声学 | 传统 ASR endpointer |
| T3 | **语义 VAD / 语义端点判定** | 音频流（+隐式文本） | **语义完整 / 不完整** | **句法-语义闭合性** | **SoulX-Duplug, X2-Turn**, Smart Turn, TEN Turn Detection |
| T4 | **PTTM / 预测性话轮模型** | 双方音频（+视觉） | 未来若干窗口的发声概率 → hold/shift | 时序节奏 + 韵律 + 视觉 | VAP, TurnGPT, MM-VAP, MuVAP |

**关键区分**：

- **T3 vs T4 是两个不同的问题。** T4 回答"接下来谁说话"，T3 回答"这句话的语义单元闭合了没有"。一个语义完整的句子后面可能仍然是同一说话人继续（T3=complete, T4=hold）；一个语义不完整的句子中间也可能被合法插入附和（T3=incomplete, T4=backchannel）。
- **几乎所有"视觉+话轮"文献都在做 T4，不是 T3。** 这是本报告最核心的判断。视觉话轮预测（MM-VAP、MuVAP、Kurata、Cano 等）全部是 T4 范式；MM-F2F 是 T4 的分类式变体（KEEP/TURN/BC）。
- **AV-Dialog 是唯一横跨 T3/T4 表述的工作**：它说自己做 "semantically grounded turn-boundary detection"，其"语义接地"来自于**同时训练流式 AVSR 流**（模型内部有转写表征），但输出侧仍是 T4 的 `<SOT>` 事件，**没有 T3 的 complete/incomplete 判决头**。

> **本项目的定位应表述为**：把 T3（语义完整性判定）从"仅音频"扩展到"视听"，而不是把 T4 加上视觉（那已经被做过多次）。

---

## 2. 音频侧 SOTA 基线审计（本地代码 + 论文）

### 2.1 SoulX-Duplug — 首个把 complete/incomplete 显式 token 化的语义 VAD

- 论文：arXiv:2603.14877，Soul AI Lab + 上交 X-LANCE + 西工大，Apache-2.0
- 模型：`Soul-AILab/SoulX-Duplug-0.6B`，自我定位为 *"plug-and-play streaming semantic VAD model"*，text-guided streaming state prediction
- 评测集：`Soul-AILab/SoulX-Duplug-Eval`（HF）；训练代码在 `training-code` 分支

**架构（本地代码核实）**

```
音频 16kHz
  → GLM-4-Voice WhisperVQ tokenizer（冻结）  model/model.py:53-58
  → codebook embedding                        model/model.py:152
  → EncoderProjector: audio_dim→2048→2048→1024（3 层 MLP + ReLU）  model/model.py:17-35
  → 与 text embedding 按 audio_mask 混合       model/model.py:159-160
  → Qwen3-0.6B-expand_vocab_v2 (llm_dim=1024) + LoRA(r=32, α=64)
  → next-token prediction，状态即特殊 token
```

**时间粒度（关键，视觉接入必须对齐这里）**

| 参数 | 值 | 含义 | 位置 |
|---|---|---|---|
| `token_samples` | 1280 samples | **80 ms / audio token** | `model/model.py:50` |
| `chunk_size` | 2560 samples | **160 ms / chunk = 2 audio tokens** | `config/config.yaml:30` |
| `audio_back_size` | 15360 samples | 960 ms 回看上下文 | `config/config.yaml:31` |
| `audio_ahead_size` | 640 samples | **40 ms 前视** | `config/config.yaml:32` |

**状态 token 表（`config/config.py:27-35`）**

| Token | ID |
|---|---|
| `<\|asr_bos\|>` | 151675 |
| `<\|user_complete\|>` | 151676 |
| `<\|user_backchannel\|>` | 151677 |
| `<\|user_incomplete\|>` | 151678 |
| `<\|assistant_backchannel\|>` | 151679 |
| `<\|user_idle\|>` | 151680 |
| `<\|user_nonidle\|>` | 151681 |
| `<\|assistant_interrupt\|>` | 151682 |

**complete/incomplete 判决逻辑（`service/model.py:708-733`）**

判决**不是每帧都做**，而是在两个触发条件下做：
1. 状态从 `<|user_nonidle|>` 跌回 `<|user_idle|>`（即检测到静音起点）；或
2. `mistake_len >= max_mistake_num`（默认 3）

触发后比较两个 logit：

```python
if complete_logit > incomplete_logit + complete_bias:   # complete_bias 默认 1.0
    state = "<|user_complete|>"
else:
    state = "<|user_incomplete|>"
```

**工程兜底机制**（这些恰好是视觉最可能替代的部分）：

| 机制 | 默认值 | 作用 | 位置 |
|---|---|---|---|
| `complete_bias` | 1.0 | 决策边界偏置，>0 更保守 | `config/config.yaml:25` |
| `max_wait_num` | 10 → **1.6 s** | 判 incomplete 后的耐心；超时强制接话 | `config/config.yaml:20` |
| `max_mistake_num` | 3 | nonidle 但无 ASR 输出的容错 | `config/config.yaml:21` |
| `far_field_threshold` | 0.02 (RMS) | 远场/噪声过滤 | `service/model.py:252` |

> **重要观察**：`max_wait_num = 1.6 s` 这个硬超时，本质上是"模型判 incomplete 但不确定要等多久"的补丁。**这正是视觉信息最有价值的落点**——如果能从人脸看出"他在思考、还要继续说"，就可以动态延长；如果看出"他已经闭嘴看着我等回应"，就可以立刻接话，而不必死等 1.6 s。这是本项目最具体、最可验证的价值假设。

对外接口状态被折叠为 4 个：`idle` / `nonidle` / `speak` / `blank`（`README.md:139-145`）。ASR 由外部模型提供（Paraformer 中文 / SenseVoice 英文双语）。

### 2.2 X2-Turn — 帧同步双头，80 ms 粒度

- 论文：arXiv:2608.10878，X Square Robot，Apache-2.0
- 模型：`x-square-robot/X2-Turn-4B-0812`
- **基座：Mistral `Voxtral-Mini-4B-Realtime-2602`**（README 致谢段明确说明）
- 定位：*Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction*

**输入输出**

- 输入：**仅音频**（全仓库无任何视觉/视频/人脸相关代码路径）
- 输出：转写 + **每 80 ms 一个话轮状态**
- 状态集（6 类，`README_zh.md:26-27`）：`idle` / `noidle` / `speaking` / `turn_end` / `backchannel` / `uncertain`

**推理接口**

```python
from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint
model = load_mtp_checkpoint(model_id, device="cuda", dtype=torch.bfloat16).eval()
result = infer_asr_turn(model, processor, "sample_en.wav")
result.transcript                       # str
result.turn_frames[i].start_ms/.end_ms/.label/.confidence
```

`load_mtp_checkpoint` 中的 MTP = multi-token prediction，即双头（ASR 头 + turn 头）通过多 token 预测在同一前向中输出。3.4 s 音频 ≈ 53 帧（`README_zh.md:112-116`）。生产部署走 patched vLLM，发出自定义 `turn.delta` 事件。要求 ≥24 GB 显存。

### 2.3 两者对比与"视觉接入难度"评估

| 维度 | SoulX-Duplug | X2-Turn |
|---|---|---|
| 参数量 | 0.6 B | 4 B |
| 骨干 | Qwen3-0.6B + LoRA | Voxtral-Mini-4B-Realtime |
| 音频前端 | GLM-4-Voice WhisperVQ（离散） | Voxtral 原生（Whisper 系） |
| 输出粒度 | 160 ms chunk（含 2×80 ms token） | **80 ms 帧** |
| complete/incomplete | ✅ 显式 token + logit 比较 | ⚠️ 折叠为 `turn_end` / `uncertain` |
| ASR | 外挂 | 内置（联合训练） |
| 前视 | 40 ms | 未公开声明 |
| 训练代码 | ✅ `training-code` 分支 | ⚠️ 仓库主要是推理/Demo |
| **加视觉分支的成本** | **低**（0.6B，projector 结构清晰，训练代码可得） | 高（4B，需 patch Voxtral + vLLM） |

**结论：视觉模态的首轮验证应在 SoulX-Duplug 上做，而非 X2-Turn。** 理由：
1. 参数量小 6.7 倍，单卡可训；
2. `EncoderProjector`（`model/model.py:17-35`）是一个独立的 3 层 MLP，**可原样复制为 `VisualProjector`**；
3. `forward()`（`model/model.py:145-166`）的 embedding 混合逻辑是简单的 mask 加权，加第三路 mask 是十几行改动；
4. 训练代码公开，可拿到 dataset/collator 与 per-token loss weight（`config/config.py:118-131` 已为每个状态 token 预留独立 `*_loss_rate`）；
5. 有配套评测集 `SoulX-Duplug-Eval`，可直接做"音频-only vs 视听"的同集对照。

X2-Turn 应作为**第二阶段**目标（证明方法在 80 ms 粒度、4B 规模上同样有效）。

---

## 3. 视觉引导话轮/端点判断的工作全景

按谱系分为 6 条线。**只有第 3、4 条线触及"语义"**。

### 3.1 谱系 A：心理语言学基础（视觉是否真的携带话轮结束信息？）

这条线回答"值不值得做"的先验问题。

- **Barkhuysen, Krahmer & Swerts (2008)**, *The interplay between the auditory and visual modality for end-of-utterance detection*, JASA。经典结论：人类判断话语结束时，**视听 > 仅听 > 仅看**；视觉单独有效但弱于听觉。这与 60 年后 AV-Dialog（V-only 72.5% < A-only 73.2% < A+V 74.5%）和 MM-F2F（Video 0.559 < Audio 0.751 < T+A+V 0.823）的机器学习结果**高度一致**。
- **Nota, Trujillo & Holler (2023)**, 眉毛动作（frowns）与问句/话轮转换的关联。
- **Best, Boyd & Sen (2023)**, *An effect of gaze direction in cocktail party listening*, Trends in Hearing。AV-Dialog 引用其作为"注视辅助鸡尾酒会问题"的依据。
- **Kendon (1967)** 起的注视-话轮经典结论：说话人在话轮末尾倾向于把注视投向听者。

**判断**：视觉携带话轮结束信息，这一点有跨越 50 年的稳健证据。但**证据强度是"弱信号、辅助性"**，而非"决定性"。且这些研究测的是 T4（话轮转换），不是 T3（语义完整性）。

### 3.2 谱系 B：多模态 VAP / PTTM（最成熟，但无语义层）

VAP（Voice Activity Projection，Ekstedt & Skantze）范式：预测未来若干时间窗内双方的发声二值模式，再映射为 hold/shift。

| 工作 | 年份/出处 | 视觉特征 | 关键量化结果 |
|---|---|---|---|
| **Kurata, Saeki, Fujie, Matsuyama** | Interspeech 2023 | 注视/嘴部/头部 → **3D-CNN 端到端** | 话语结束预测 **AUC 0.896 → 0.920**；消融显示**眼动贡献 > 嘴部 > 头部** |
| **MM-VAP**（O'Connor Russell & Harte） | Findings of ACL 2025（2025.findings-acl.12）, arXiv:2505.21043 | 面部特征（表情贡献最大） | 互静默期 hold/shift 平衡准确率 **79% → 83~84%**；代码开源 |
| **Visual Cues Support Robust Turn-taking Prediction in Noise**（同作者） | Interspeech 2025 pp.1073-1077, arXiv:2505.22088 | 同上 | **干净 84% → 10dB 音乐噪声 52%**；含噪训练的多模态 PTTM 恢复至 **72%**；在所有噪声类型与 SNR 下优于纯音频，**但不总能泛化到新噪声类型** |
| **MuVAP**（Qi & Skantze） | arXiv:2606.16731 | 人脸轨迹接地，多人场景 | 多方 VAP + 配套视听对话语料 |
| **MM-VAP**（Cano et al.，同名不同工作） | RO-MAN 2026, arXiv:2607.07294 | 视听 VAP | 人机交互场景 |
| **Saga & Pelachaud** | arXiv:2506.03980 | 音频 + 人脸编码器 | 开源 |
| **Gaze-Enhanced Triadic** | Interspeech 2025, arXiv:2505.13688（Meta Reality Labs） | **注视（智能眼镜第一视角）** | 三人对话下一说话人预测 |
| **Sign Language Activity Projection** | arXiv:2606.09424 | 手语视频 | VAP 范式迁移到手语 |
| **Onishi et al.** | IEICE Trans. Inf. Syst. 2024 | 非语言行为 | VAP + 非语言特征 |

**这条线的共同局限（对本项目至关重要）**：

1. **完全没有语义/句法层。** VAP 输入是 waveform + (可选)视觉，输出是发声概率图。它无法区分"他停下来是因为说完了"和"他停下来是因为在想词"——而这正是语义 VAD 存在的理由。
2. **只在双人对齐视频会议语料上验证**，数据规模远小于语音语料。
3. **noise 论文的一个被忽视的结论**：*"successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions."* 这说明即使在 VAP 这种"无语义"框架里，转写质量仍是瓶颈——反过来支持"语义层必须做好"。

### 3.3 谱系 C：AV-Dialog —— 最接近"视觉引导语义端点"的工作 ★

**arXiv:2511.11124**，Tuochao Chen, Bandhav Veluri, Hongyu Gong, Shyamnath Gollakota（华盛顿大学 Allen School + Meta AI Research），2025-11-14，CC BY 4.0，项目页 `avdialog.cs.washington.edu`。

摘要原文（关键句）：

> We present AV-Dialog, **the first multimodal dialog framework that uses both audio and visual cues to track the target speaker, predict turn-taking, and generate coherent responses.** … AV-Dialog achieves robust streaming transcription, **semantically grounded turn-boundary detection** and accurate responses …

**架构**

```
音频 → DAC (Descript Audio Codec)，每 40ms → 16 个 codebook token
视频 → dlib 人脸检测 → 唇部区域 → AV-HuBERT 连续视觉特征（25 Hz）
        ↓
e_n = L_A(Σ_{i=1..16} E(A_{n,i})) + L_V(V_n) + E(U_{n-1}) + E(T_{n-1})
        ↓
    Llama3-8B
        ↓
两个线性头：
  L_U → 文本流 U_n（时间对齐的流式 AVSR，静音输出 <EMP>）
  L_T → 话轮事件流 T_n
```

**话轮事件标签集**（采用 PairwiseTurnGPT 分类体系）：

| 事件 | 输出 token |
|---|---|
| Normal turn（用户说完后 agent 说） | `<SOT>` |
| Overlapping turn（部分重叠） | `<SOT>` |
| Backchannel（"hmm"、"yeah"） | `<SOB>` |
| 无事件 | `<EMP>` |

**核心结果（Table 3，InterAct 测试集，Response Ratio = FTO 落在 −2s~3s 的比例）**

| 模型 | Clean | BG 噪声 | 干扰说话人 |
|---|---|---|---|
| Moshi | 54.0% | 53.8% | 52.5% |
| SE + Moshi | 55.9% | 56.0% | 54.4% |
| Ours (A) | 73.2% | 70.2% | 65.8% |
| Ours (V) | 72.5% | 72.5% | 72.5% |
| **Ours (A+V)** | **74.5%** | **78.3%** | **78.8%** |
| Ours (Unified) | 68.1% | 75.6% | 75.9% |

**视觉增益：Clean +1.3%，BG +8.1%，Interf +13.0%**（论文原文明确给出这三个数）。

AVSR WER（InterAct，Table 2）：A 28.6/68.0/92.2 → **A+V 16.3/37.4/30.8**，视觉在干扰条件下把 WER 从 92.2% 打到 30.8%，效果惊人——**但那是识别任务，不是端点判断**。

其他值得记录的：
- **消融（Table 6）**：声学 token（DAC）显著优于语义 token（DinoSR）。DinoSR(A+V) 在噪声下 Response Ratio 反而**掉到 49.1/47.8**，低于 DinoSR(A) 的 63.3/63.2。→ **视觉能否发挥作用，强依赖于音频表征选型。**
- **消融（Table 8）**：去掉 `<SOT>` 显式监督，Response Ratio 从 68.1/75.6/75.9 崩到 **48.0/35.1/38.0**。→ **显式话轮监督是必需的，不能指望端到端自行涌现。**
- **算法延迟（Appendix A）**：DAC 与视觉编码器均因果运行于 25 Hz，但 **AV-HuBERT 有 2 帧前视，总算法延迟约 120 ms**；对比 Moshi 的 80 ms。论文自己指出这是主要限制，"可通过预训练更小或零前视的视觉编码器缓解"。
- **训练数据**：Stage 1 = 文本续写(48%) + LibriLight/MLS/VP400k ASR(32%) + AudioSet captioning(4%) + VoxCeleb2 AVSR(16%)；Stage 2 = Fisher(音频) + **InterAct / Seamless Interaction**(视听)。128×A100（Stage 1）/ 32×A100（Stage 2）。
- **合成混音增广**：20% clean / 40% MUSAN 背景噪声 / 40% 1–4 个干扰说话人，SNR −8~8 dB。

**Limitations 原文**：

> It currently does not explicitly model non-verbal auditory cues (e.g., laughter, sighs) or **visual cues (e.g., facial expressions, gestures) beyond lip movements**. … factors like poor lighting, occlusions (e.g., hands covering the mouth) or extreme head poses, can impair lip movement extraction, affecting speaker tracking and speech understanding.

**对本项目的意义（三点，都很关键）**：

1. ✅ **它证明了"视听 LLM + 显式话轮 token 头 + 40ms 流式"这条技术路线是可行的**，且规模化训练配方公开。这是本项目最直接可借鉴的蓝图。
2. ❌ **它的"semantically grounded"是弱主张**。语义性来自"同时训 AVSR 流让模型内部有转写表征"，输出侧没有 complete/incomplete 判决。它无法回答"用户说了'我想订一张去'然后停顿 800ms"该不该接话——它只会根据学到的 FTO 分布决定要不要发 `<SOT>`。
3. ❌ **它只用唇部特征**（作者自己列为 limitation）。而唇动主要贡献 AVSR 与说话人跟踪；对"turn-holding intent"最有信息量的注视、表情、手势**完全未建模**。而 Kurata 2023 的消融恰恰显示**眼动 > 嘴部**。→ **这是一个明确的、有文献支撑的改进入口。**

### 3.4 谱系 D：MM-F2F —— 唯一同时有语义与视觉，且明确记录了语义失败案例 ★

**arXiv:2505.12654**，Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi（厦门大学信息学院），**ACL 2025**，CC BY 4.0，代码 <https://github.com/Linyx1125/MM-F2F>。

**任务形式**：词帧（word frame）级三分类 —— `KEEP` / `TURN` / `BACKCHANNEL`。

**架构**：
- 文本：**GPT-2**（取最后 token hidden state）
- 音频：**HuBERT**（时间维平均池化）
- 视频：**VideoMAE（仅人脸区域）**，每 clip 最后 16 帧
- 三模态各投影为 256 维 → **基于低秩分解（LMF）的 Flexible Fusion，rank=16** → 模态选择机制 + 随机模态丢弃训练（RMDT）→ 3 层 MLP [256, 64, 3]

**模态消融（Table 3）—— 本报告最重要的一张表**

| 训练模态 | Accuracy | F1 Keep | F1 Turn | F1 BC |
|---|---|---|---|---|
| Text | 0.751 | 0.747 | 0.767 | 0.707 |
| Audio | 0.751 | 0.737 | 0.735 | 0.805 |
| **Video** | **0.559** | 0.597 | 0.536 | 0.513 |
| Text + Audio | 0.811 | 0.783 | 0.809 | 0.894 |
| Text + Video | 0.757 | 0.751 | 0.766 | 0.743 |
| Audio + Video | 0.742 | 0.742 | 0.770 | 0.829 |
| **Text + Audio + Video** | **0.823** | 0.806 | 0.811 | **0.906** |

**读法**：
- Video 单模态 0.559，几乎接近三分类随机偏置水平，**视觉自身几乎不携带话轮判定信息**。
- 加入 Text 后：Text 0.751 → T+V 0.757（**+0.6%**）。加入 Audio 后：T+A 0.811 → T+A+V 0.823（**+1.2%**）。
- **视觉唯一显著贡献的是 Backchannel**：A 0.805 → A+V 0.829；T+A 0.894 → T+A+V 0.906。
- 与 SOTA 对比（Table 5）：TurnGPT(T) 0.645，Wang'24(T+A) 0.737，**Kurata'23(T+A+V) 0.720**，Ours 0.823。

**backbone 选型消融（Table 2）对视觉前端选型有直接参考价值**：

| 模态 | Backbone | Accuracy |
|---|---|---|
| Video | ViT（单帧） | 0.473 |
| Video | VideoMAE（全画面） | 0.533 |
| Video | **VideoMAE（人脸裁剪）** | **0.559** |

→ **单帧 ViT 明显不如时序模型；人脸裁剪优于全画面。** 这两条都直接指导本项目的视觉前端设计。

**MM-F2F 数据集**（可能是本项目最有用的现成资源之一）

| 项目 | 数值 |
|---|---|
| 时长 | **210 小时** |
| 视频数 | 773 段野外英语双人对话 |
| 说话人 | ~955 人，覆盖不同种族性别 |
| utterance | 169,029 条 |
| 帧数 | ~20 M |
| 词帧级标注 | >1.5 M |
| turn-taking 实例 | ~51 K |
| backchannel 实例 | ~22 K |
| 平均 utterance | 9.33 词 / 4.18 s |
| 自动标注精度 | >95%（100 名志愿者复核，剔除约 4%） |
| 去标识化 | >10,000 张合成人脸替换 + 声纹扰动 20% std |

去标识化对性能影响很小（Table 7：全原始 0.836 vs 全去标识 0.823），说明该数据集可用且泛化性可接受。

**★ Limitations 原文（第 7 节）—— 本报告的核心引用**

> There is still improvement space due to the subtle and complicated nature of multi-modal conversation signals. For example, **when the speaker is semantically incomplete and pauses to think, a backchannel response might be expected. Instead, our framework may occasionally misinterpret this as an indication for turn-taking due to the speaker's prolonged silence and frozen expression**, as shown in Fig. 5. To solve this problem, incorporating additional visual cues such as body movements or gestures could be a potential direction.

**这段话的分量**：一篇 ACL 2025 论文，拥有文本（=语义）+ 音频 + 视觉三模态，210 小时数据，SOTA 结果 —— **仍然在"语义不完整 + 思考型停顿 + 表情凝固"这个场景上失败**，并把"引入身体动作/手势"列为未来方向。

这直接证明了三件事：
1. **语义不完整 + 长停顿是当前多模态话轮模型的公认难例**（不是我们臆想的问题）；
2. **仅有"文本模态"不等于有"语义完整性判决"**：MM-F2F 有 GPT-2 文本分支，却仍然被长静默 + 冻结表情误导为 TURN。这说明**需要一个显式的 complete/incomplete 监督信号**，而非指望分类头隐式学到；
3. **人脸/唇部视觉不足以解决它**（"frozen expression"恰恰是误判的原因之一），需要**更广的视觉线索**（注视、手势、躯体）。

### 3.5 谱系 E：Omni 全双工大模型（架构上有视觉→是否说话的通路，但从未被验证）

#### MiniCPM-o 4.5 / Omni-Flow（arXiv:2604.27393，OpenBMB，9.34 B）

这是本调研中**架构上最接近"视觉引导语义 VAD"的通用模型**，值得详细记录。

**Omni-Flow 三条时间对齐流**：
- `env-visual`：环境视觉观测
- `env-audio`：声学场景（含用户语音）
- `out-stream`：assistant 的文本/语音输出

**统一序列化**：第 k 个时间 chunk 组成 `g_k = [v^k; a^k; o^k]`，串接为单一因果序列。**当不该输出时，`o^k` 只含一个特殊 `[listen]` token。**

论文原文（§3.1）：

> the model does not rely on explicit requests as the trigger before responding. … it must determine not only *what* to output, but also ***whether* and *when* to output on its own**.

论文原文（§3.2）：

> Since the model determines whether to output in each time window, it **naturally supports proactive behavior and reduces the reliance on external VAD modules**.

**→ 这是一个明确的"用 omni LLM 取代外部 VAD"的主张，且 `[listen]/[speak]` 决策在每个 chunk 内是在处理完 `v^k` 与 `a^k` 之后做的，即形式上以视觉为条件。**

**Design tradeoffs 消融（Table 1）—— 对本项目极为关键**

| Chunk Size | Boundary | Control | AdvBench | AlpacaEval | IFEval | SDQA | MMLU |
|---|---|---|---|---|---|---|---|
| **1.0 s** | Explicit | **LS** | 0.98 | **3.56** | **0.29** | **0.36** | **0.65** |
| 1.0 s | Explicit | LT | 0.92 | 3.60 | 0.24 | 0.35 | 0.56 |
| 1.0 s | Implicit | LT | 0.96 | 3.31 | 0.22 | 0.28 | 0.45 |
| **0.2 s** | Explicit | LS | 0.81 | **1.22** | 0.10 | **0.09** | 0.45 |
| **0.1 s** | Explicit | LS | 0.67 | 2.40 | 0.10 | 0.13 | 0.32 |

三条结论（论文自述）：
1. **时间粒度**：1.0 s 最优；缩到 0.2 s / 0.1 s **性能崩塌**（"chunk 太短时模型在每个时间窗内没有足够信息做稳定决策"）。
2. **边界显式化**始终有益。
3. **LS（Listen-Speak，先预测二值 listen/speak 控制 token，再生成内容）优于 LT**（在共享输出空间直接预测 `[listen]` 或文本）——即 **"决定是否说话"应与"决定说什么"解耦**。

**→ 对本项目的三个直接推论**：

- **推论 1（强）**：Omni LLM 路线**无法达到语义 VAD 需要的时间分辨率**。MiniCPM-o 4.5 在 1.0 s 才稳定，而 SoulX 是 160 ms、X2-Turn 是 80 ms。差 6.25~12.5 倍。**这为"独立的轻量视听语义 VAD 模块"提供了强有力的存在理由**——不是"omni 模型迟早会覆盖"，而是"omni 模型的架构在这个时间尺度上会崩"。
- **推论 2（强）**：**"whether to speak" 应与 "what to say" 解耦**（LS > LT）。这与 SoulX/X2-Turn 的"独立状态头"设计不谋而合，且是被 ablation 验证过的。本项目可直接引用为设计依据。
- **推论 3**：视觉编码代价高：SigLIP ViT 0.4B，全双工模式下最大 448×448，每 slice 1024 token 压缩为 64 token（16× 压缩率），1–5 FPS。这个开销对 0.6B 的 SoulX 是不可接受的，**必须换更轻的视觉前端**（见 §6.3）。

**关键缺陷：视觉引导的话轮能力从未被评测。** 全双工评测（Table 8）只报告 LiveSports-3K-CC，论文明确标注这是 **"audio-free full-duplex benchmark"**（无音频）。也就是说：模型架构上视觉参与了 listen/speak 决策，**但没有任何实验证明视觉改善了话轮/端点判断**。

#### 其他 Omni 全双工 + 视觉工作

| 工作 | ID | 与本题关系 |
|---|---|---|
| **ELLSA** | arXiv:2510.16756（ICLR 2026 投稿） | 全双工 视觉+语音+动作，SA-MoE 架构；关注具身交互，非端点判断 |
| **Wan-Streamer** | 见 §Methodology | 原生全双工音视频；话轮管理与生成联合学习 |
| **RoboEgo** | — | 全双工具身 omni 模型 |
| Qwen3-Omni / Qwen2.5-Omni | 2509.17765 / 2503.20215 | Omni 但非原生全双工；无视觉条件端点判断 |
| VITA-1.5 / Mini-Omni2 | — | 早期视觉-语音交互；VideoFDB 评测中表现最差（见 §3.7） |
| Let's Go Real Talk | ACL 2024 (2024.acl-long.860) | 面对面口语对话模型（AV2AV 生成）；无话轮建模（AV-Dialog 明确指出这点） |

### 3.6 谱系 F：工业方案

| 方案 | 是否视觉 | 说明 |
|---|---|---|
| **LiveKit** turn detector | ❌ | 基于转写文本的语义端点模型；纯文本+音频 |
| **Pipecat** Smart Turn | ❌ | 开源语音端点模型；纯音频 |
| **TEN** Turn Detection | ❌ | 中英语义端点；纯文本/音频 |
| **Tavus** CVI | ⚠️ **分离** | `Raven-0` 做视觉感知（情绪/注意力），`Sparrow-0` 做话轮检测。二者是**独立模块**，未见证据表明 Raven 的视觉信号进入 Sparrow 的端点判决 |
| **Anam / Keyframe** 数字人 | ❌ 架构性不可能 | 音频驱动数字人本质回合制，VideoFDB 实测其 Verbal Backchanneling 时序对齐率 **0%** |
| Gemini Live / gpt-realtime | ⚠️ | 有视频输入，但 VideoFDB 实测视觉不改善时序（见 §3.7） |

**结论：工业界目前没有任何产品把视觉信号接入话轮/端点判决。** 视觉能力（情绪识别、注意力检测）与话轮能力是两条独立管线。

### 3.7 ★ 关键负面证据：VideoFDB 证明"当前系统的视觉流对时序毫无帮助，甚至有害"

**arXiv:2605.30256**，*VideoFDB: Evaluating Full-Duplex Vision-Speech Capabilities in Conversational Agents*，NVIDIA + David AI。

**数据集**：237 段真实双人视频通话（226 测试 / 11 验证），130 位说话人，≥720p/30fps/24kHz，双端本地录制（规避网络延迟伪影），**11 类非语言对话动态**，三轮人工标注。

**评测系统**：Gemini 2.5 Flash Native、Gemini 3.1 Flash Live、gpt-realtime、gpt-realtime-mini、MiniCPM-o 4.5、Mini-Omni2、VITA-1.5，每个都跑 AV 与纯音频两版做同片段配对比较；另有 Gemini+Anam、Gemini+Keyframe 两条级联数字人管线。

论文明确指出：**"目前不存在公开可用的端到端全双工 AV2AV 系统"**。

**核心结果（感知轴 + 时序）**

| 模型 | Fluency | Conv. Flow | Vis. Ground. | Overall | TOR-Align / 中位延迟 |
|---|---|---|---|---|---|
| **人类参考** | 4.16 | **4.20** | 4.24 | **4.20** | **90% / 1400 ms** |
| Gemini 2.5 Flash Native (AV) | 3.33 | 2.81 | 3.37 | 3.17 | 72% / 3160 ms |
| Gemini 3.1 Flash Live (AV) | 3.15 | 2.20 | 3.16 | 2.84 | 66% / 1720 ms |
| gpt-realtime (AV) | 2.72 | 2.50 | 3.02 | 2.75 | 72% / 5400 ms |
| gpt-realtime-mini (AV) | 2.91 | 2.37 | 2.90 | 2.73 | 66% / 5320 ms |
| **MiniCPM-o 4.5 (AV)** | 3.03 | 3.54 | **3.63** | 3.40 | 73% / **720 ms** |
| Mini-Omni2 (AV) | 0.65 | 1.37 | 1.54 | 1.19 | 64% / 3080 ms |
| VITA-1.5 (AV) | 1.19 | 1.57 | 2.53 | 1.76 | 58% / 400 ms |
| **MiniCPM-o 4.5（仅音频）** | **3.45** | **3.76** | 3.10 | **3.44** | 72% / 920 ms |
| Gemini 2.5（仅音频） | 3.35 | 2.98 | 3.17 | 3.17 | 73% / 2760 ms |

**四个必须记录的发现**：

1. **视觉让时序变差**：相对纯音频，加入视觉使 **7 个模型中 6 个的 TOR-Alignment 下降 0–5 个百分点**。唯一例外 gpt-realtime 提升 5 点，但中位延迟增加 1000 ms。论文结论：**"加入视频没有让任何模型在实时容差内改善时序。"**
2. **MiniCPM-o 4.5 的 Conversational Flow：AV 3.54 < 纯音频 3.76**。即上文 §3.5 中"架构上视觉参与 listen/speak 决策"的模型，实测视觉反而损害了话轮流畅度。
3. **两类失败模式**：
   - **Captioning collapse**：模型把视觉输入当成看图说话 prompt。Mini-Omni2 **87% 的回复是视觉描述**；VITA-1.5 出现"我只是个程序，我看不见也听不见"这类能力免责声明，约 74% 回复 token 重复。
   - **Visual-stream ignorance**：gpt-realtime-mini 的 AV 与纯音频输出**互为改写**，视觉流既不改变时机也不改变内容。
   - 论文总体结论：**当前系统只在"被显式言语提及"时才用视觉（VQA 式使用），不会做自然对话所需的流式联合视听接地。**
4. **视觉帧率不是越高越好**：MiniCPM-o 4.5 在 1–10 FPS 扫描中 **2 FPS 达峰**，之后下降（总分 8 FPS 3.04 → 10 FPS 2.81；Fluency 3.55 → 2.33）。论文推断更密视觉输入会挤占共享跨模态注意力预算。

**这对本项目意味着**：

- ✅ **强化了立项理由**：把视频塞给通用 omni LLM **不但没用，还有害**。真正需要的是**专门为端点判断设计、以时序精度为第一目标的轻量视听模块**。
- ⚠️ **警示**：视觉容易变成噪声或干扰。**必须设计防退化机制**（模态 dropout、门控、视觉不可用时的优雅降级），并且**必须报告"加视觉后是否变差"**这一负向指标，否则评审会直接用 VideoFDB 打回。
- ✅ **给了现成的评测框架**：VideoFDB 的 TOR-Alignment（把动态映射为 stay-silent / continue-speaking / yield-required / smooth-handoff / backchannel-produced 五个时序类，统计满足期望时序的比例 + 中位延迟）**可以直接用于本项目的视听语义 VAD 评测**，且有人类上界（90% / 1400 ms）。

### 3.8 其他相关基准

| 基准 | 模态 | 与本题关系 |
|---|---|---|
| **SoulX-Duplug-Eval** | 音频 | 本项目的音频侧对照基线，必用 |
| **Full-Duplex-Bench** | 音频 | takeover-rate 指标的来源；VideoFDB 的 TOR 是其多模态扩展 |
| **Easy Turn** | 音频 | 语义端点评测集 |
| **TurnBench**（arXiv:2608.25218） | 音频 | 话轮系统横向评测 |
| **VideoFDB**（2605.30256） | **视听** | ★ 唯一的全双工视听基准；但**不测语义完整性** |
| **Real-TurnTurk**（arXiv:2608.22071） | 视觉+声学+语言 | 多模态土耳其语话轮语料 |
| Daily-Omni / WorldSense / JointAVBench / AVUT-Human | 视听 | 理解类，非交互时序类 |

---

## 4. 证据综合：视觉到底能带来什么、不能带来什么

### 4.1 量化汇总

| 工作 | 任务 | 是否含语义/文本 | 纯音频 | +视觉 | Δ | 条件 |
|---|---|---|---|---|---|---|
| AV-Dialog | Response Ratio | 间接（AVSR 流） | 73.2% | 74.5% | **+1.3** | Clean |
| AV-Dialog | Response Ratio | 同上 | 70.2% | 78.3% | **+8.1** | BG 噪声 |
| AV-Dialog | Response Ratio | 同上 | 65.8% | 78.8% | **+13.0** | 干扰说话人 |
| AV-Dialog | AVSR WER↓ | — | 92.2% | 30.8% | **−61.4** | 干扰说话人 |
| MM-VAP | hold/shift 平衡准确率 | ❌ | 79% | 83~84% | **+4~5** | Clean |
| MM-VAP (noise) | hold/shift | ❌ | 52% | 72% | **+20** | 10dB 音乐噪声 |
| MM-F2F | 三分类 Accuracy | ✅ GPT-2 | 0.811 | 0.823 | **+1.2** | 野外视频 |
| MM-F2F | Backchannel F1 | ✅ | 0.894 | 0.906 | **+1.2** | 野外视频 |
| MM-F2F | Backchannel F1（无文本） | ❌ | 0.805 | 0.829 | **+2.4** | 野外视频 |
| Kurata'23 | 话语结束 AUC | ✅ | 0.896 | 0.920 | **+2.4** | 日语对话 |
| **VideoFDB** | **TOR-Alignment** | ✅（LLM） | — | — | **−0~5（6/7 模型变差）** | 真实视频通话 |

### 4.2 三条稳健结论

**结论 1：视觉增益与音频退化程度成正比，在干净音频 + 已有语言模态时趋近于零。**

AV-Dialog 三条件（+1.3 / +8.1 / +13.0）是一条干净的单调曲线。MM-VAP 噪声实验（+20 @ 10dB 音乐噪声）同理。MM-F2F 在野外视频（音频质量参差）上也只有 +1.2。

→ **视觉的机理是"当音频不可靠时，提供冗余的说话人活动证据"**，而不是"提供音频没有的语义信息"。

**结论 2：视觉单独几乎不携带话轮判定信息，更不携带语义完整性信息。**

MM-F2F Video-only 0.559（三分类）；AV-Dialog V-only AVSR WER 67.8~87.1%；Barkhuysen 2008 的 V-only 显著弱于 A-only。

→ **视觉必须是辅助模态，不能是主模态。** 任何设计中，音频/文本通路必须能在视觉缺失时独立工作。

**结论 3：把视觉喂给通用 omni LLM 会主动损害时序表现。**

VideoFDB：6/7 模型 TOR-Alignment 下降；MiniCPM-o 4.5 Conv.Flow AV 3.54 < audio-only 3.76；captioning collapse + visual-stream ignorance 两类失败模式；2 FPS 之后越多帧越差。

→ **需要专用的、时序优先的、轻量的视听端点模块，而不是更大的 omni 模型。**

### 4.3 ★ 尚未被任何工作检验的假设（本项目的核心科学问题）

上述所有工作测的都是 **"视觉能否帮助判断话轮该不该转移"**。

**没有任何工作测过：视觉能否帮助判断"说话人打算继续说"（turn-holding intent）——特别是在语义不完整的思考型停顿中。**

这个空白由两条独立证据交叉确认：

1. **MM-F2F 的 Limitations**：语义不完整 + 长停顿 + 表情凝固 → 误判为 turn-taking。作者提出的解法是"引入身体动作/手势"，**但未实现、未验证**。
2. **Kurata 2023 的消融**：视觉线索中**眼动贡献 > 嘴部 > 头部**。而 **AV-Dialog 只用唇部**（自列为 limitation），**MM-F2F 用 VideoMAE 人脸裁剪**（含表情但被"frozen expression"误导）。

→ **假设 H**：思考型停顿（语义不完整）与结束型停顿（语义完整）在**注视行为**上系统性不同（前者常伴注视回避/上移、口型未闭合、手势保持；后者常伴注视投向对方、口型闭合、身体放松）。若成立，视觉可以在**干净音频条件下**改善 complete/incomplete 判决——这正是现有工作全部缺失的增益来源。

**这个假设是可判定的**，且判定成本不高（见 §6.6 的最小验证实验）。**如果 H 成立，本方向有独立的科学贡献；如果 H 不成立，本方向只是噪声鲁棒性工程，价值有限。建议先花 2 周验证 H，再决定是否全力投入。**

---

## 5. 空白点清单

| # | 空白 | 证据 | 严重度 |
|---|---|---|---|
| G1 | **任务定义空白**：不存在视听条件下的 complete/incomplete 标签体系与形式化定义 | 全部视觉话轮工作输出均为 shift/hold 或 SOT/SOB 或 KEEP/TURN/BC | ★★★ |
| G2 | **数据空白**：不存在带 complete/incomplete 标注的视听语料 | MM-F2F 标 KEEP/TURN/BC；InterAct 标 turn 事件；SoulX-Duplug-Eval 无视频 | ★★★ |
| G3 | **评测空白**：语义端点基准（SoulX-Eval / Easy Turn / TurnBench / Full-Duplex-Bench）全部纯音频；唯一视听基准 VideoFDB 不测语义完整性 | §3.8 | ★★★ |
| G4 | **视觉线索空白**：最有信息量的注视/手势在端点任务上未被建模 | AV-Dialog 仅唇部（自述 limitation）；Kurata 消融显示眼动 > 嘴部 | ★★★ |
| G5 | **时间粒度空白**：视觉前端延迟与语义 VAD 需求不匹配 | AV-HuBERT 2 帧前视 = 120ms；OpenFace 非实时；Omni-Flow 1.0s chunk vs SoulX 160ms / X2-Turn 80ms | ★★ |
| G6 | **鲁棒性空白**：遮挡/侧脸/暗光/无人脸下的优雅降级无人系统研究 | AV-Dialog limitation 明确列出；VideoFDB 显示视觉常有害 | ★★ |
| G7 | **负向指标空白**：无人报告"加视觉后是否变差" | VideoFDB 是唯一做配对比较的，结果是 6/7 变差 | ★★ |

---

## 6. 技术路线建议

### 6.1 三条路线对比

| | 路线 A：SoulX-Duplug + 视觉 projector | 路线 B：X2-Turn fork | 路线 C：轻量旁路门控 |
|---|---|---|---|
| 做法 | 复制 `EncoderProjector` 为 `VisualProjector`，在序列中插入视觉 token 位 | patch Voxtral-Realtime + vLLM，加视觉分支 | 主干完全不动，视觉侧独立小模型只输出对 `complete_bias` / `max_wait_num` 的调制 |
| 训练成本 | 中（0.6B + LoRA，单机可行） | 高（4B，需 patch 两层基础设施） | **极低**（只训一个小 head） |
| 上限 | 高 | 最高 | 低-中 |
| 风险 | 中 | 高 | 低 |
| 可解释性 | 中 | 中 | **高**（能直接量化视觉改变了多少等待时间） |
| **建议** | **主线** | 第二阶段 | **先做，用于验证假设 H** |

### 6.2 推荐执行顺序

```
Phase 0（2 周）路线 C：验证假设 H —— 视觉能否区分"思考型停顿"与"结束型停顿"
  ↓ H 成立
Phase 1（6-8 周）路线 A：SoulX-Duplug + 视觉分支，端到端训练，SoulX-Eval + 自建视听集对照
  ↓
Phase 2 路线 B：迁移到 X2-Turn 的 80ms 帧同步双头，证明方法在细粒度/大规模下有效
```

### 6.3 视觉前端选型

这是本项目最关键的工程决策，因为它同时决定延迟、鲁棒性和可解释性。

| 前端 | 维度/开销 | 延迟 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|---|
| **MediaPipe Face Landmarker** | 478 点 + blendshape + 头部姿态矩阵 | **实时、因果、零前视** | 端侧可跑、CPU 友好、含注视/口型/表情、可解释 | 非学习式表征，需自己接编码器 | ★ **首选（Phase 0/1）** |
| **OpenFace 2.0** | AUs + 注视 + 头部姿态 | 非实时 | 学术界标准、AU 可解释、MM-VAP 系用它 | 工程上无法实时部署 | 离线研究/标注 |
| **AV-HuBERT** | 唇部连续特征 25Hz | **120 ms（2 帧前视）** | AV-Dialog 验证过、AVSR 强 | 只有唇部（G4）、前视代价、模型大 | 若要复现 AV-Dialog |
| **VideoMAE（人脸裁剪）** | clip 级 | 非流式 | MM-F2F 验证优于 ViT 单帧与全画面 | 非因果、clip 级不适合帧同步 | 离线对照 |
| **SigLIP + resampler** | 0.4B，64 token/slice | 高 | omni 生态兼容 | 对 0.6B 主干过重；VideoFDB 显示帧率高反而变差 | ❌ 不建议 |
| 自训因果轻量 CNN/TCN | 可控 | **可做到零前视** | 延迟最优、可蒸馏 | 需数据与训练成本 | Phase 2 优化 |

**建议**：**MediaPipe Face Landmarker 抽取显式几何/语义特征**（注视方向、眼睑开合、口型开合度与闭合速度、头部 pitch/yaw、眉毛 blendshape、注视是否投向摄像头），再接一个小的因果 TCN/GRU 编码器。

理由：
1. **零前视、实时、端侧** —— 直接解决 G5；
2. **显式覆盖注视与口型闭合** —— 直接针对 G4 与假设 H；
3. **可解释** —— 能画出"注视回避 → 模型延长等待"的因果链，这是论文里最有说服力的图；
4. **人脸缺失时天然降级** —— landmark 检测失败即置 `<NULL>`，直接解决 G6。

> AV-Dialog 用 `<NULL>` token（embedding 全零、target 中不计 loss）处理缺失模态，这个技巧应直接采用。

### 6.4 时序对齐方案

| | 视频 | SoulX-Duplug | X2-Turn |
|---|---|---|---|
| 原生速率 | 25 / 30 fps | 80 ms/token（12.5 Hz），160 ms/chunk | 80 ms/frame（12.5 Hz） |
| 对齐方式 | 25 fps → 每 2 帧池化 = 80 ms；30 fps → 每 2.4 帧插值/池化 = 80 ms | 每 chunk 2 个视觉 token，与 2 个音频 token 一一配对 | 每帧 1 个视觉 token |

**推荐**：视频统一重采样到 **12.5 Hz**（= 80 ms/视觉 token），与音频 token 严格 1:1 对齐。序列布局：

```
... [v_t] [a_t] [v_{t+1}] [a_{t+1}] ... [state_token]
```

或（更省 token、且与 AV-Dialog 一致）**embedding 相加而非拼接**：

```
e_t = AudioProjector(audio_emb_t) + VisualProjector(visual_feat_t)
```

AV-Dialog 用的是相加（`e_n = L_A(...) + L_V(V_n) + E(U_{n-1}) + E(T_{n-1})`）。**相加的优势**：序列长度不变，KV-cache 不膨胀，对 0.6B 主干友好，且视觉缺失时置零即等价于纯音频模型 —— **天然满足 G6 的优雅降级**。**强烈建议采用相加方案。**

### 6.5 代码改动点清单（路线 A，基于本地实际代码）

| # | 文件 | 位置 | 改动 |
|---|---|---|---|
| 1 | `model/model.py` | 17-35 | 新增 `VisualProjector`（复制 `EncoderProjector`，输入维改为视觉特征维） |
| 2 | `model/model.py` | 60-71 | `__init__` 中按 `enable_visual_projector` 实例化，支持 freeze |
| 3 | `model/model.py` | 145-166 | `forward` 中 `inputs_embeds = audio_embeds*mask_a + text_embeds*mask_t` **改为再加 `visual_embeds`（对齐位置相加，缺失置零）** |
| 4 | `config/config.py` | 27-35 | 若引入新状态（如 `<\|user_thinking\|>`）在此扩展 token id |
| 5 | `config/config.py` | 118-131 | 新增/调整 `*_loss_rate`，建议给 `user_incomplete` 更高权重（打断代价高于等待） |
| 6 | `config/config.yaml` | 29-33 | `input` 段新增 `video_fps` / `visual_feat_dim` / `visual_ahead_size` |
| 7 | `service/model.py` | 105-113 | 若新增状态，注册对应 action token 与 embeds |
| 8 | `service/model.py` | ~620 | embeds 序列拼装处插入视觉 embeds |
| 9 | `service/model.py` | 708-733 | `complete_bias` 由常量改为 **视觉条件调制的动态值**（这是路线 C 的落点，也是最小改动的价值验证点） |
| 10 | `service/model.py` | 252-256 | `far_field_threshold` 的 RMS 门控可用"是否检测到说话人人脸/唇动"加固 |
| 11 | training-code 分支 | dataset/collator | 每条样本新增 `visual_feats` 字段 + `visual_mask`；实现随机模态 dropout（参考 MM-F2F 的 RMDT） |

### 6.6 ★ Phase 0 最小验证实验（验证假设 H）

**目标**：在**不训练主干**的前提下，判定视觉是否携带区分"思考型停顿（incomplete）"与"结束型停顿（complete）"的信息。

**做法**：
1. 取一批双人视听对话（候选：MM-F2F 210h 已开源 / InterAct / Candor / NoXi）。
2. 用现成 ASR + LLM（可复用 X2-Turn 论文的标注思路）为每个静音段自动标注 `complete` / `incomplete`，人工抽检。
3. 对每个静音段的**起始前 500 ms + 起始后 500 ms**，用 MediaPipe 抽取特征序列。
4. 训一个**极小的分类器**（logistic regression / 小 GRU）：仅视觉 → complete/incomplete。
5. 同时训：仅音频韵律 → complete/incomplete；视觉+韵律 → complete/incomplete。

**判据**：

| 结果 | 结论 | 行动 |
|---|---|---|
| 仅视觉 AUC 显著 > 0.5（比如 > 0.62） | **H 成立**，视觉携带独立的语义完整性证据 | 全力推进 Phase 1，这是论文的核心卖点 |
| 仅视觉 AUC ≈ 0.5~0.55，但视觉+韵律 > 韵律 | H 弱成立，视觉是条件性增益 | 推进 Phase 1，但把定位改为"噪声/远场鲁棒性" |
| 仅视觉 AUC ≈ 0.5 且无交互增益 | **H 不成立** | 停止或转向 turn-taking(T4) 而非 semantic VAD(T3) |

**这个实验 2 周内可完成，成本极低，但决定整个项目的定位。强烈建议作为第一步。**

### 6.7 数据方案

| 语料 | 规模 | 视听 | complete/incomplete 标注 | 可得性 |
|---|---|---|---|---|
| **MM-F2F** | 210 h / 773 视频 / 955 人 | ✅ | ❌（KEEP/TURN/BC） | ✅ 开源（GitHub） |
| **Seamless Interaction (InterAct)** | 大规模双人视听（Meta，arXiv:2506.22554） | ✅ | ❌（turn 事件） | ✅ 公开 |
| **SoulX-Duplug-Eval** | — | ❌ 纯音频 | ✅ | ✅ HF |
| **VideoFDB** | 237 段真实视频通话 | ✅ | ❌（11 类非语言动态） | 承诺发表前开源 |
| Candor / NoXi / EgoCom / AVCC | 中等 | ✅ | ❌ | 部分公开 |

**建议标注流水线**（借用 X2-Turn / SoulX 的做法并扩展）：

```
视听语料
 → ASR + 词级时间戳（Whisper-Large / Paraformer）
 → 静音段切分（≥200ms）
 → LLM 判定该静音点处前文是否语义完整 → complete/incomplete 弱标签
 → 人工抽检校准（MM-F2F 用 100 志愿者复核，剔除 4%，可参照）
 → 对齐 MediaPipe 视觉特征（12.5 Hz）
 → 训练集
```

### 6.8 评测设计（必须做的对照）

**这是本项目最容易被评审攻击的地方，必须前置设计。**

| 维度 | 必测条件 | 理由 |
|---|---|---|
| 音频质量 | **Clean / 背景噪声 / 干扰说话人 / 远场** | AV-Dialog 证明增益强依赖此维度；只报噪声结果会被质疑 |
| 视觉质量 | **正脸 / 侧脸 / 遮挡 / 暗光 / 无人脸** | G6；AV-Dialog 自列 limitation |
| **负向指标** | **加视觉后是否变差（配对比较）** | ★ VideoFDB 显示 6/7 模型变差；不报这个等于自曝其短 |
| 指标 | complete/incomplete F1、打断率、等待时长分布、**VideoFDB 式 TOR-Alignment + 中位延迟** | 需同时覆盖判决准确性与交互时序 |
| 消融 | 音频-only / 视觉-only / 视听；唇部 / 表情 / 注视 / 头部姿态逐项 | 复现并扩展 Kurata'23 的"眼动 > 嘴部"结论 |
| 延迟 | **算法延迟**（平台无关，前视 + chunk）+ 端到端 RTF | AV-Dialog Appendix A 的做法，是正确范式 |

**特别提醒**：**必须单独报告 Clean 条件下的增益。** 如果 Clean 增益 ≈ 0（如 AV-Dialog 的 +1.3%），就必须诚实地把工作定位为"鲁棒性增强"而非"语义判断增强"，否则会被 AV-Dialog Table 3 直接反驳。

### 6.9 主要风险

| 风险 | 概率 | 依据 | 缓解 |
|---|---|---|---|
| Clean 条件视觉增益 ≈ 0 | **高** | AV-Dialog +1.3%；MM-F2F +1.2% | Phase 0 先验证 H；若不成立就改定位 |
| 视觉主动损害时序 | **中高** | VideoFDB 6/7 模型变差 | embedding 相加 + 模态 dropout + 视觉门控 + 缺失置零降级 |
| 视觉前端延迟吃掉全部收益 | 中 | AV-HuBERT 120ms 前视 | 用 MediaPipe（零前视）而非 AV-HuBERT |
| 无 complete/incomplete 视听数据 | **高** | G2 | LLM 弱标注 + 人工抽检；复用 MM-F2F 210h |
| 唇部/表情不足以捕捉 holding intent | 中 | MM-F2F "frozen expression" 失败案例 | 显式加入注视与手势特征 |
| 被 omni 大模型覆盖 | **低** | Omni-Flow 在 0.2s chunk 崩溃；VideoFDB 显示 omni 视觉时序更差 | 强调 80~160ms 粒度 + 端侧轻量是结构性优势 |

---

## 7. 与原 `turn-taking-landscape.md` 的关系

原文档结论"检索未发现任何视觉引导的语义 VAD 工作"**部分错误**：

| 原结论 | 实际情况 |
|---|---|
| 无任何视觉引导话轮工作 | ❌ 至少 10+ 篇（AV-Dialog、MM-F2F、MM-VAP×2、MuVAP、Kurata、Cano、Saga、Gaze-Triadic、Sign-VAP…） |
| 视觉+语义完整性判定是空白 | ✅ **正确**，且有 MM-F2F Limitations 与 AV-Dialog 标签集两条硬证据支撑 |
| 视觉会带来明显增益（隐含假设） | ⚠️ **需修正**：Clean 条件仅 +1.2~1.3%；VideoFDB 显示对通用 omni 模型是负增益 |

**修正后的立项论述应为**：

> 视觉引导的**话轮预测**（T4）已有相当积累，但视觉引导的**语义完整性判定**（T3）仍是空白。
> 且现有证据表明视觉的已知增益集中在噪声鲁棒性而非语义判断，
> 因此本项目的科学价值取决于一个可判定的假设：**视觉是否携带关于 turn-holding intent（尤其思考型停顿）的独立证据**。
> 这一假设由 MM-F2F 的失败案例与 Kurata'23 的"眼动 > 嘴部"消融共同指向，但从未被直接检验。

---

## 8. 参考文献

### 直接相关（视觉 + 话轮/端点）

1. Chen, T., Veluri, B., Gong, H., Gollakota, S. **AV-Dialog: Spoken Dialogue Models with Audio-Visual Input.** arXiv:2511.11124, 2025-11-14. UW + Meta AI. <https://arxiv.org/abs/2511.11124> ｜ 项目页 avdialog.cs.washington.edu
2. Lin, Y., Zheng, Y., Zeng, M., Shi, W. **Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals.** ACL 2025. arXiv:2505.12654. 厦门大学. 代码 <https://github.com/Linyx1125/MM-F2F>
3. O'Connor Russell, S., Harte, N. **Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction.** Findings of ACL 2025, 2025.findings-acl.12. arXiv:2505.21043
4. O'Connor Russell, S., Harte, N. **Visual Cues Support Robust Turn-taking Prediction in Noise.** Interspeech 2025, pp. 1073–1077. DOI 10.21437/Interspeech.2025-668. arXiv:2505.22088
5. Kurata, F., Saeki, M., Fujie, S., Matsuyama, Y. **Multimodal Turn-Taking Model Using Visual Cues for End-of-Utterance Prediction in Spoken Dialogue Systems.** Interspeech 2023, pp. 2658–2662. DOI 10.21437/Interspeech.2023-578
6. Qi, ..., Skantze, G. **MuVAP.** arXiv:2606.16731
7. Cano, ... et al. **MM-VAP** (audio-visual VAP for HRI). RO-MAN 2026. arXiv:2607.07294
8. Saga, ..., Pelachaud, C. arXiv:2506.03980
9. **Gaze-Enhanced Triadic Next-Speaker Prediction.** Interspeech 2025. arXiv:2505.13688. Meta Reality Labs
10. **Sign Language Activity Projection.** arXiv:2606.09424
11. Park, S. et al. **Let's Go Real Talk: Spoken Dialogue Model for Face-to-Face Conversation.** ACL 2024, 2024.acl-long.860

### 全双工 + 视觉（Omni）

12. Cui, J. et al. **MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction.** arXiv:2604.27393. OpenBMB. 代码 <https://github.com/OpenBMB/MiniCPM-o>
13. **ELLSA** (full-duplex vision-speech-action, SA-MoE). arXiv:2510.16756
14. Xu, J. et al. **Qwen3-Omni Technical Report.** arXiv:2509.17765
15. Défossez, A. et al. **Moshi: a speech-text foundation model for real-time dialogue.** arXiv:2410.00037
16. Veluri, B. et al. **Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents (SyncLLM).** EMNLP 2024
17. Leishman, S., Bell, P., Wallbridge, S. **PairwiseTurnGPT.** SemDial 2024

### 基准与数据集

18. Mazumdar, A. et al. **VideoFDB: Evaluating Full-Duplex Vision-Speech Capabilities in Conversational Agents.** arXiv:2605.30256. NVIDIA + David AI
19. Agrawal, V. et al. **Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset.** arXiv:2506.22554. Meta
20. **TurnBench.** arXiv:2608.25218
21. **Real-TurnTurk.** arXiv:2608.22071
22. Chung, J.S. et al. **VoxCeleb2.** ｜ Cieri, C. et al. **The Fisher Corpus.** LREC 2004 ｜ Snyder, D. et al. **MUSAN.** arXiv:1510.08484

### 音频侧语义 VAD 基线（本仓库）

23. Yan, R., Chen, W., Liu, Z., Ma, Z. et al. **SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation.** arXiv:2603.14877, 2026. Soul AI Lab + SJTU X-LANCE. Apache-2.0
24. Fu, K., Wen, R., Lin, A., Qin, S., Gan, R., Wang, H., Wang, Q. **X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction.** arXiv:2608.10878, 2026. X Square Robot. Apache-2.0
25. Mistral AI. **Voxtral-Mini-4B-Realtime-2602.** <https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602>

### 心理语言学基础

26. Barkhuysen, P., Krahmer, E., Swerts, M. **The interplay between the auditory and visual modality for end-of-utterance detection.** JASA, 2008
27. Nota, N., Trujillo, J.P., Holler, J. 眉毛动作与话轮转换, 2023
28. Best, V., Boyd, A.D., Sen, K. **An effect of gaze direction in cocktail party listening.** Trends in Hearing 27, 2023
29. Kendon, A. **Some functions of gaze-direction in social interaction.** Acta Psychologica, 1967

### 视觉/视听前端

30. Shi, B., Hsu, W.-N., Lakhotia, K., Mohamed, A. **AV-HuBERT.** ICLR 2022
31. Ma, P. et al. **Auto-AVSR.** ICASSP 2023
32. Tong, Z. et al. **VideoMAE.** NeurIPS 2022 ｜ Zhai, X. et al. **SigLIP.** ICCV 2023
33. Google. **MediaPipe Face Landmarker.** ｜ Baltrusaitis, T. et al. **OpenFace 2.0.** FG 2018
34. Kumar, R. et al. **DAC (High-Fidelity Audio Compression with Improved RVQGAN).** NeurIPS 2023

---

*报告完成时间：2026-08-27。检索方法、来源可信度评级与未核实项见 [`methodology.md`](methodology.md)；逐条工作速查见 [`evidence-table.md`](evidence-table.md)。*
