# AV-X2-Turn 架构设计

> **v4 — 2026-09-02 重写**。三段结构：现状 → 否决的路线 → 采纳的路线。
>
> **核心结论**：以 **X2-Turn-4B-0812 权重**为 base（不是 Voxtral），
> 套用 **SoulX-Duplug 范式**（冻结前端 + Projector + LoRA），侧接一路视觉，
> 在 `vad_lm_head` 的空闲 token id 上判别式输出逐帧 complete/incomplete。
>
> 所有代码锚点均已本地核实。

---

## 目录

- [第一部分 · 两个基座的现状](#第一部分--两个基座的现状)
  - [1.1 X2-Turn 结构](#11-x2-turn-结构)
  - [1.2 SoulX-Duplug 结构](#12-soulx-duplug-结构)
  - [1.3 两者对比：为什么一个当 base、一个当范式](#13-两者对比为什么一个当-base一个当范式)
- [第二部分 · 被否决的路线：端到端 Omni](#第二部分--被否决的路线端到端-omni)
  - [2.1 端到端 Omni 路线的结构会是什么样](#21-端到端-omni-路线的结构会是什么样)
  - [2.2 硬性否决理由：时间粒度](#22-硬性否决理由时间粒度)
  - [2.3 「换个小一点的 omni」为什么不成立](#23-换个小一点的-omni为什么不成立)
    - [2.3.1 ★ Qwen3.5-Omni 专项核查（原 V9，已解决）](#231--qwen35-omni-专项核查原-v9已解决)
  - [2.4 次级否决理由](#24-次级否决理由)
  - [2.5 核心论证：加模态是宽度改动，改粒度是架构改动](#25-核心论证加模态是宽度改动改粒度是架构改动)
- [第三部分 · 采纳的路线：X2-Turn 为 base 的 SoulX 范式](#第三部分--采纳的路线x2-turn-为-base-的-soulx-范式)
  - [3.1 为什么 base 必须是 X2-Turn 权重，而不是 Voxtral](#31-为什么-base-必须是-x2-turn-权重而不是-voxtral)
  - [3.2 支撑论证（五条，均有外部证据或代码依据）](#32-支撑论证五条均有外部证据或代码依据)
  - [3.3 总体架构](#33-总体架构)
  - [3.4 SoulX 部件到我们的逐项映射](#34-soulx-部件到我们的逐项映射)
  - [3.5 唯一不能照抄的地方：融合方式](#35-唯一不能照抄的地方融合方式)
  - [3.6 ci 头：三方案，且必须是判别式](#36-ci-头三方案且必须是判别式)
  - [3.7 视觉前端：三个选项（容量阶梯）](#37-视觉前端三个选项容量阶梯)
  - [3.8 时间对齐](#38-时间对齐)
  - [3.9 参数预算（必须与数据量绑定）](#39-参数预算必须与数据量绑定)
  - [3.10 训练配置](#310-训练配置)
  - [3.11 降级保证](#311-降级保证)
  - [3.12 张量形状](#312-张量形状)
  - [3.13 实验设计：同模型自带对照](#313-实验设计同模型自带对照)
  - [3.14 前置探针（动手前必做）](#314-前置探针动手前必做)
- [附录 A · 待核实项](#附录-a--待核实项)
- [附录 B · 关键外部引文](#附录-b--关键外部引文)
  - [B.1 ★ Kurata et al., Interspeech 2023 全文摘要](#b1--kurata-et-al-interspeech-2023-全文摘要)
- [附录 C · 一句话总结](#附录-c--一句话总结)

---

# 第一部分 · 两个基座的现状

## 1.1 X2-Turn 结构

`X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/modeling.py`（264 行，
文件头自述 *"Training-side definition for the Voxtral ASR + turn dual-head model"*）

```
                        audio 16 kHz
                            ▼
            ┌───────────────────────────────┐
            │ feature_extractor (mel)       │  hop=160 → 10 ms/mel 帧
            └───────────────┬───────────────┘
                            │ input_features [B, n_mels, T_mel]
                            ▼
    ┌───────────────────────────────────────────────┐
    │ Voxtral-Mini-4B-Realtime-2602                 │
    │  ├─ 因果音频编码器 ≈970 M（从零训练）           │  8 mel 帧 = 1 audio token
    │  └─ Mistral decoder ≈3.4 B                    │  ⇒ 80 ms / token
    │     （由 Ministral-3-3B-Base-2512 微调）        │  滑动窗口注意力
    └───────────────────────┬───────────────────────┘
                            │ hidden_states [B, L, H]
                            │ L = ceil(T_mel / 8)      inference.py:134
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌──────────────┐        ┌──────────────────┐
        │   lm_head    │        │   vad_lm_head    │  ★ X2-Turn 新增
        │    H → V     │        │      H → V       │  modeling.py:61
        │   （原生）    │        │  权重初始化 =     │  初始化自 lm_head
        └──────┬───────┘        │  lm_head 拷贝     │  的权重拷贝 :68-69
               │                └────────┬─────────┘
               ▼                         ▼
           ASR 转写              softmax 仅取 id 35-40
                                 inference.py:86
                                 TURN_CLASS_IDS = (35,36,37,38,39,40)
                                 idle / noidle / speaking /
                                 turn_end / backchannel / uncertain

loss = asr_loss + 0.1 × vad_loss              modeling.py:180
微调方式：全量（无 LoRA）
发布：x-square-robot/X2-Turn-4B-0812，Apache-2.0，权重不入源码仓
```

### 关键事实

| # | 事实 | 锚点 | 对我们的含义 |
|---|---|---|---|
| F1 | **帧长 80 ms**，运行时算出 `8 × 160 × 1000 / 16000` | `inference.py:118-120` | 视觉必须重采样到 12.5 Hz |
| F2 | 序列长度 `L = ceil(T_mel/8)` 即 audio token 数 | `inference.py:134` | 视觉序列与 hidden **天然 1:1**，无需对齐层 |
| F3 | 话轮读 `prefix_length + frame_index − 1` 位置 | `inference.py:165` | next-token 偏移，视觉插入需同步 −1 |
| F4 | **`vad_lm_head` 是全词表维度 H×V，只用 6 个 id** | `modeling.py:58-67` | **id 41+ 空闲，可直接征用 ⇒ 新头零参数** |
| F5 | **`train_vad_head_only` 冻结开关已存在** | `modeling.py:51,125-157` | 冻骨干训头的模式原作者已验证 |
| F6 | 推理两趟：先 `generate()` 出 ASR，再单次 forward 取 `vad_logits` | `inference.py:137-162` | 第二趟已 teacher-forced，视觉只需在此注入 |
| F7 | `transcription_delay_ms` 默认 480 ms（`default_num_delay_tokens=6`） | `inference.py:122` | **音频通路本身有 480 ms 前视** |
| F8 | **无训练循环 / 无 Dataset** | 全仓库检索 `Trainer\|AdamW\|optimizer.step\|DataLoader` 仅命中 1 测试文件 | 需自写 Trainer（≈150 行） |
| F9 | 时序决策实际在规则控制器 | `turn/controller.py`（429 行 if-else，12 个手调常数 `:42-56`，头注释 `Rules (user-approved)`） | 部署时 ci 判决须接进它（§3.13） |

> **F4 是本设计最有价值的发现**：`vad_lm_head` 已是全词表输出，
> `complete/incomplete` 可直接占用空闲 token id，**不新增任何头参数**。

## 1.2 SoulX-Duplug 结构

`SoulX-Duplug/model/model.py`（`State_Prediction_Model`，PyTorch Lightning）

```
        audio 16 kHz
            ▼
┌────────────────────────────┐
│ GLM-4-Voice WhisperVQ      │  ★ 冻结
│      Encoder               │    requires_grad=False + .eval()
└─────────────┬──────────────┘    model.py:53-58, config.yaml:4
              │ 离散 token
              ▼
        codebook(audio_tokens)      model.py:152
              │ audio_embed_dim
              ▼
┌────────────────────────────┐
│    EncoderProjector        │  ★ 可训（可选 freeze_projector）
│  audio_embed_dim → 2048    │    3 层 Linear + ReLU
│       → 2048 → llm_dim     │    model.py:17-35, 60-71
└─────────────┬──────────────┘
              │ audio_embeds
              ▼
   ┌──────────────────────────────────────────────┐
   │ inputs_embeds = audio_embeds ×  mask         │  ★ mask 二选一
   │               + text_embeds  × ~mask         │    model.py:159-160
   └─────────────┬────────────────────────────────┘
                 ▼
┌────────────────────────────┐
│  Qwen3-0.6B-expand_vocab_v2│  ★ LoRA r=32, α=64
│  llm_dim = 1024            │    model.py:105-114
│                            │    config.yaml:5-9
└─────────────┬──────────────┘
              ▼
      next-token prediction         model.py:162
      状态 = 8 个新增特殊 token（151675–151682）
      含显式 <|user_complete|> / <|user_incomplete|>
```

### 范式的三个要素

1. **冻结一个预训练 SSL 编码器**处理新模态 —— 注意是**学出来的表征**（WhisperVQ），不是手工特征（MFCC）
2. **可训 Projector** 把它对齐到 LLM 表征空间
3. **LoRA 微调主干**，让主干**学会解释**这个新信号

### 时间粒度与状态

| 参数 | 值 | 锚点 |
|---|---|---|
| `token_samples` | 1280 = **80 ms / token** | `model.py:50` |
| `chunk_size` | 2560 = **160 ms / chunk**（2 token） | `config.yaml:30` |
| `audio_back_size` | 15360 = 960 ms 回看 | `config.yaml:31` |
| `audio_ahead_size` | 640 = **40 ms 前视** | `config.yaml:32` |
| complete/incomplete 判决 | logit 比较 + `complete_bias=1.0` | `service/model.py:708-733` |
| 判 incomplete 后耐心 | `max_wait_num=10` → **1.6 s 硬超时** | `config.yaml:20` |

## 1.3 两者对比：为什么一个当 base、一个当范式

| 维度 | X2-Turn | SoulX-Duplug |
|---|---|---|
| 参数量 | **4 B** | 0.6 B |
| 骨干 | Voxtral-Mini-4B-Realtime | Qwen3-0.6B + LoRA |
| 音频前端 | 基座内置，**随主干全量训** | **冻结** WhisperVQ + 可训 projector |
| 输出粒度 | **80 ms 帧** | 160 ms chunk（含 2×80 ms token） |
| complete/incomplete | ⚠️ 折叠为 `turn_end`/`uncertain` | ✅ 显式 token |
| ASR | 内置联合训练（MTP 双头） | 外挂（Paraformer/SenseVoice） |
| 训练代码 | ❌ 无（仅模型定义） | ✅ `training-code` 分支 |
| 微调方式 | 全量 | **LoRA r=32/α=64** |
| **本项目角色** | **base 权重**（粒度最细、话轮表征最强） | **架构范式**（如何低成本接一个新模态） |

**分工的逻辑**：
- X2-Turn 提供**最好的起点表征**（80 ms 粒度 + 已被话轮监督塑造的 hidden）
- SoulX 提供**最好的扩展方法**（冻结前端 + Projector + LoRA，0.6B 上验证过）

---

# 第二部分 · 被否决的路线：端到端 Omni

## 2.1 端到端 Omni 路线的结构会是什么样

若走这条路，架构大致如下（以 MiniCPM-o 4.5 的 Omni-Flow 为原型）：

```
  video ──► SigLIP ViT ──► resampler ──► v_k  ┐
                          (1024→64 token)     │
                                              │  按时间 chunk 交织成单一因果序列
  audio ──► audio encoder ─────────────► a_k  ├──►  g_k = [v_k ; a_k ; o_k]
                                              │
  上一 chunk 输出 ─────────────────────► o_k  ┘
                            ▼
              ┌──────────────────────────┐
              │   Omni LLM (9.3 B)       │
              │   全量或大规模微调         │
              └────────────┬─────────────┘
                           ▼
                  每个 chunk 预测：
                  [listen] / [speak] 控制 token（LS 头）
                     ↓ 若 speak
                  内容 token（文本/语音）

  ⇒ 视觉在架构上参与「是否说话」的决策
  ⇒ 但 chunk = 1.0 s
```

**这条路线并非空想** —— MiniCPM-o 4.5（arXiv:2604.27393）已经这么做了，
且论文明确声称"自然支持主动行为、**减少对外部 VAD 模块的依赖**"（§3.2）。
理论上它就该覆盖我们的任务。

## 2.2 硬性否决理由：时间粒度

**MiniCPM-o 4.5 自己的消融表（Table 1）就是判决书**：

| Chunk Size | Boundary | Control | AdvBench | AlpacaEval | IFEval | SDQA | MMLU |
|---|---|---|---|---|---|---|---|
| **1.0 s** | Explicit | LS | 0.98 | **3.56** | **0.29** | **0.36** | **0.65** |
| **0.2 s** | Explicit | LS | 0.81 | **1.22** | 0.10 | **0.09** | 0.45 |
| **0.1 s** | Explicit | LS | 0.67 | 2.40 | 0.10 | 0.13 | 0.32 |

论文自述原因：**chunk 太短时模型在每个时间窗内没有足够信息做稳定决策。**

我们的需求是 **80 ms**：

```
Omni 稳定工作点     1000 ms  ████████████████████████████████████████
Omni 已崩溃         200 ms   ████████
我们需要             80 ms   ███
                             ↑ 差 12.5 倍，且 200 ms 处已崩
```

**为什么这是架构性的、不是调参问题**：把 1 s → 80 ms 需要改
chunk 划分、注意力掩码、位置编码 —— 这些是架构级属性，改了等于重训整个模型。

## 2.3 「换个小一点的 omni」为什么不成立

**先澄清：我否决 Qwen3-Omni 的理由不是"太大"，是粒度。** 换小的解决不了。

| 候选 | 视觉 | 时间粒度 | 结论 |
|---|---|---|---|
| **Qwen2.5-Omni-3B** | ✅ | ❌ **音频 2 s chunked prefill**，视频低 fps | **规模合适但粒度差 25 倍** |
| Qwen2.5-Omni-7B | ✅ | ❌ 同上 | 同上 |
| Qwen3-Omni-30B-A3B | ✅ | ⚠️ 首包 234 ms，**非帧同步** | 输出不是时间对齐的帧序列 |
| **Qwen3.5-Omni-Plus / Flash** | ✅ | ⚠️ 音频 **160 ms**（6.25 Hz）；视频 **1 FPS** | **见 §2.3.1 —— 否决理由是「权重未开放」，不是粒度** |
| MiniCPM-o 4.5 (9.3 B) | ✅ | ❌ 1.0 s，0.2 s 崩塌 | §2.2 |
| VITA-1.5 (7 B) | ✅ | ❌ | VideoFDB 实测 Overall **1.76**（人类 4.20） |
| Mini-Omni2 (**0.5 B**) | ✅ | ❌ | VideoFDB: **87% 回复是看图说话**（captioning collapse） |

```
Qwen2.5-Omni：  规模 30B → 3B   ✅ 缩小 10 倍
                粒度  2s → 2s    ❌ 完全没变
                需求      80ms   ✗ 仍差 25 倍
```

**时间粒度是架构属性，不随参数量缩放。** Mini-Omni2 只有 0.5 B，反而是 VideoFDB 里表现最差的。

### 2.3.1 ★ Qwen3.5-Omni 专项核查（原 V9，已解决）

**arXiv:2604.15804v2**（2026-04-17 提交 / 04-21 修订，Qwen Team）。
这是最新、最强、也是唯一**明确声称具备话轮能力**的 omni 模型，必须逐项核查。

#### 事实核查结果

| 维度 | 实际情况 | 对我们的影响 |
|---|---|---|
| **权重是否开放** | ❌ **仅 API**。原文：*"Qwen3.5-Omni is publicly accessible via **API**"*，脚注指向阿里云 Model Studio；HF/ModelScope 只有 Online Demo | **★ 致命。路线 B 结构性不可能** —— 无法加 projector、无法加 LoRA、无法读 hidden_states |
| 规模 | *"scales to **hundreds of billions** of parameters"*，Plus / Flash 两档，256k 上下文 | 即使权重开放也训不动 |
| 音频粒度 | AuT 编码器 **6.25 Hz token rate**；§2.3 明确 *"each output frame corresponds to approximately **160 ms**"* | **2 倍**于我们的 80 ms（比我原先估计的好，须如实修正） |
| 视频粒度 | 摘要：400 s 720P **@ 1 FPS**；§2.3 视频 temporal ID *"consistent temporal resolution of **160 ms** per ID"*，帧采样为 dynamic frame rate | **1 FPS 采样对人脸动态过粗**（唇动/注视需 ≥12.5 Hz） |
| **视觉编码器是否流式** | ❌ **Table 1 中 Vision Encoder = SigLIP2，Streaming 栏为「–」**；而 Audio Encoder / Thinker / Talker / MTP / Code2wav 全部为 ✓ | **视觉通路本身非流式** |
| 首包延迟 | Table 1：音频输入 Plus **435 ms** / Flash **235 ms**；视频输入 Plus **651 ms** / Flash **426 ms** | 视频输入使延迟增加 200–216 ms |
| Thinker TTFT（Table 2, 1 并发） | Flash **80 / 255 ms**（A/V）；Plus **162 / 377 ms**（A/V） | 视频路径 TTFT 是音频的 2.3–3.2 倍 |

#### ★★ 必须在论文里正面处理的一条声称

Qwen3.5-Omni 在 §1 的能力清单中明确写道：

> *"(2) comprehensive real-time interaction, encompassing **semantic interruption through
> native turn-taking intent recognition**, end-to-end voice control over volume, speed,
> and emotion, and voice cloning from user-provided samples."*

**这是我们任务的直接声称。** 但核查其 §5 全部评测：

| 评测类别 | 具体基准 | 是否含话轮/端点 |
|---|---|---|
| Text→Text | MMLU-Pro / IFEval / GPQA / LiveCodeBench / BFCL-V4 … | ❌ |
| Audio→Text | MMAU / MMAR / MMSU / VoiceBench / URO-Bench-pro / Fleurs / LibriSpeech … | ❌ |
| Vision→Text | MMMU / MathVista / VideoMME / MLVU … | ❌ |
| Audio-Visual→Text | DailyOmni / WorldSense / AVUT / AV-SpeakerBench / Qualcomm IVD / Omni-Cloze / OmniGAIA | ❌ |
| X→Speech | SEED-TTS / 多语种 TTS / 跨语言克隆 | ❌ |

**结论：215 个子任务中，没有一个评测话轮转换、语义打断或端点判断。
「native turn-taking intent recognition」是一条未被任何实验支撑的能力声称。**

#### 第三方评测提供了反证

**arXiv:2606.26083**《Real-Time Voice AI Hears but Does Not Listen》
（Bartelds, Bianchi, Zou，2026-06-24）评测了 GPT Realtime 2、Gemini 3.1 Flash Live、
**Qwen3.5 Omni Plus**、**Qwen3.5 Omni Flash** 四个生产级实时语音系统：

| 场景 | 副语言线索 | 系统决策 |
|---|---|---|
| 客服通话 | 来电者**在哭**，但口头说「我没事」 | **直接结束通话** |
| 金融授权 | 声音**充满恐惧**（疑似被胁迫） | **批准电汇** |
| 服务注册 | 语气**明显反讽** | 照字面办理 |

核心发现（作者命名为 **"emotional intelligence gap"**）：
**4 个系统中有 3 个在被直接询问时能可靠识别出苦恼/恐惧/反讽 —— 但在决策时忽略它们。**
显式 prompt 提醒只能部分且不稳定地改善。作者结论：这些系统的行为
*"近似于把语音降维成了一份文字转录稿"*。

> **注意这与 VideoFDB 的发现是同一个模式**：
> VideoFDB 发现 omni 模型存在 **visual-stream ignorance**（视觉流既不改变时机也不改变内容）；
> 2606.26083 发现同类模型存在 **paralinguistic ignorance**（能感知但不据此决策）。
> **两者共同指向「感知—行动脱节」是当前 omni 架构的系统性缺陷，而非个别模型的实现问题。**

#### V9 裁定

**Qwen3.5-Omni 不推翻 §2 的结论，但改变了否决理由的排序**：

```
原预判的否决理由：时间粒度太粗
实际的否决理由：
  ① 权重未开放（仅 API）        ← ★ 致命，路线 B 结构性不可能
  ② 数千亿参数                   ← 即使开放也训不动
  ③ 视觉编码器非流式（Table 1）
  ④ 视频 1 FPS 采样，对人脸动态过粗
  ⑤ 声称有 turn-taking 能力但零评测
```

**须诚实修正的一处**：Qwen3.5-Omni 的**音频**粒度是 160 ms（6.25 Hz），
不是我先前按 Qwen2.5-Omni 推断的 2 s。160 ms 恰好等于 SoulX-Duplug 的 chunk 粒度，
仅为我们目标（80 ms）的 2 倍 —— **在音频维度上，粒度已不再是主要障碍。**
真正的障碍转移到了「权重不可得」与「视觉通路非流式」。

**论文中的表述建议**：

> Qwen3.5-Omni [2604.15804] 声称具备 native turn-taking intent recognition，
> 但其 215 项评测中未包含任何话轮或端点任务，该能力未获实验支撑；
> 且第三方评测 [2606.26083] 显示该系列存在感知—行动脱节（能识别副语言线索却不据此决策）。
> 更关键的是，其权重未公开发布（仅 API 可用），无法作为可扩展的研究基座。


## 2.4 次级否决理由

即使粒度问题不存在，还有三条：

### ① 视觉引导话轮的能力从未被验证，且实测有害

MiniCPM-o 4.5 的全双工评测只在 **LiveSports-3K-CC** 上做，论文明确标注这是
**"audio-free full-duplex benchmark"（无音频）**。即：架构上视觉参与了 listen/speak 决策，
**但没有任何实验证明视觉改善了话轮判断**。

而 **VideoFDB**（arXiv:2605.30256，NVIDIA）做了配对实测：

| 发现 | 数字 |
|---|---|
| 加入视频流后时序对齐率 | **7 个模型中 6 个下降 0–5 pt** |
| MiniCPM-o 4.5 Conversational Flow | AV **3.54** < 纯音频 **3.76** |
| 视觉帧率 | **2 FPS 达峰**，8 FPS 3.04 → 10 FPS 2.81 |
| Mini-Omni2 失败模式 | 87% 回复是视觉描述（captioning collapse） |
| gpt-realtime-mini 失败模式 | AV 与纯音频输出互为改写（**visual-stream ignorance**） |

论文结论原文：**"加入视频没有让任何模型在实时容差内改善时序。"**

**同一缺陷在副语言维度上重现**（arXiv:2606.26083，评测含 Qwen3.5-Omni Plus/Flash）：
4 个生产级实时语音系统中 3 个能在被直接询问时可靠识别苦恼/恐惧/反讽，
**但在实际决策时忽略它们** —— 哭泣的来电者说「我没事」，系统直接结束通话。
作者称之为 **"emotional intelligence gap"**，并指出这些系统的行为
*"近似于把语音降维成了一份文字转录稿"*。

> **两条独立证据指向同一结论：「感知—行动脱节」是当前 omni 架构的系统性缺陷。**
> 模型能*看到*/*听到*线索，却不让这些线索影响*何时开口*的决策。
> 这正是为什么需要一个**以时序精度为第一目标的专用判别模块**，而不是更大的 omni 模型。

### ② 训练成本不可承受

端到端要么全量微调（9.3 B / 30 B，多机多卡），要么大规模 LoRA + 海量视听对话数据。
而我们的数据规模是**万级停顿事件**，撑不起这个量级的可训参数。

### ③ 没有现成的流式视听骨干可借

对 `audio-visual ∧ (streaming ∨ causal ∨ low-latency) ∧ (ASR ∨ speech recognition)`
全量检索 34 篇，结论是**「几乎没有专门做流式因果 AVSR 的工作，这是明显的研究空白」**，
真正涉及流式推理的仅约 6 篇，且无一可作为帧同步判别任务的骨干。

| 最接近的候选 | 为何仍不可用 |
|---|---|
| **Wan-Streamer** (2606.25041) | 160 ms 单元、block-causal、模型延迟 ~200 ms —— **但代码未开源**，且是交互生成模型非帧级判别 |
| **AV-Dialog** (2511.11124) | 40 ms 流式、视听、语义接地 —— **未开源**，配方是 128×A100 |
| AV-HuBERT (2201.02184) | 是 SSL 编码器，**不是 LLM 骨干**，无 text token 输出 |
| HI-AVSNN (2408.16564) | SNN + **事件相机**输入，非常规视频，规模小 |

## 2.5 核心论证：加模态是宽度改动，改粒度是架构改动

```
从 X2-Turn 出发：缺视觉
  → 视觉是一条旁路，可作侧信道注入（相加 / 拼接）
  → 时间网格、注意力模式、位置编码【全部不动】
  → 增量参数 O(10⁶–10⁷)，单卡可训
  → ✅ 可行

从 Omni 出发：粒度太粗
  → 1 s → 80 ms 要改 chunk 划分、注意力掩码、位置编码
  → 这些是【架构级属性】，改了等于重训
  → MiniCPM-o 自己的消融证明：0.2 s 就崩
  → ❌ 不可行
```

> **论文里的表述**：X2-Turn 的成功配方是「找一个时间结构已经匹配任务的基座，再加一个廉价的头」。
> 我们遵循同一配方，只是任务多了一个模态维度 —— **而模态可以侧接，时间结构不能。**

### 选型结论一览

| 判据 | 结论 | 依据 |
|---|---|---|
| 有更小的 omni 吗 | 有（Qwen2.5-Omni-3B、Mini-Omni2 0.5B） | §2.3 |
| 能用吗 | **不能**，粒度差 25 倍 | §2.3 |
| 否决理由是规模吗 | **不是** | §2.2 |
| **最新最强的 Qwen3.5-Omni 呢** | **不能** —— 权重仅 API、数千亿参数、视觉编码器非流式、视频 1 FPS | **§2.3.1** |
| omni 声称有话轮能力吗 | **有声称（Qwen3.5-Omni），但 215 项评测中零验证** | §2.3.1 |
| omni 的视觉引导话轮被验证过吗 | **没有**，且实测有害 | §2.4① |
| 有现成流式视听骨干可借吗 | **无** | §2.4③ |
| 非 omni 骨干 + 冻结编码器可行吗 | **可行，且能打赢原生 omni** | §3.2 证据 A |

---

# 第三部分 · 采纳的路线：X2-Turn 为 base 的 SoulX 范式

**一句话**：以 **X2-Turn-4B-0812 权重**为 base 全部冻结，
套 SoulX 范式侧接一路视觉（冻结前端 + 可训 Projector + LoRA），
在 `vad_lm_head` 空闲 id 上判别式输出逐帧 complete/incomplete。

## 3.1 为什么 base 必须是 X2-Turn 权重，而不是 Voxtral

这是一个容易被忽略但有实质意义的区别（本文档 v3 曾写错为"冻结 Voxtral"）：

| | Voxtral-Realtime 权重 | **X2-Turn-4B-0812 权重** |
|---|---|---|
| hidden 里有什么 | 转写所需的声学 + 词法信息 | **同上 + 已被话轮监督塑造过的表征** |
| `vad_lm_head` | **不存在** | 已训好，能判 `turn_end`/`uncertain` |
| 起点 | 需从零学习话轮概念 | **话轮概念已在 hidden 里** |
| 加载方式 | `VoxtralRealtimeForConditionalGeneration(config)` | **`load_mtp_checkpoint("x-square-robot/X2-Turn-4B-0812")`** |

X2-Turn 用全量微调把话轮信息压进了 hidden_states。我们要在**这个已含话轮语义的表征上加视觉**，
而不是退回 Voxtral 重走一遍。

**所以 base 是整个 `VoxtralMTP` 对象（含 `vad_lm_head`），全部冻结。**

## 3.2 支撑论证（五条，均有外部证据或代码依据）

### 证据 A：非 omni 骨干 + 冻结编码器 + Projector 可以打赢原生 omni

**Qwen-MusicAVQA-7B**（arXiv:2608.11329，2026-08）做法与我们**同构**：

```
冻结 Whisper encoder ──► 学习式线性 projector ──┐
（音乐轨与 TTS 问句各一个 projector）            ├──► Qwen2-VL-7B-Instruct
                                                ┘    靠预训练自注意力融合
                                                     无任务专用融合网络
```

| 系统 | MUSIC-AVQA 准确率 |
|---|---|
| **冻结 Whisper + projector + Qwen2-VL**（非 omni 骨干） | **95.9%** |
| **微调后的 Qwen2.5-Omni-7B**（原生 omni） | **80.9%** |

同数据、同输入；作者诚实标注为 system-level 比较，但 **+15 pt** 难以用噪声解释。
**训练成本：单张 A100 80GB，约 5 小时。**

#### ★★ 其核心发现正是我们的选型论据

> *"downstream accuracy tracks how much fine-grained local temporal information the audio representation preserves."*

同样 **32 token 预算**下：

| 表征 | 相对差距 |
|---|---|
| stride-pooled Whisper **帧序列** | 基准 |
| 全局池化 PANNs（编码器更大、看的音频更多、projector 远大得多） | **低 26 pt** |

且在 Whisper 内部，固定 token 预算下降低时间分辨率，代价相当
→ **不是「序列 vs 向量」的问题，是时间分辨率本身的问题。**

**写进论文的话术**：

> 决定下游性能的是表征保留了多少细粒度局部时间信息，而不是编码器多大、
> 也不是骨干是否原生 omni [2608.11329]。我们以 X2-Turn 为 base，
> 正因为它提供了唯一开源的 80 ms 帧同步因果话轮表征。

### 证据 B：判别式读出显著优于生成式，且单卡可行

**arXiv:2606.05713**（2026-06）的架构就是我们的 ci_head：
取 Qwen2.5-Omni-7B Thinker 的最后层 hidden → 轻量回归头 → 单次前向，**不走生成式解码**。

**受控对照**（固定骨干、数据、LoRA 配置，唯一变量是读出方式）：

| 读出方式 | CMU-MOSI MAE |
|---|---|
| **判别式头** | **0.551**（SOTA，Corr 0.888） |
| 生成式（同等监督训练后） | **>2 倍误差**；2.8% 输出无法解析或越界；延迟更高 |

作者结论：*"how an LMM is read out is as consequential as how it is trained."*

**工程配置**：QLoRA 4-bit、**单张 RTX 5090（32 GB）**、峰值显存 10–21 GB、**可训参数 1.14%**。

### 证据 C：SoulX 自身就是该范式在同一任务上的成功实例

SoulX-Duplug 做的是**同一个任务**（complete/incomplete 语义 VAD），
采用的正是"冻结前端 + Projector + LoRA"，且**没有全量微调主干**。

→ 这不是我们发明的方法，是同任务 SOTA 之一已验证的配方。我们只是把新模态从音频换成视觉。

### 证据 D：X2-Turn 自带冻结训练开关

`modeling.py:51` 的 `train_vad_head_only=True` 会把骨干包进 `torch.no_grad()`
（`:125-157`），只训话轮头。**原作者已在这个模型上验证过"冻骨干训头"可行**，我们扩展它即可。

### 证据 E：LoRA 解决了"冻结骨干读不懂视觉"的致命问题

这是纯冻结方案的结构性缺陷：往一个**从没见过非音频输入**的网络里注入视觉，
等于注入 OOD 扰动。最好情况被忽略（α 学到 ≈0），最坏情况干扰原有计算 ——
而后者正是 VideoFDB 观察到的现象（§2.4①）。

LoRA 是解药：主干获得少量自由度去**学习如何解释**视觉信号。
**SoulX 范式的核心价值不是"省参数"，而是"让主干配合新模态"。**

**因此三个候选臂重排为**：

| 臂 | 组成 | 定位 |
|---|---|---|
| ~~早融合 + 全冻结~~ | — | **删除**。最差组合：注入了但主干不会用 |
| **A（下界）** | 晚融合 + 全冻结 + 小头 | 消融基线，证明"不改主干也有一点效果" |
| **B（主力）** | **相加 + LoRA + 征用 id** | **SoulX 范式，出数字的那个** |

## 3.3 总体架构

```
 video 25/30 fps                              audio 16 kHz
      │                                            │
      ▼                                            │
┌──────────────────┐                               │
│ 视觉前端（冻结）  │  ★ frozen                     │
│  §3.7 三选项      │                               │
└────────┬─────────┘                               │
         │ [B, L, D_v]  重采样至 12.5 Hz            │
         ▼                                         │
┌──────────────────┐                               │
│ VisualProjector  │  ★ 可训 ≈0.8 M                 │
│  D_v→256→H       │    窄版（§3.9）                 │
│  末层零初始化     │                                │
└────────┬─────────┘                               │
         │ visual_embeds [B, L, H]                  │
         │            × α (零初始化)                 │
         │                                          │
 ┌───────┼──────── X2-Turn-4B-0812（冻结）──────────┼──────────────┐
 │       │                                          ▼              │
 │       │                          ┌───────────────────────────┐  │
 │       │                          │ Voxtral 因果音频编码器      │  │
 │       │                          │        ≈970 M              │  │
 │       │                          └─────────────┬─────────────┘  │
 │       │                                        │ audio embeds   │
 │       └────────────────► ( + ) ◄───────────────┘  [B, L, H]     │
 │                            │  逐位置相加（视听共时）              │
 │                            ▼                                    │
 │              ┌──────────────────────────────┐                    │
 │              │  Mistral decoder ≈3.4 B      │                    │
 │              │    + LoRA r=16/32  ★可训      │                    │
 │              └──────────────┬───────────────┘                    │
 │                             │ hidden_states [B, L, H]            │
 │        ┌────────────────────┼────────────────────┐               │
 │        ▼                    ▼                    ▼               │
 │  ┌──────────┐      ┌────────────────┐   ┌────────────────┐       │
 │  │ lm_head  │      │  vad_lm_head   │   │  vad_lm_head   │       │
 │  │  (冻结)  │      │   id 35-40     │   │   id 41,42     │ ★新增 │
 │  │          │      │    (冻结)      │   │  complete /    │  零参数│
 │  └────┬─────┘      └───────┬────────┘   │  incomplete    │       │
 │       │                    │            └───────┬────────┘       │
 └───────┼────────────────────┼────────────────────┼────────────────┘
         ▼                    ▼                    ▼
      ASR 转写           6 类话轮            ci 判决（逐帧 80 ms）
                      （原能力保留）              ★ 本项目目标

loss = 0 · asr_loss  +  0.1 · vad_loss  +  λ_ci · ci_loss
可训参数：VisualProjector (≈0.8 M) + LoRA (≈15 M) + [ci 头 0]
```

## 3.4 SoulX 部件到我们的逐项映射

| SoulX 部件 | 代码锚点 | 我们的对应物 |
|---|---|---|
| 冻结 WhisperVQ 前端 | `model.py:53-58` | 冻结视觉前端（§3.7） |
| 可训 `EncoderProjector` | `model.py:17-35, 63` | `VisualProjector`（窄版） |
| `freeze_projector` 开关 | `model.py:64-69` | 同名开关，用于 S1/S2 分阶段 |
| mask 混合融合 | `model.py:159-160` | **逐位置相加**（§3.5 —— 唯一不能照抄处） |
| LoRA 主干 | `model.py:105-114` | LoRA 加在 X2-Turn 的 Mistral decoder 上 |
| 8 个新增状态 token | `config.py:27-35` | 征用 `vad_lm_head` 空闲 id 41,42 |
| 载入 LoRA ckpt | `model.py:116-136` | 同构 |

| 要素 | SoulX-Duplug | AV-X2-Turn（我们） |
|---|---|---|
| 新模态 | 音频（进纯文本 LLM） | 视觉（进音频 LLM） |
| Projector 形状 | `audio_dim→2048→2048→1024` | `D_v→256→H`（窄版，§3.9 说明原因） |
| 主干 | Qwen3-0.6B + LoRA r=32/α=64 | **X2-Turn-4B** + LoRA r=16/32 |
| 输出粒度 | 160 ms chunk | **80 ms 帧** |

## 3.5 唯一不能照抄的地方：融合方式

```
SoulX：音频 token 与文本 token 占【不同序列位置】
       → mask 二选一
       inputs_embeds = audio_embeds * mask + text_embeds * (~mask)     model.py:160

我们：  视觉与音频【共时】，同一个 80 ms 格子
       → 不能二选一，只能相加
       inputs_embeds = audio_embeds + α * visual_embeds
```

这不是实现细节，是**模态关系的本质差异**：
SoulX 是"这个位置放音频还是放文本"，我们是"同一时刻同时有视听两路"。

相加也正是 **AV-Dialog** 的做法（`e_n = L_A(...) + L_V(V_n) + E(U_{n-1}) + E(T_{n-1})`），有先例。

**相加的三个额外好处**：
1. 序列长度不变 → KV-cache 不膨胀
2. 视觉缺失时置零 → **逐比特等价于纯音频模型**（§3.11）
3. α 零初始化 → 训练起点等价于原始 X2-Turn

### 注入点选择（三种做法，风险递减）

| 做法 | 风险 |
|---|---|
| 传 `inputs_embeds` | **最脆** —— 需知道库内部音频/文本嵌入合并逻辑 |
| 在 `base_model.model` 挂 forward pre-hook | 中等 |
| **在 decoder layer 0 挂 forward pre-hook 加到 hidden 上** | **最稳，推荐** —— 不依赖任何库内部构造细节 |

## 3.6 ci 头：三方案，且必须是判别式

| 方案 | 做法 | 新增参数 | 适用臂 |
|---|---|---|---|
| **H-A 征用空闲 token id** | 复用 `vad_lm_head`，`complete/incomplete` 放 **id 41,42**，单独对 {41,42} softmax | **0** | B |
| H-B 小型 2 类头 | `nn.Linear(H, 2)` 接 hidden | ≈ 6 K | B |
| H-C 拼视觉的 2 类头 | `MLP(H + D_v → 256 → 2)` | ≈ 0.8 M | A（晚融合唯一可行） |

**H-A 为何可能**：`vad_lm_head` 已是全词表输出（H×V ≈ 10⁸ 参数），只用 6 个 id，
**id 41 以上是空闲输出位**（F4）。推理侧 `inference.py:86` 只需换 id 列表。
→ **写进论文 implementation detail，这是「极低成本扩展」主张的硬证据。**
（V2 若不成立则退 H-B，+6 K 参数，无实质影响。）

### 为什么必须判别式，不能让模型「说」出判决

依据证据 B（§3.2）。三条含义：

1. ci 判决走 **logits 直出**，不做 `generate()`，不解析文本
2. 与流式需求天然吻合：每 80 ms 一个 logit 向量，无解码开销
3. 生成式还有个我们尤其不能接受的问题 —— **输出不可解析时无法降级**；
   判别头永远给出 well-formed 的 2 维分布

## 3.7 视觉前端：三个选项（容量阶梯）

严格遵循 SoulX 范式应用**预训练学习式编码器**而非手工特征（SoulX 用 WhisperVQ 不用 MFCC）。
但小数据下手工特征有优势，故三者并列做消融。

| | 选项 1：MediaPipe 几何 | 选项 2：视听 SSL 编码器 | 选项 3：摘 omni vision tower |
|---|---|---|---|
| 具体 | Face Landmarker → 24 维 | AV-HuBERT 视觉分支 / RAVEN 式复用 | Qwen2.5-Omni-3B 的 vision tower |
| D_v | 24 | 512–1024 | ≈1024–2048 |
| 前视 | **0** | 2 帧 @25Hz = **80 ms** | 0（逐帧编码） |
| 算力 | CPU 单核 | GPU | GPU，**最重** |
| 可解释 | **强**（可画"注视回避→延长等待"） | 弱 | 弱 |
| 容量 | 低 | 中 | **高** |
| 小数据表现 | **好** | 易过拟合 | **最易过拟合** |
| 缺失降级 | **天然**（检测失败即置零） | 需额外处理 | 需额外处理 |
| 符合 SoulX 范式 | ❌ 手工特征 | ✅ | ✅ |
| 先例 | Kurata'23、MM-VAP | RAVEN (2507.21448, 开源 CPU 实时)、AV-Dialog | **2608.11329**（跨模型摘编码器） |

### 选项 1 的 24 维特征构成

| 组 | 特征 | 维 | 假设作用 |
|---|---|---|---|
| **注视 ★** | 偏离摄像头角度、是否投向设备、回避持续时长 | 4 | 思考时回避/上移；说完时投向对方（Kurata'23 消融：去掉眼动 AUC **0.801 → 0.684**，掉幅最大，见附录 B.1） |
| **口型 ★** | 唇间距、闭合速度、是否完全闭合、闭合时长 | 5 | 未闭合 = 可能继续；闭合稳定 = 说完（Kurata'23：去掉 0.801 → 0.739） |
| 头部 | pitch/yaw/roll + 角速度 | 6 | 点头/转头（Kurata'23：去掉 0.801 → 0.758，三者中最弱） |
| 眉毛 | browInnerUp / browDown blendshape | 4 | 思考时常见皱眉（Kurata'23 未用，我们新增） |
| 面部动度 | landmark 速度 RMS | 2 | 区分「表情凝固」（MM-F2F 失败案例）与活跃（Kurata'23 未用） |
| 质量 | valid / confidence / 人脸框占比 | 3 | 降级门控 |

**与 Kurata'23 的 7 维对照**：它用 MediaPipe FaceMesh **478 landmarks → 7 维**
（瞳孔 x/y、张嘴 x/y、头姿 roll/pitch/yaw）+ 5 层单向 LSTM(hidden=15)。
**我们的 24 维是其超集**（多了眉毛、面部动度、质量门控、以及速度类导出特征）。
→ **选项 1 有直接文献先例，不是自创。**

窗口 = 过去 1.6 s（20 帧 @12.5 Hz），因果零前视，接小型因果 GRU(hidden 64)。

### 选项 3：从小 omni「摘」vision tower

```
Qwen2.5-Omni-3B ──► 仅取 vision tower（丢弃 Thinker/Talker）
                          │ 冻结
                          ▼
              空间池化（patch → 每帧 1 向量）  ← 必需
                          ▼
              时间池化 → 12.5 Hz              ← tower 原生非 80 ms 网格
                          ▼
                  VisualProjector（可训）
```

**优点**：表征已在视听-文本对齐上大规模预训练过（省掉 projector 一半工作）；
有直接先例（2608.11329 就是跨模型摘编码器再拼装，且打赢原生 omni）。

**代价与风险**：
- omni tower 每帧产多个 patch token，须先空间池化 —— 增加设计不确定性
- 算力最重，与 0.8 M projector 预算不匹配（需重估）
- **VideoFDB 警示**：视觉 2 FPS 达峰，密集视觉表征未必更好
- 许可与可独立加载性待核实（V10/V11）

### ★ 关键发现：80 ms 前视在这个 base 上是免费的

X2-Turn 的 `transcription_delay_ms` 默认 **480 ms**（`default_num_delay_tokens=6` × 80 ms，F7）。
即**音频通路本身就有 480 ms 前视**。

```
音频前视  ████████████████████████ 480 ms
视觉前视  ████ 80 ms  ← 完全被音频延迟预算吸收，系统总延迟不变
```

**推论**：早期设计里"必须零前视"的约束在这个 base 上是**过度约束**。
AV-HuBERT 的 80 ms 前视恰好 = 1 个 X2-Turn 帧，边际延迟成本为零。

⚠️ 待核实（V7）：turn head 是否同样吃 delay tokens。若 turn 头严格因果，则前视约束重新生效。

### 取舍逻辑

```
数据 < 5 K 事件   → 选项 1（MediaPipe）最可能赢，24 维不易过拟合
数据 5–20 K       → 选项 2（AV-HuBERT 唇部专精）
数据 > 20 K       → 选项 3（omni vision tower）容量才用得上
```

**若选项 1 就够 → 这本身是很好的结论**（端侧可部署、可解释、零 GPU 视觉开销）。
论文里应把它写成 finding 而非退让。

## 3.8 时间对齐

```
wall clock   0    80   160  240  320  400  480  560 (ms)
             ├────┼────┼────┼────┼────┼────┼────┤
audio token  │ a0 │ a1 │ a2 │ a3 │ a4 │ a5 │ a6 │   12.5 Hz
video 25fps  ▲▲   ▲▲   ▲▲   ▲▲   ▲▲   ▲▲   ▲▲      每 2 帧池化
video 30fps  ▲▲▲  ▲▲▲  ▲▲▲  ▲▲▲  ▲▲▲  ▲▲▲  ▲▲▲     每 2.4 帧插值/池化
visual_enc   │ v0 │ v1 │ v2 │ v3 │ v4 │ v5 │ v6 │   12.5 Hz
             └────┴────┴────┴────┴────┴────┴────┘
                        ↕ 逐位置相加
hidden 读取位置：prefix_length + i − 1      ← inference.py:165（next-token 偏移）
```

**视觉帧率上限 12.5 Hz** —— VideoFDB 实测 2 FPS 达峰，过高反而变差（挤占跨模态注意力预算）。

**必须写的单测**：构造只在第 k 帧非零的视觉输入，断言 ci_logits 变化恰好出现在预期位置。
错一帧 = 全局 80 ms 系统性偏差，**且在指标上表现为"视觉略有帮助"，极难察觉。**

## 3.9 参数预算（必须与数据量绑定）

### Projector 宽度不能照抄 SoulX

SoulX 的 `→2048→2048→1024` 是为 0.6B / 1024 维主干 + 大规模数据校准的。
我们主干 H 更大（待核实 V1），照抄会得到过宽的 projector：

| Projector 形状 | 参数量（设 H≈3072） |
|---|---|
| `24→2048→2048→3072`（照抄 SoulX） | ≈ **10.5 M** |
| `768→2048→2048→3072`（照抄 + AV-HuBERT） | ≈ **12 M** |
| **`24→256→3072`（窄版，推荐）** | ≈ **0.8 M** |
| **`768→256→3072`（窄版 + AV-HuBERT）** | ≈ **1.0 M** |

### ★ 与数据量的耦合（当前计划最大的内在矛盾）

| 停顿事件数 | 建议可训参数 | 配置 |
|---|---|---|
| ~2–3 K（Phase A 现定） | **0.5–1 M** | 窄 projector，**不加 LoRA** 或 r=4 |
| ~10 K | 3–5 M | 窄 projector + LoRA r=8 |
| **~30 K+（目标）** | 15–25 M | 宽 projector + LoRA r=16/32 |

**矛盾**：只标 2–3 K 事件撑不起 LoRA；而不加 LoRA 又回到"主干读不懂视觉"的困境（证据 E）。

**解法**：把 Phase A 目标提到**万级以上**。静音段候选本身数万级
（MM-F2F 有 51 K turn 实例），瓶颈在**人审而非弱标注**。所以：

> **弱标注全量铺开做训练集（万级），人审只做测试集（500–800）+ κ 抽检（300）。**

这样既撑起 LoRA，又不增加人力。

### 完整预算

| 组件 | 参数 | 状态 |
|---|---|---|
| Voxtral 因果音频编码器 | ≈970 M | 冻结 |
| Mistral decoder | ≈3.4 B | 冻结 + LoRA |
| lm_head / vad_lm_head | H×V ≈ 10⁸ 各 | 冻结 |
| 视觉前端 | 0（MediaPipe）/ 数十 M（AV-HuBERT）/ 数百 M（omni tower） | **冻结** |
| VisualProjector | 0.8–1.0 M（窄版） | 可训 |
| LoRA r=16 | ≈15 M（待核实 V1） | 可训 |
| ci 头（H-A 征用 id） | **0** | — |
| **可训合计** | **1–16 M**（随数据量选档），占骨干 **0.03–0.4 %** | |

**外部对照**：2606.05713 在 Qwen2.5-Omni-7B 上做同类判别任务，
可训 **1.14%**、峰值显存 10–21 GB、单张 RTX 5090 完成。我们更保守 → 预算可信。

## 3.10 训练配置

### 三阶段（对齐 SoulX 的 freeze 策略）

```
S1  只训 VisualProjector（主干全冻结，LoRA 关闭）
      → 对应 SoulX 的 freeze_projector 开关（model.py:64-69）
      → 目的：验证视觉信号能否被读出
S2  开 LoRA，联合训 Projector + LoRA
      → 主干学习解释视觉（证据 E）
S3  （可选）vad_lm_head 以 1/10 lr 微调 → 必过 WER 门禁
```

### 损失

```
loss = 0   · asr_loss                              # 冻结骨干，ASR 不训
     + 0.1 · vad_loss                              # 保留原权重，防旧能力漂移
     + λ_ci · CE(ci_logits[mask], ci_labels)       # λ_ci ≈ 1.0 起试
```

`ci_loss` 用 `pos_weight` 做**代价非对称**：
incomplete→complete（抢话打断）的代价 > 反向（多等一会儿）。这是产品语义，
必须体现在损失里，而不只是在推理阈值上。

### 监督铺满整段静音

```
frames:   ... speaking speaking │ sil sil sil sil sil │ speaking ...
ci_label: ...   -100     -100   │  1   1   1   1   1  │   -100
                                 └──── 同一事件标签 ────┘
```

好处：① 单事件从 1 个监督点变 3–13 个，数据量放大近一个数量级；
② 与流式推理形态一致 —— ci 头每 80 ms 都出，只在静音期被消费。

### 视觉 dropout（防退化）

参考 MM-F2F 的 RMDT：训练时以 p≈0.3 随机整段置零 `visual_valid`。
让模型学会没视觉也能干活 —— 这是把降级保证从「结构成立」升级为「统计成立」的关键。

## 3.11 降级保证

| 机制 | 保证强度 |
|---|---|
| α / Projector 末层零初始化 | 训练**起点**逐比特等价于原始 X2-Turn |
| `visual_valid=0` → 相加项为零 | 推理时视觉失效 → **等价于纯音频模型** |
| 视觉 dropout p=0.3 | 统计上不依赖视觉 |

**必须有确定性单测**：注入 `valid` 全零，断言输出等于 −视觉臂。

这一条同时回应三件事：IROS 审稿人必问的 robustness、VideoFDB 揭示的"视觉可能有害"、
以及消融实验的干净度。

## 3.12 张量形状

| 张量 | 形状 | 来源 |
|---|---|---|
| `input_features` | `[B, n_mels, T_mel]` | feature_extractor |
| `input_ids` | `[B, L]`，`L = ceil(T_mel/8)` | `inference.py:134` |
| `hidden_states` | `[B, L, H]` | decoder 输出 |
| `logits` | `[B, L, V]` | `lm_head` |
| `vad_logits` | `[B, L, V]` | `vad_lm_head` |
| **`visual_feats`** | `[B, L, D_v]` | 视觉前端 @12.5 Hz |
| **`visual_valid`** | `[B, L]` bool | 人脸可用性 |
| **`visual_embeds`** | `[B, L, H]` | VisualProjector |
| **`ci_labels`** | `[B, L]` ∈ {0, 1, −100} | −100 = 不计损失 |

## 3.13 实验设计：同模型自带对照

因为 loss 里保留了 `0.1 · vad_loss`，原生 6 类话轮能力不会漂移。这带来一个很干净的设计：

> **同一个模型同时输出「X2-Turn 原生 6 类话轮」与「新增 complete/incomplete」，
> 可直接比较两者在思考停顿上的表现差异 —— 无需跑两个模型。**

三层对照，全部零额外训练成本：

| 对照 | 做法 |
|---|---|
| −视觉 vs +视觉 | 同一权重，`visual_valid` 全零 即得 −视觉臂 |
| 原生话轮 vs ci 判决 | 同一次前向，读不同 id 组 |
| 模型判决 vs 规则控制器 | ci 输出接进 `controller.py`（下） |

### 与规则控制器的联动（部署层）

X2-Turn 真正决定"何时开口"的是 `turn/controller.py`（429 行 if-else + 12 个手调常数）。
ci 判决须接进去才能影响实际交互：

```
ci_logits ──► p(continue) ──► 调制 silence_end_frames (240 ms)
                                   tail_max_frames    (400 ms)
                              factor = 1 + g·conf·(2p−1)，clamp
```

**模型负责判决，控制器负责时序。** 仓库里已有外部信号注入先例可照抄：
`AcousticVoiceGate`（`server.py:102` → `on_frame(..., acoustic_active)`），
接入点全仓库仅一处（`server.py:115-120`）。

## 3.14 前置探针（动手前必做）

X2-Turn 的 hidden 里**已经有话轮信息**（它训过 `turn_end`/`uncertain`）。
所以先做一件成本极低但决定论文 framing 的事：

> 冻结 X2-Turn 抽 hidden → 训线性探针 → 预测 complete/incomplete，看 AUC。

| 结果 | 含义 | 行动 |
|---|---|---|
| AUC > 0.85 | 音频基线很强，视觉空间小 | 论文必须走**分层汇报**（思考停顿 × 噪声条件切片） |
| AUC ≈ 0.7 | 有真实空间 | 按主线推进 |

配套第二个探针：**仅用视觉 24 维**预测 complete/incomplete 的 AUC
（MM-F2F 的 Video-only 基准是 0.559，可作警示线）。

两个探针都不需要训练循环、不需要 LoRA，只要能跑通推理拿 hidden。

---

# 附录 A · 待核实项

| # | 待查 | 方法 | 影响 |
|---|---|---|---|
| V1 | `hidden_size` / `vocab_size` / 层数 | 读 HF `x-square-robot/X2-Turn-4B-0812` 的 `config.json` | Projector 与 LoRA 参数量 |
| **V2** | **id 41+ 是否真空闲** | 检查 Tekken 词表 41–50 | 决定 H-A 零参数方案；不成立则退 H-B（+6 K） |
| V3 | decoder layer 0 模块路径 | `print(model)` | 相加注入的 hook 挂点 |
| V4 | `−1` 偏移语义 | 单帧脉冲视觉输入单测 | 全局 80 ms 错位风险 |
| V5 | batch>1 与变长 padding 支持 | 试跑 | 训练吞吐 |
| V6 | 两趟推理能否合并为一趟 | 读 `generate()` 能否复用 KV cache | 部署延迟 |
| **V7** | **turn head 是否吃 delay tokens** | 对比不同 `delay_ms` 下 turn 帧时序 | 决定视觉前视约束能否放松（§3.7） |
| V8 | LoRA 是否伤 ASR | WER 门禁 | Δ ≤ +0.5% 否则回退 |
| **V9** | ~~Qwen3.5-Omni 的粒度与规模~~ | ✅ **已解决，见 §2.3.1** | **不推翻结论**。否决理由为：权重仅 API / 数千亿参数 / 视觉编码器非流式 / 视频 1 FPS。附带发现：其声称 native turn-taking intent recognition 但零评测，须在论文中正面处理 |
| V10 | Qwen2.5-Omni-3B vision tower 许可与可独立加载性 | 读 HF 模型卡 + 试只取 tower | 视觉选项 3 是否可用。⚠️ 注意 Qwen3.5-Omni 权重不开放，故选项 3 只能用 **2.5 版**的 tower |
| V11 | 该 tower 每帧 patch token 数与空间池化方式 | 试跑一帧 | 选项 3 算力与 projector 预算重估 |

---

# 附录 B · 关键外部引文

| 引文 | 用途 | 关键数字 |
|---|---|---|
| **2608.11329** Qwen-MusicAVQA-7B | 证明「非 omni 骨干 + 冻结编码器 + projector」可打赢原生 omni；**时间分辨率 > 编码器规模** | 95.9% vs 微调 Qwen2.5-Omni-7B 的 80.9%；同 token 预算下帧序列高 **26 pt**；单 A100 5 h |
| **2606.05713** Discriminative Hidden-State Readout | 证明**判别头 ≫ 生成式读出**；佐证单卡可行 | MAE 0.551 vs 生成式 >2×；**1.14%** 可训参数；RTX 5090 32 GB |
| **2604.27393** MiniCPM-o 4.5 / Omni-Flow | 证明 omni 路线**粒度缩小即崩塌** | 1.0 s 最优；0.2 s / 0.1 s 崩塌（Table 1）；LS > LT |
| **2604.15804** Qwen3.5-Omni | **最强 omni 也不可用的证据**；且其 turn-taking 声称无评测 | 数千亿参数、**仅 API**、AuT 6.25 Hz（160 ms）、视频 1 FPS、**Vision Encoder(SigLIP2) 非流式**、首包 A 435/235 ms、V 651/426 ms |
| **2606.26083** Real-Time Voice AI Hears but Does Not Listen | 证明 omni 的**感知—行动脱节**（与 VideoFDB 同一模式） | 评测含 Qwen3.5-Omni Plus/Flash；4 系统中 3 个能识别却不据此决策；"emotional intelligence gap" |
| **2605.30256** VideoFDB | 证明**视觉可能有害**，需降级保证；帧率不宜过高 | 加视频后 6/7 模型时序对齐下降；MiniCPM-o AV 3.54 < 音频 3.76；**2 FPS 达峰** |
| **2511.11124** AV-Dialog | 融合方式先例（embedding 相加）；视觉增益的条件性 | `e_n = L_A(...) + L_V(V_n) + ...`；clean **+1.3%** / 干扰说话人 **+13.0%**；120 ms 前视 |
| **2603.14877** SoulX-Duplug | **本方案的范式来源**，且是同任务实例 | 冻结 WhisperVQ + Projector + LoRA r=32/α=64 |
| **2608.10878** X2-Turn | **本方案的 base** | Voxtral-Mini-4B-Realtime 基座；80 ms 帧；6 类话轮 |
| **2505.12654** MM-F2F | 视觉前端选型（人脸裁剪 > 全画面 > 单帧 ViT）；RMDT 模态 dropout；**语义不完整+思考停顿是公认难例** | V-only **0.559**；T+A 0.811 → T+A+V 0.823 |
| **Kurata'23** (Interspeech, pp.2658-2662) | **最接近的先行工作**（视听 + 静音点）；视觉线索消融的对照假设；**同时是 C1 措辞的依据** | 见下方 B.1 全文数字 |
| **2504.09980** GRASS | **语义完整性标注的唯一先例**（但纯音频）；C1 的对照锚点 | κ=0.875；`incomplete-hold` 占 turn-hold **≈39%**（⚠️ 待复核，plan V12）；95 分钟；奥地利德语 |
| **2606.25041** Wan-Streamer | 最接近的流式视听骨干（但未开源） | 160 ms 单元 @25 fps；模型延迟 ~200 ms |

---

## B.1 ★ Kurata et al., Interspeech 2023 全文摘要

> Fuma Kurata, Mao Saeki, Shinya Fujie, Yoichi Matsuyama.
> *Multimodal Turn-Taking Model Using Visual Cues for End-of-Utterance Prediction
> in Spoken Dialogue Systems.* Interspeech 2023, pp. 2658–2662.
> DOI 10.21437/Interspeech.2023-578（Waseda Univ. + Chiba Inst. of Tech.）
> 全文已存 `research/kurata23_interspeech.pdf`，文本提取在 `research/kurata_extracted.txt`。

### 为什么它不与我们的 C1 冲突（关键）

论文 §3 原文定义标签：

> "the speaker **either continues speaking after IPU or gives the turn to the interlocutor**.
> We assigned the label of *"continue utterance"* in the former case and
> that of *"end utterance"* in the latter case."

→ **行为结果轴**，不是语义完整性轴。详细论证见 `implementation-plan.md` §0.1.1。

### 基本设置

| 项 | 值 |
|---|---|
| 语料 | Waseda 自建，**210 场 10 分钟在线面试**（日本英语学习者 × 英语教师），源自 Saeki'21 SLT 口语能力评测语料 |
| **是否公开** | ❌ 全文无发布声明；NEDO 项目 **JPNP20006** 内部语料 |
| 切分 | **IPU-based**，静音 **>300 ms**；从 IPU 末尾切 **5 s 片段**（含 IPU 结束后 0.3 s 静音） |
| 规模 | **21,728 片段** = 18,910 训 / 1,911 验 / 1,046 测 |
| 训练标签 | **规则自动生成**（VAD + 转写规则）；训练集 8,109 continue / 10,801 end |
| 测试标签 | 3 人标注多数投票；19 条无多数被剔除；**Krippendorff's α = 0.767**；530 continue / 497 end |
| 触发形态 | **IPU 触发，取末尾 2 s** → 反应式，**非帧同步流式** |
| 模型 | wav2vec2-base（音频）+ BERT-base-uncased（文本，≤512 tok 对话历史）+ X3d-S（视觉）→ **拼接 + 5 层 FC** → softmax。**不是 LLM** |
| 训练策略 | 两阶段：先各单模态头，**再冻结各子模块只训 FC** |

### 视觉特征（与我们 §3.7 选项 1 同源）

MediaPipe **FaceMesh 478 landmarks** → **7 维**，标准化到 μ=0/σ=1：

| 组 | 维度 |
|---|---|
| 瞳孔位置 | x, y（相对眼睛尺寸） |
| 张嘴程度 | x, y（相对脸部尺寸） |
| 头部姿态 | roll, pitch, yaw |

编码器：**5 层单向 LSTM**，hidden=15，+3 层 FC → 2 类。输入末尾 **2 s**。

### ★ 视觉线索消融（我们 §4.5.1 的对照序）

| 配置 | AUC | Acc | F1 |
|---|---|---|---|
| **全部特征** | **0.801** | 0.797 | 0.811 |
| **w/o 眼动** | **0.684** ⬅ 掉最多 | 0.678 | 0.723 |
| w/o 嘴部 | 0.739 | 0.733 | 0.770 |
| w/o 头姿 | 0.758 | 0.759 | 0.778 |

→ **眼动 > 嘴部 > 头姿**。这是我们视觉特征优先级的文献依据。

### 视觉编码器容量阶梯（我们 §4.5.2 的先例）

| 方法 | AUC | Acc | F1 |
|---|---|---|---|
| Non-E2E（FaceMesh 7 维 + LSTM） | 0.801 | 0.797 | 0.811 |
| **E2E（X3d-S 3D-CNN，人脸裁剪）** | **0.830** | 0.831 | 0.830 |

→ 高维端到端视觉特征优于低维工程特征，**但只 +2.9 AUC**。
这支持我们「MediaPipe 够用就是好结论」的立场（§3.7 取舍逻辑）。

### 模态组合（我们主表的外部参照）

| 输入 | AUC | Acc | F1 |
|---|---|---|---|
| Audio | 0.887 | 0.885 | 0.889 |
| Vision | 0.830 | 0.831 | 0.830 |
| Language | 0.827 | 0.826 | 0.827 |
| **Audio + Vision** | **0.917** | 0.918 | 0.917 |
| Audio + Language | 0.896 | 0.896 | 0.895 |
| Vision + Language | 0.836 | 0.835 | 0.833 |
| **Audio + Vision + Language** | **0.920** | 0.919 | 0.919 |

**视觉增益：A → A+V 为 +3.0；A+L → A+V+L 为 +2.4。**

### ⚠️ 引用这些数字时必须附带的警告

**它的音频基线是 wav2vec2-base（AUC 0.887），我们的音频基线是 X2-Turn 4B。**

**基线越强，视觉的边际空间越小。** 因此 **+3.0 不能当作我们的预期增益** ——
Kurata 的 +3.0 与 AV-Dialog clean 条件的 +1.3% 之间的差距，很可能主要来自基线强度差异。

→ **这加重了 §3.14 探针 A 的重要性**：若 X2-Turn hidden 的线性探针已达 0.90+，
视觉可捞空间极为有限，须立即切换到分层汇报 framing。

---

# 附录 C · 一句话总结

> 以 **X2-Turn-4B-0812 权重**为 base 全部冻结（含 `vad_lm_head`）；
> 冻结一个预训练视觉前端，用**窄 Projector** 把视觉对齐到 hidden 空间并**逐位置相加**（视听共时）；
> 用 **LoRA** 让 decoder 学会解释这路新信号；
> 在 `vad_lm_head` 的**空闲 token id 41,42** 上以**判别式**方式输出逐帧 complete/incomplete。
>
> 可训参数 **1–16 M**（随数据量选档），占骨干 **0.03–0.4 %**，单卡 24 GB 可训。
>
> —— 这是 **SoulX-Duplug 范式在视觉模态上的直接移植**，并有
> 2608.11329（非 omni 骨干可胜原生 omni、时间分辨率优先）与
> 2606.05713（判别头优于生成式、单卡可行）两项外部证据支撑。
