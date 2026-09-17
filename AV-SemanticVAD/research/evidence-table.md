# 证据速查表

> **判定标准（v2，2026-09-04 收紧）**
>
> **「是否判语义完整性」= 模型/标注是否有一个显式区分**
> 「说话人的话语在**语义/句法**上闭合了」与「尚未闭合」，
> **且该判定只依据静音点之前的内容**。
>
> ★ **必须与「行为结果轴」区分开**：
>
> | 轴 | 问的问题 | 典型标签 |
> |---|---|---|
> | **语义完整性轴**（我们要的） | 这句话闭合了吗？（只看之前） | complete / incomplete |
> | **行为结果轴**（大部分工作在做的） | 接下来实际发生了什么？（看之后） | hold/shift、continue/end、SOT/SOB、KEEP/TURN/BC |
>
> 两轴**正交**。行为结果轴会把「完整处的思考停顿」与「不完整处的思考停顿」
> 坍缩成同一类（前者接话合法，后者接话是硬错误）。
>
> 因此：仅输出 shift/hold、SOT/SOB、KEEP/TURN/BC、**continue/end utterance** 的
> 一律记为 ❌。

## A. 视觉 + 话轮/端点（核心检索目标）

| 工作 | ID / 出处 | 输入模态 | 输出标签 | 流式 | 粒度 | **判语义完整性** | 关键数字 | 开源 |
|---|---|---|---|---|---|---|---|---|
| **AV-Dialog** | arXiv:2511.11124 (UW+Meta, 2025-11) | 音频(DAC 16 codebook) + 视频(dlib→AV-HuBERT 唇部) | `<SOT>`/`<SOB>`/`<EMP>` + AVSR 文本流 | ✅ | **40 ms chunk**，视觉 25Hz | ❌（宣称 "semantically grounded" 但无 complete/incomplete 头） | Response Ratio 74.5/78.3/78.8%（clean/BG/interf）；视觉 Δ **+1.3/+8.1/+13.0**；算法延迟 **120 ms** | ⚠️ 仅项目页 |
| **MM-F2F** | arXiv:2505.12654 (ACL 2025, 厦大) | 文本(GPT-2) + 音频(HuBERT) + 视频(VideoMAE 人脸) | KEEP / TURN / BACKCHANNEL | ⚠️ 词帧级 | 词帧 | ❌（且 Limitations 明确承认在语义不完整+思考停顿上失败） | Acc: T .751 / A .751 / **V .559** / T+A .811 / **T+A+V .823**；BC F1 .894→.906 | ✅ github.com/Linyx1125/MM-F2F |
| **MM-VAP** | Findings ACL 2025 (2025.findings-acl.12); arXiv:2505.21043 | 音频 + 面部特征 | hold / shift（VAP 投影） | ✅ | VAP 窗 | ❌ | 互静默期平衡准确率 **79% → 83~84%**；表情贡献最大 | ✅ |
| **MM-VAP (noise)** | Interspeech 2025 pp.1073-1077; arXiv:2505.22088 | 同上 | hold / shift | ✅ | VAP 窗 | ❌ | 干净 **84%** → 10dB 音乐噪声 **52%** → 含噪训练多模态 **72%**；全噪声类型/SNR 均优于纯音频，**但不总泛化到新噪声** | ✅ |
| **Kurata et al.** | Interspeech 2023 pp.2658-2662；DOI 10.21437/Interspeech.2023-578；**全文已下载** `kurata23_interspeech.pdf` | 视觉(FaceMesh 7维+LSTM / X3d-S) + 声学(wav2vec2) + 言语(BERT)；拼接+5层FC | **`continue utterance` / `end utterance`** | ❌ **IPU 触发（静音>300ms），取末尾 2 s，非帧同步** | 事件级 | ❌ **行为结果轴** —— §3 原文：标签由「说话人继续说」还是「让给对方」决定，**与语义是否闭合无关** | **A .887 / V .830 / L .827 / A+V .917 / A+L .896 / A+V+L .920**；视觉消融 **全量 .801，去眼 .684，去嘴 .739，去头姿 .758**（眼>嘴>头姿）；E2E X3d .830 > LSTM .801 | ❌ **语料未公开**（NEDO JPNP20006 内部，Waseda 自建 210 场教师-学生在线面试） |
| **MuVAP** | arXiv:2606.16731 (Qi & Skantze) | 音频 + 人脸轨迹（多人） | 多方 VAP | ✅ | VAP 窗 | ❌ | 多方场景 + 配套视听语料 | ⚠️ |
| **MM-VAP (Cano)** | RO-MAN 2026; arXiv:2607.07294 | 视听 | VAP | ✅ | VAP 窗 | ❌ | HRI 场景（与 #3 同名不同工作） | ⚠️ |
| **Saga & Pelachaud** | arXiv:2506.03980 | 音频 + 人脸编码器 | VAP | ✅ | VAP 窗 | ❌ | — | ✅ |
| **Gaze-Enhanced Triadic** | Interspeech 2025; arXiv:2505.13688 (Meta Reality Labs) | 注视（智能眼镜第一视角） + 音频 | 下一说话人 | ⚠️ | — | ❌ | 三人对话 | ⚠️ |
| **Sign-LAP** | arXiv:2606.09424 | 手语视频 | VAP 迁移 | ✅ | — | ❌ | — | ⚠️ |
| **Let's Go Real Talk** | ACL 2024 (2024.acl-long.860) | 视听 → 视听 | 无话轮建模 | ❌ | — | ❌ | AV-Dialog 明确指出其"不建模话轮，不知何时该回应" | ✅ |

## A.1 ★ 标注轴对照（C1 的核心论证，2026-09-04 新增）

**结论**：现有工作在「语义完整性标注」与「视听」之间**二选一**，没有交集。

| 工作 | 语义完整性轴 | 视频 | 公开 | 规模 | 语言 |
|---|---|---|---|---|---|
| **GRASS** [2504.09980] | ✅ κ=0.875 | ❌ | ✅ | 95 分钟 | 奥地利德语 |
| **Kurata'23** | ❌ 行为结果 | ✅ | ❌ | 21.7 K 片段 | 日本人说英语（面试） |
| MM-F2F [2505.12654] | ❌ KEEP/TURN/BC | ✅ | ✅ | 51 K turn | 英文 |
| AV-Dialog [2511.11124] | ❌ SOT/SOB | ✅ | ⚠️ | — | 英语 |
| SoulX-Duplug-Eval | ✅ | ❌ | ✅ | — | 中文 |
| **AVSC-Corpus（我们）** | ✅ **+ 双轴** | ✅ | ✅ | ≥30 K 事件 | **中 + 英** |

### GRASS 的四格标注体系（我们双轴体系的来源）

| | 说话人继续 | 对方接话 |
|---|---|---|
| **句法完整点停顿** | `hold` | `change` / `question` |
| **句法不完整点停顿** | `incomplete-hold` | `trail-off` / `self-interruption` |

**Kurata'23 的坍缩**：

```
"continue utterance" = { hold , incomplete-hold }
"end utterance"      = { change , question , trail-off , self-interruption }
```

**为什么坍缩致命**：

| 话语 | 语义完整性 | Kurata 标签 | 系统接话 |
|---|---|---|---|
| "我想去巴黎…（800 ms）…明年夏天" | complete | continue | **合法**（真 TRP） |
| "我想去…（800 ms）…巴黎" | incomplete | continue | **硬错误** |

**GRASS 关键统计**（⚠️ 二手转述，投稿前须核原文 —— plan V12）：
- `incomplete-hold` 占全部 turn-hold **≈39%**
- turn-change 中句法不完整（trail-off / self-interruption）**≈17%**
- 完整/不完整二分的标注者一致性 **κ = 0.875**

→ 说明这不是边缘情况，行为标注混掉的是接近四成的样本。

**行为标注的第二个固有缺陷**：标签继承对话伙伴的个人习惯。
Kurata 是教师-学生面试，教师的耐心/礼貌会系统性偏置「对方是否接话」。

---

## B. Omni 全双工 + 视觉（有视觉→是否说话的通路，但未验证）

| 工作 | ID | 视觉参与「是否说话」决策 | 时间粒度 | 是否评测视觉引导话轮 | 关键数字 |
|---|---|---|---|---|---|
| **MiniCPM-o 4.5 / Omni-Flow** | arXiv:2604.27393 | ✅ **架构上是**：`g_k=[v^k;a^k;o^k]`，`[listen]` token，LS 控制头；自称"减少对外部 VAD 的依赖" | **1.0 s**（0.2s/0.1s 性能崩塌） | ❌ 全双工只测**无音频**的 LiveSports-3K-CC | LS>LT；Explicit>Implicit；SigLIP 0.4B, 64 tok/slice, 1–5 FPS；9.34B |
| **ELLSA** | arXiv:2510.16756 | ⚠️ | — | ❌ | 视觉+语音+动作全双工，SA-MoE |
| Qwen3-Omni | arXiv:2509.17765 | ❌ | — | ❌ | 非原生全双工 |
| Moshi | arXiv:2410.00037 | — 无视觉 | 80 ms | — | AV-Dialog 基线：Response Ratio 仅 54.0/53.8/52.5% |
| VITA-1.5 / Mini-Omni2 | — | ⚠️ | — | ❌ | VideoFDB 中最差（1.76 / 1.19） |

## C. ★ 负面证据：VideoFDB 实测（arXiv:2605.30256, NVIDIA）

237 段真实双人视频通话，11 类非语言动态，AV 与纯音频**同片段配对比较**。

| 模型 | Overall (AV) | Overall (仅音频) | Conv. Flow (AV) | Conv. Flow (仅音频) | TOR-Align (AV) |
|---|---|---|---|---|---|
| **人类参考** | **4.20** | — | **4.20** | — | **90% / 1400 ms** |
| MiniCPM-o 4.5 | 3.40 | **3.44** ⬆️ | 3.54 | **3.76** ⬆️ | 73% / 720 ms |
| Gemini 2.5 Flash Native | 3.17 | 3.17 | 2.81 | **2.98** ⬆️ | 72% / 3160 ms |
| gpt-realtime | 2.75 | **2.97** ⬆️ | 2.50 | 2.37 | 72% / 5400 ms |
| Gemini 3.1 Flash Live | 2.84 | — | 2.20 | — | 66% / 1720 ms |
| gpt-realtime-mini | 2.73 | — | 2.37 | — | 66% / 5320 ms |
| VITA-1.5 | 1.76 | — | 1.57 | — | 58% / 400 ms |
| Mini-Omni2 | 1.19 | — | 1.37 | — | 64% / 3080 ms |

⬆️ = 纯音频优于视听。

**核心结论**：
- 加视觉使 **7 个模型中 6 个的 TOR-Alignment 下降 0–5 个百分点**；论文原话：**"加入视频没有让任何模型在实时容差内改善时序。"**
- 失败模式 1 **Captioning collapse**：Mini-Omni2 **87%** 回复是视觉描述；VITA-1.5 出现能力免责声明 + **74%** token 重复。
- 失败模式 2 **Visual-stream ignorance**：gpt-realtime-mini 的 AV 与 A2A 输出互为改写，视觉既不改时机也不改内容。
- 视觉帧率 **2 FPS 达峰**，之后越多越差（8 FPS 3.04 → 10 FPS 2.81）。
- 级联数字人：Verbal Backchanneling 时序对齐 **0%**（架构性不可能）；延迟 2840–3520 ms。
- **"目前不存在公开可用的端到端全双工 AV2AV 系统。"**

## D. 音频侧语义 VAD 基线（本仓库，参照系）

| | SoulX-Duplug | X2-Turn |
|---|---|---|
| ID | arXiv:2603.14877 | arXiv:2608.10878 |
| 权重 | `Soul-AILab/SoulX-Duplug-0.6B` | `x-square-robot/X2-Turn-4B-0812` |
| 机构 | Soul AI Lab + SJTU X-LANCE + NPU | X Square Robot |
| 许可 | Apache-2.0 | Apache-2.0 |
| 骨干 | Qwen3-0.6B-expand_vocab_v2 + LoRA(r=32, α=64) | **Voxtral-Mini-4B-Realtime-2602** |
| 音频前端 | GLM-4-Voice WhisperVQ（冻结）→ codebook → 3层MLP projector | Voxtral 原生 |
| 输入模态 | **仅音频** | **仅音频**（全仓库零视觉代码） |
| 输出粒度 | 80 ms/token，**160 ms/chunk（=2 token）** | **80 ms/frame** |
| 前视 | 40 ms (`audio_ahead_size=640`) | 未公开 |
| 回看 | 960 ms (`audio_back_size=15360`) | — |
| **complete/incomplete** | ✅ `<\|user_complete\|>`(151676) / `<\|user_incomplete\|>`(151678) logit 比较 | ⚠️ 折叠为 `turn_end` / `uncertain` |
| 全状态集 | asr_bos, user_complete, user_backchannel, user_incomplete, assistant_backchannel, user_idle, user_nonidle, assistant_interrupt (151675–151682) | idle, noidle, speaking, turn_end, backchannel, uncertain |
| 对外状态 | idle / nonidle / speak / blank | 同上 6 类逐帧 |
| ASR | 外挂（Paraformer / SenseVoice） | 内置联合训练（MTP 双头） |
| 模型结构开源 | ✅ `model/model.py` | ✅ `voxtral-realtime/src/voxtral_realtime/transformers/modeling.py`（`VoxtralMTP` 双头，文件头自述 "Training-side definition"；含 `train_vad_head_only` 冻结骨干开关） |
| 训练循环/数据管线 | ✅ `training-code` 分支 | ❌ 无 Trainer/Dataloader（检索确认） |
| 话轮标签实现细节 | 8 个特殊 token（151675–151682），显式 complete/incomplete | 词表 id **35–40**（`TURN_CLASS_IDS`，6 类，无 complete/incomplete；推理仅对此 softmax） |
| 评测集 | ✅ `Soul-AILab/SoulX-Duplug-Eval` | — |
| 显存 | 低 | ≥24 GB |
| 兜底机制 | `complete_bias=1.0`、`max_wait_num=10`(**1.6 s**)、`max_mistake_num=3`、`far_field_threshold=0.02` | — |

**关键代码锚点（SoulX-Duplug）**

| 内容 | 位置 |
|---|---|
| `EncoderProjector`（audio_dim→2048→2048→1024） | `model/model.py:17-35` |
| `token_samples = int(0.08 * 16000)` | `model/model.py:50` |
| WhisperVQ tokenizer 加载（冻结） | `model/model.py:53-58` |
| projector 实例化 / freeze | `model/model.py:60-71` |
| `forward()` audio/text embedding 混合 | `model/model.py:145-166` |
| chunk / back / ahead size | `config/config.yaml:29-33` |
| 状态 token id | `config/config.py:27-35` |
| per-token loss rate | `config/config.py:118-131` |
| action token 注册 | `service/model.py:105-113` |
| 远场 RMS 门控 | `service/model.py:252-256` |
| embeds 序列拼装 | `service/model.py:~620` |
| **complete/incomplete 判决** | `service/model.py:708-733` |

**X2-Turn 锚点**：状态集 `README_zh.md:26-27`；80 ms/帧与期望输出 `README_zh.md:112-116`；Python API `README_zh.md:130-152`；Voxtral 基座致谢 `README_zh.md:223-225`；模型结构与双头 `transformers/modeling.py:15-69,114-188`；检查点加载 `:228-264`。

## E. 数据集

| 语料 | 规模 | 视听 | 话轮标注 | complete/incomplete | 可得 |
|---|---|---|---|---|---|
| **★ GRASS** [2504.09980] | **95 分钟**，奥地利德语 | ❌ 纯音频 | ✅ 四格（hold / incomplete-hold / change / trail-off …） | ✅ **κ=0.875** | ✅ |
| **★ Kurata'23 语料** | 210 场 ×10 分钟在线面试，**21,728 片段**（18,910/1,911/1,046） | ✅ | ✅ continue / end utterance（**行为轴**；训练集规则自动标，测试集 3 人标 α=0.767） | ❌ | ❌ NEDO 内部 |
| **MM-F2F** | **210 h**, 773 视频, 955 人, 169K utterance, 20M 帧, 1.5M 词帧, 51K turn, 22K BC | ✅ | ✅ KEEP/TURN/BC | ❌ | ✅ |
| **Seamless Interaction (InterAct)** | 大规模双人视听 (arXiv:2506.22554, Meta) | ✅ | ✅ turn 事件 | ❌ | ✅ |
| **VideoFDB** | 237 段, 130 人, ≥720p/30fps/24kHz | ✅ | ✅ 11 类非语言动态 + TOR | ❌ | 承诺发表前开源 |
| **SoulX-Duplug-Eval** | — | ❌ | ✅ | ✅ | ✅ HF |
| EgoCom | 38.5 h, 28 视频, 34 人 | ✅ | ✅ turn | ❌ | ✅ |
| Fisher / Switchboard / MapTask | 大 | ❌ | ✅ | ❌ | ✅ |
| VoxCeleb2 | 大（单人） | ✅ | ❌ | ❌ | ✅ |
| Real-TurnTurk | 土耳其语多模态 (arXiv:2608.22071) | ✅ | ✅ | ❌ | ⚠️ |

## F. 工业方案

| 方案 | 视觉进入端点判决 | 说明 |
|---|---|---|
| LiveKit turn detector | ❌ | 转写文本语义端点 |
| Pipecat Smart Turn | ❌ | 纯音频 |
| TEN Turn Detection | ❌ | 中英文本/音频 |
| Tavus CVI | ⚠️ 分离 | Raven-0 视觉感知 + Sparrow-0 话轮检测，两个独立模块 |
| Anam / Keyframe 数字人 | ❌ 架构性不可能 | VideoFDB: Verbal Backchanneling 对齐 **0%** |
| Gemini Live / gpt-realtime | ⚠️ 有视频输入 | VideoFDB 实测不改善时序 |

## G. 一句话判定汇总

| 问题 | 答案 | 最硬证据 |
|---|---|---|
| 有视频+音频判话轮边界的模型吗？ | **有** | AV-Dialog、MM-F2F、MM-VAP、MuVAP、Kurata'23 |
| **有视听 + *语义完整性* 标注的语料吗？** | **没有** | GRASS 有标注但纯音频 95 分钟；**Kurata'23 有视听但标的是行为结果**（§A.1） |
| 有视觉引导的 Semantic VAD（complete/incomplete 判决）吗？ | **没有** | 全部工作输出标签均在**行为结果轴**上 |
| **Kurata'23 会不会推翻我们的 C1？** | **不会** | 其 §3 标签定义为 continue/end utterance，由「对方是否接话」决定；坍缩了 complete/incomplete（§A.1）。**C1 措辞须换到「语义完整性 vs 行为结果」轴** |
| 视觉在干净音频下有用吗？ | **几乎没用**（但取决于基线强度） | AV-Dialog +1.3%；MM-F2F +1.2%；**Kurata'23 +3.0，但其音频基线仅 wav2vec2-base(.887)** |
| 视觉在噪声下有用吗？ | **很有用** | AV-Dialog +13%；MM-VAP 52%→72% |
| 把视频塞给 omni 大模型能解决吗？ | **不能，反而更差** | VideoFDB: 6/7 模型时序变差 |
| omni 模型迟早覆盖这个能力吗？ | **短期不会** | Omni-Flow 在 0.2s chunk 性能崩塌，粒度差 6–12 倍 |
| 语义不完整+思考停顿是公认难题吗？ | **是** | MM-F2F Limitations 明确记录该失败案例 |
| 最有价值但未被建模的视觉线索？ | **注视 / 手势** | **Kurata'23 消融：去眼动 .801→.684，掉幅最大**；AV-Dialog 仅唇部（自列 limitation）；MM-F2F 建议加身体动作 |
| 工程 24 维特征 vs 端到端视觉？ | **端到端只领先一点** | Kurata'23：FaceMesh 7维+LSTM **.801** vs X3d-S 端到端 **.830**，仅 +2.9 |
