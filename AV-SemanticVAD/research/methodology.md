# 调研方法论、来源评级与勘误

调研时间：2026-08-27。本文件记录检索过程、每条证据的核验层级、未能核实项，以及与既有文档的分歧。目的是让报告的每一条结论都可追溯、可复核、可反驳。

---

## 1. 研究问题分解

用户原始问题："目前有没有视觉引导的 Semantic VAD？有没有模型能输入视频流（或脸部特征流）+ 音频流，判断这个人有没有说完话、说话是否完整、是否可以接话？"

拆解为 5 个可检索子问题：

| # | 子问题 | 检索策略 |
|---|---|---|
| Q1 | 是否存在视听输入 + 语义完整性判决的模型？ | arXiv 全文检索 "semantic VAD/endpoint" ∩ visual；逐篇核验输出标签 |
| Q2 | 视觉话轮预测（PTTM/VAP）线做到什么程度？ | arXiv "voice activity projection" 全量列举 + turn-taking ∩ visual |
| Q3 | Omni 全双工大模型是否已隐含此能力？ | full-duplex ∩ (visual/video)；核验其 listen/speak 决策是否条件于视觉 |
| Q4 | 工业界有无落地方案？ | 产品文档/技术博客检索 |
| Q5 | 视觉到底带来多少增益？ | 提取所有可比的配对消融数字（音频-only vs 视听） |

**关键方法论决定**：把 Q5 单独立为一个子问题，而不是只回答"有没有"。因为"存在但无效"与"不存在"对立项决策的含义完全不同。事后看，这是本次调研最有价值的决定 —— VideoFDB 的负面证据只有在 Q5 框架下才会被认真对待。

---

## 2. 检索式与覆盖范围

### 2.1 arXiv API 结构化检索（优于关键词网搜，可穷举）

```
abs:"turn-taking" AND abs:"visual"
abs:"turn-taking"                                        （按日期降序全量扫描）
(abs:"endpoint detection" OR abs:"endpointing" OR abs:"voice activity detection")
   AND (abs:"visual" OR abs:"audio-visual" OR abs:"video")
abs:"semantic VAD" OR abs:"semantic endpoint" OR abs:"semantic voice activity"
(abs:"full-duplex" OR abs:"full duplex") AND (abs:"visual" OR abs:"video" OR abs:"audio-visual")
abs:"voice activity projection"                          （全量）
(abs:"gaze" OR abs:"head pose" OR abs:"facial expression" OR abs:"lip")
   AND (abs:"turn-taking" OR abs:"end of turn" OR abs:"turn taking")
```

### 2.2 全文核验来源

- **ar5iv.org**（LaTeX→HTML，表格可解析）— 主要手段
- **ACL Anthology** PDF
- **ISCA Archive**（Interspeech 论文页 + PDF）
- arXiv `/abs/` 摘要页（逐字引用摘要）

**踩过的坑**：直接抓 `arxiv.org/pdf/*` 返回 FlateDecode 压缩字节流，无法解析。改用 `ar5iv.org/abs/*` 后表格与 Limitations 章节均可完整提取。**后续调研应默认走 ar5iv。**

### 2.3 本地代码审计

对 `SoulX-Duplug/` 与 `X2-Turn/` 做只读审计，读取 README、`config/`、`model/model.py`、`service/model.py`，并全仓库检索 `vision|visual|video|image|frame|face|landmark|ViT|CLIP|siglip|mediapipe`。

**结论：两个仓库均无任何视觉相关实现。** SoulX 中 `model/glm_4_voice/speech_tokenizer/modeling_whisper.py` 的 `projector` 命中属 Whisper 分类头，与视觉无关。

### 2.4 覆盖范围与已知盲区

**覆盖**：arXiv（cs.CL/cs.SD/cs.CV/eess.AS）、ACL Anthology、ISCA Archive、主要工业产品文档。

**已知盲区**（诚实声明）：
- IEEE Xplore / ACM DL 全文未系统覆盖（HRI、ICMI、RO-MAN 部分工作可能遗漏）
- 日语/韩语本土会议（如日本人工智能学会）未覆盖 —— 考虑到 Kurata/Fujie/Matsuyama 组在此方向的持续产出，**这是最可能的遗漏源**
- 企业内部未公开系统（字节、Google、OpenAI）无法核实
- 专利文献未检索

---

## 3. 来源可信度评级

| 级别 | 定义 | 本报告中的工作 |
|---|---|---|
| **A：全文核验，表格逐格提取** | 通过 ar5iv/PDF 获取完整正文，实验表格数字逐格核对，Limitations 原文引用 | **AV-Dialog** (2511.11124)、**MM-F2F** (2505.12654)、**MiniCPM-o 4.5** (2604.27393)、**VideoFDB** (2605.30256)、**Kurata'23** (Interspeech 2023，2026-09-04 补全文，PDF 已存本地) |
| **B：摘要逐字核验 + 关键数字确认** | 摘要原文逐字引用，核心数字确认，正文细节未全部取到 | **MM-VAP noise** (2505.22088)、**Seamless Interaction** (2506.22554) |
| **C：部分正文/结论核验** | 获取到部分正文或权威页面，核心结论可确认，细节待补 | **MM-VAP** (Findings ACL 2025 / 2505.21043) |
| **D：检索列表级** | 通过 arXiv API 列表或搜索结果确认标题/ID/日期/摘要要点，未获取正文 | MuVAP (2606.16731)、MM-VAP Cano (2607.07294)、Saga & Pelachaud (2506.03980)、Gaze-Enhanced Triadic (2505.13688)、Sign-LAP (2606.09424)、ELLSA (2510.16756)、TurnBench (2608.25218)、Real-TurnTurk (2608.22071) |
| **E：本地一手代码/文档** | 直接读取仓库源码与 README | **SoulX-Duplug**、**X2-Turn** |
| **F：未核实** | 见 §4 | Tavus Sparrow-0 输入模态、Wan-Streamer / RoboEgo 细节 |

### 3.1 报告核心结论与其证据级别

| 结论 | 级别 | 依据 |
|---|---|---|
| 不存在视觉引导的 Semantic VAD（complete/incomplete） | **A** | AV-Dialog 标签集为 `<SOT>`/`<SOB>`/`<EMP>`（全文 §3.1.2 核验）；MM-F2F 标签为 KEEP/TURN/BC（全文 Table 3 核验） |
| 视觉在干净音频下增益 ≈ +1% | **A** | AV-Dialog Table 3（73.2→74.5）；MM-F2F Table 3（.811→.823） |
| 视觉在噪声下增益显著 | **A/B** | AV-Dialog Table 3（+8.1/+13.0，A 级）；MM-VAP noise 摘要（52→72，B 级） |
| 视觉对通用 omni 模型的时序是负增益 | **A** | VideoFDB 全文，6/7 模型 TOR-Alignment 下降，配对比较表逐格核验 |
| Omni-Flow 架构上视觉参与 listen/speak 决策 | **A** | MiniCPM-o 4.5 全文 §3.1–3.3，`g_k=[v^k;a^k;o^k]` + LS 控制头 |
| Omni LLM 在 0.2s 粒度性能崩塌 | **A** | MiniCPM-o 4.5 Table 1 逐格核验 |
| 语义不完整+思考停顿是公认难例 | **A** | MM-F2F §7 Limitations 原文引用 |
| 眼动 > 嘴部 > 头部 | **A** | **Kurata'23 全文核验（2026-09-04）**：消融表原文数字：全量 .801，去眼动 .684（掉最多），去嘴部 .739，去头姿 .758 |
| SoulX/X2-Turn 无任何视觉代码 | **E** | 全仓库关键词检索 |
| 工业界无视觉端点判决方案 | **C/F** | 产品文档级；Tavus 内部是否融合未能确认 |

---

## 4. 未能核实项（明确列出，不掩饰）

| # | 项目 | 缺口 | 对报告结论的影响 |
|---|---|---|---|
| F1 | **Tavus Sparrow-0 的输入模态** | 官方博客未能抓取；无法确认 Raven-0 的视觉信号是否进入 Sparrow-0 的端点判决 | 中等。若实际已融合，则"工业界完全没有"需下调为"工业界未公开细节"。**建议后续直接联系或抓取其 API 文档确认。** |
| F2 | **MuVAP 正文** | 仅列表级；Role-Relative Projection 机制、AVCC 语料规模、消融表未取到 | 低。不影响"无 complete/incomplete"的判定（VAP 范式定义上即无） |
| F3 | **MM-VAP (2505.21043) 完整表格** | 摘要称 84% vs 79%，ACL PDF Table 3 中互静默期为 83 vs 79，存在版本/取整不一致 | 低。报告中已同时给出"79% → 83~84%"区间而非单一数字 |
| F4 | **Real-TurnTurk 正文** | 仅列表级 | 低 |
| F5 | **TurnBench 的模态构成** | 未确认其 14 个系统中是否有视觉系统 | 低-中。报告标注为"音频"，若实含视觉需修正 §3.8 |
| F6 | **Wan-Streamer / RoboEgo** | 仅在检索列表中出现，未取正文 | 中。若 Wan-Streamer 真的"联合学习话轮管理 + 原生全双工音视频"，可能比 AV-Dialog 更接近。**建议优先补查。** |
| F7 | **Seamless Interaction 精确规模** | 摘要级；报告中未给具体小时数 | 低 |
| F8 | **X2-Turn / SoulX-Duplug 论文正文实验表** | 本次未取到 arXiv 全文表格；报告中的 SOTA 数字未引用，仅引用 README 与代码可确认的架构/接口事实 | **中。报告刻意回避引用两者的性能数字以避免臆造。若要写论文的 Related Work，需补齐。** |

---

## 5. 与 `SoulX-Duplug/research/turn-taking-landscape.md` 的勘误

| 原文档表述 | 判定 | 更正 |
|---|---|---|
| 检索未发现任何视觉引导的话轮/端点工作 | ❌ **错误** | 至少 11 项：AV-Dialog、MM-F2F、MM-VAP×2、MuVAP、Kurata'23、MM-VAP(Cano)、Saga、Gaze-Triadic、Sign-LAP、Let's Go Real Talk |
| 视觉 + 语义完整性判定是空白 | ✅ **正确** | 保留，并补上两条硬证据：AV-Dialog 输出标签为 SOT/SOB（非 complete/incomplete）；MM-F2F Limitations 承认该场景失败 |
| （隐含）加入视觉会带来明显增益 | ⚠️ **需修正** | 干净音频条件下仅 +1.2~1.3%；VideoFDB 显示对通用 omni 模型是 **负增益**（6/7 模型时序变差） |
| （隐含）omni 大模型尚未涉及 | ⚠️ **需修正** | MiniCPM-o 4.5 的 Omni-Flow **架构上**已让视觉参与 listen/speak 决策，并明确声称减少对外部 VAD 的依赖。真正的差距在**时间粒度**（1.0s vs 160/80ms）与**从未被评测**，而非"未涉及" |

**为什么原文档会错**（值得记录，避免重犯）：
1. 术语陷阱 —— 用 "semantic VAD" 检索，而学界几乎不用这个词（arXiv 上 `abs:"semantic VAD"` 命中极少）；相关工作用 "turn-taking prediction"、"end-of-utterance prediction"、"turn-boundary detection"、"endpointing"、"voice activity projection"。
2. 只搜"视觉+语义 VAD"的交集，而没有分别扫描"视觉+话轮"与"语义端点"两条线再做交叉。
3. 未做结构化 arXiv API 全量扫描，依赖关键词网搜（噪声大、召回低）。

**方法论教训**：**先建立任务分类学（本报告 §1 的 T1–T4），再针对每个任务分别穷举检索，最后做交叉。** 直接搜"目标能力"的组合词几乎必然漏检。

---

## 6. 结论稳健性自评

### 6.1 最可能被推翻的结论

| 结论 | 推翻条件 | 稳健度 |
|---|---|---|
| "不存在视觉引导 Semantic VAD" | 存在一篇 IEEE/ACM/日语会议论文，输入视听、输出 complete/incomplete。**F6（Wan-Streamer）与 §2.4 的日语会议盲区是最可能的来源。** | **中高**（AV-Dialog 若已做，其摘要会宣称；未宣称即强推论） |
| "视觉在干净条件增益 ≈ +1%" | 有工作在干净条件报告 >5% 增益 | **高**（三项独立工作数字一致：+1.3 / +1.2 / +2.4） |
| "视觉对 omni 模型时序是负增益" | VideoFDB 方法学被质疑（LM-as-judge、1 FPS 默认视频输入） | **中**（VideoFDB 自身列出局限：仅英语、仅单轮、评审上界受 LM 约束；且默认 1 FPS 可能低估视觉） |
| "Omni LLM 无法达到 160ms 粒度" | 有 omni 模型在 ≤200ms chunk 上不退化 | **中**（仅 MiniCPM-o 4.5 一个数据点） |

### 6.2 已做的反驳性检索（devil's advocate）

为避免确认偏误，主动检索了以下"若存在则推翻本报告"的方向：

- ✅ 检索 "semantic VAD/endpoint" ∩ visual → 无命中
- ✅ 检索 endpointing/EPD ∩ audio-visual → 命中均为 AVSR 或 VAD（T1/T2），无 T3
- ✅ 检索 full-duplex ∩ visual → 命中 omni 模型，逐一核验其输出标签，均无 complete/incomplete
- ✅ 逐字核验 AV-Dialog 的 "semantically grounded" 究竟指什么 → 指同时训 AVSR 流，输出侧仍是 SOT/SOB
- ✅ 主动寻找"视觉有大幅增益"的反例 → 只在噪声/干扰条件下找到，干净条件未找到
- ✅ 主动寻找"视觉有害"的证据 → 找到 VideoFDB（这条对立项定位影响最大，且是本次调研最有价值的发现）
- ⚠️ 未做：中文/日文本土会议的系统性检索（列为 §2.4 盲区）

### 6.3 对本项目最重要的一句方法论建议

报告中的 §4.3「假设 H」与 §6.6「Phase 0 最小验证实验」是本次调研的实质产出。

原因：现有全部证据都指向"视觉的增益来自噪声鲁棒性，不来自语义判断"。如果不先用 2 周成本验证"视觉是否携带 turn-holding intent 的独立证据"，就直接投入训练一个视听语义 VAD，**最可能的结局是在干净评测集上拿到 +0~1%，然后被 AV-Dialog Table 3 与 VideoFDB 同时反驳**。

**先证伪，再投入。**

---

## 7. 建议的后续补查清单

按优先级：

1. **Wan-Streamer / RoboEgo 正文**（F6）—— 可能是最接近的未核实工作
2. **Tavus Sparrow-0 输入模态**（F1）—— 决定"工业界是否已做"的结论
3. **X2-Turn (2608.10878) 与 SoulX-Duplug (2603.14877) 全文表格** —— 写论文 Related Work 必需
4. **MM-F2F 代码与数据实际可用性**（clone `github.com/Linyx1125/MM-F2F`）—— 210h 视听数据是 Phase 0 的最佳载体
5. **IEEE Xplore / ACM DL 检索**（HRI / ICMI / RO-MAN）—— 补 §2.4 盲区
6. **日语文献**（早稻田 Matsuyama / Fujie 组后续工作）—— Kurata'23 之后是否有延续
7. **VideoFDB 数据与代码**（承诺发表前开源）—— 现成的视听时序评测框架
