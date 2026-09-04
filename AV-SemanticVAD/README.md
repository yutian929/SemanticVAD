# AV-SemanticVAD — 融合视觉模态的语义 VAD

> Audio-Visual Semantic VAD / 首个带**语义完整性**标注的视听语料 + 冻结骨干视觉扩展
>
> 目标会议：**IROS** ｜ 语言：**中 + 英**
>
> 一句话：把 complete/incomplete 判定从纯音频扩展到视听 —— 建语料（C1）、改模型（C2）、上真机（C3）。

## 目录

| 文件 | 内容 |
|---|---|
| [`plan/implementation-plan.md`](plan/implementation-plan.md) | **实施计划单 v3.1**。Phase 0 探针 / Phase 1 数据 / Phase 2 模型 / Phase 3 训练 / Phase 4 Benchmark / Phase 5 真机+论文；**§0.1.1 C1 论证**、**§1.3 双轴标注体系**、Gate 表 / 风险单 |
| [`plan/architecture.md`](plan/architecture.md) | **架构设计 v4**。①两个基座现状 ②被否决的 Omni 路线及理由 ③X2-Turn 为 base 的 SoulX 范式（重点）；**附录 B.1 Kurata'23 全文摘要** |
| [`research/visual-semantic-vad-survey.md`](research/visual-semantic-vad-survey.md) | **主调研报告**。文献全景、量化证据综合、空白点清单、技术路线 |
| [`research/evidence-table.md`](research/evidence-table.md) | 证据速查表：每篇工作的输入模态 / 输出标签 / 是否流式 / 是否判语义完整性 / 关键数字 / 开源状态；**§A.1 标注轴对照** |
| [`research/methodology.md`](research/methodology.md) | 检索方法论、来源可信度评级、未核实项、与 `SoulX-Duplug/research/turn-taking-landscape.md` 的勘误 |
| `research/kurata23_interspeech.pdf` | Kurata et al. Interspeech'23 全文（最接近的先行工作），`kurata_extracted.txt` 为文本提取 |

## 方案：以 X2-Turn 为 base，套用 SoulX 范式

**base = `x-square-robot/X2-Turn-4B-0812` 权重全部冻结**（不是 Voxtral —— X2-Turn 的 hidden 里已含话轮表征）。
**范式 = SoulX-Duplug**：冻结前端 + 可训 Projector + LoRA。

三大贡献：
1. **AVSC-Corpus**：首个带**语义完整性**（complete/incomplete）标注的开源视听对话语料，中/英双语，**双轴标注**（弱标注全量铺开 + 测试集全人工 + 双人 κ）
2. **AV-X2-Turn**：冻结骨干上的轻量视觉扩展（Projector ≈0.8M + LoRA ≈15M + **征用空闲 token id 的零参数判别头**）
3. **真机系统评测**：错误打断率 / 响应延迟 / 固定阈值 Pareto / 降级安全

### ★ C1 为什么成立（Kurata'23 核实后，2026-09-04）

**最接近的先行工作是 Kurata et al. Interspeech'23** —— 它确实是视听 + 静音点触发。
但它标注的是**行为结果**，不是语义完整性：

> §3 原文：标签由「说话人**继续说**」还是「**让给对方**」决定
> → `continue utterance` / `end utterance`

这会把两种本质相反的情形合并：

| 话语 | 语义完整性 | Kurata 标签 | 系统接话 |
|---|---|---|---|
| "我想去巴黎…（800ms）…明年夏天" | complete | continue | **合法**（真 TRP） |
| "我想去…（800ms）…巴黎" | incomplete | continue | **硬错误** |

现有格局是**二选一**：GRASS [2504.09980] 有语义完整性标注但纯音频且仅 95 分钟；
Kurata'23 有视听但只有行为轴且语料未公开。
**我们的 C1 落在两者交集上，并同时记录两轴**（详见 `plan/implementation-plan.md` §0.1.1、§1.3）。

### 支撑路线的代码事实

1. **模型结构已开源**：`transformers/modeling.py` 的 `VoxtralMTP`（共享骨干 + `vad_lm_head`），文件头自述 "Training-side definition"；`forward(labels, vad_labels)` 完整，loss = asr + 0.1×vad。
2. **冻结骨干开关现成**：`train_vad_head_only=True` 包 `no_grad` 只训话轮头 —— 正是我们要的训练模式。
3. **缺的只有训练循环与 Dataset**：全仓库无 Trainer/Dataloader，需自写（HF Trainer ≈150 行）。
4. **话轮标签 = 词表 id 35–40**（`TURN_CLASS_IDS`）：complete/incomplete 作为新增第三头监督目标，不与现有标签冲突。

### 三条纪律（决定审稿生死）

| 纪律 | 原因 |
|---|---|
| **WER 回归门禁 Δ≤+0.5%** | 早融合触碰骨干输入路径；丢了转写质量话轮预测无从谈起 |
| **同构对照**（同权重 mask 掉视觉） | 一切增益声明的唯一可信来源 |
| **固定阈值 Pareto 图** | 正面回答「为什么不直接调大等待常数」 |

## 一句话调研结论

**存在视觉+音频联合判断话轮边界的模型（AV-Dialog、MM-F2F、MM-VAP、MuVAP、Kurata'23、MiniCPM-o 4.5 等），
但截至检索时点，不存在任何「视觉引导的 Semantic VAD」——
即没有工作把视觉特征接入一个 *流式的、逐帧输出「语义完整 / 语义不完整」判决* 的头。**

现有视觉话轮工作的输出全部落在**行为结果轴**上 ——
`shift/hold`（谁该说话）、`SOT/SOB/BC`（何时开口）、`continue/end utterance`（对方接没接话）；
而 X2-Turn 类语义 VAD 的输出在**语义完整性轴**上 —— `complete / incomplete`（这句话说完了吗）。
**两轴正交**：标签空间不同、监督信号不同、评测协议不同，是本方向的核心缺口。

> ⚠️ 原 `SoulX-Duplug/research/turn-taking-landscape.md` 中「检索未发现任何视觉引导工作」的结论
> **不成立**，详见 `research/methodology.md` §勘误。

## 关键量化速览（视觉带来多少增益）

| 工作 | 条件 | 纯音频 | +视觉 | Δ |
|---|---|---|---|---|
| AV-Dialog | 干净语音 | 73.2% | 74.5% | **+1.3** |
| AV-Dialog | 背景噪声 | 70.2% | 78.3% | **+8.1** |
| AV-Dialog | 干扰说话人 | 65.8% | 78.8% | **+13.0** |
| MM-VAP | 互静默期 shift/hold 平衡准确率 | 79% | 83% | **+4** |
| MM-VAP (noise) | 10dB 音乐噪声 | 52% | 72% | **+20** |
| MM-F2F | 三分类 macro-F1（含文本模态） | 0.811 | 0.823 | **+1.2** |
| Kurata'23 | continue/end AUC（A → A+V） | 0.887 | 0.917 | **+3.0** |
| Kurata'23 | continue/end AUC（A+L → A+V+L） | 0.896 | 0.920 | **+2.4** |

**读法**：视觉增益随「音频退化程度」单调放大，在干净音频 + 已有语言模态时增益仅 +1.2~1.3。
→ 视觉在现有范式里主要扮演**噪声鲁棒性 / 说话人归属**的角色，**尚未被证明能改善语义完整性判断本身**。

**⚠️ 增益与基线强度反相关**：Kurata'23 的 +3.0 建立在 **wav2vec2-base（AUC 0.887）** 这一较弱音频基线上；
我们的基线是 **X2-Turn 4B**。**不能把 +3.0 当作预期值** ——
这正是 Phase 0 探针 A（X2-Turn hidden 线性探针）必须先做的原因。

**对 IROS 路线的含义**：机器人工况恰好就是「音频退化」的一侧 —— 风扇/电机噪声、远场麦克风、旁边有人说话。
AV-Dialog 的 **+13%（干扰说话人）** 才是机器人的真实条件。因此可正当地聚焦噪声/远场，
但**必须单独诚实报告 clean 条件的增益**，否则会被 AV-Dialog Table 3 直接反驳。

## ⚠️ 两条必须主动讨论、不能回避的对照

1. **声学门控已能部分救援**。X2-Turn 的 `acoustic_vad_max_hold_frames=8` 在说话人停顿时有呼吸/填充音能量的情况下已能撑住 640ms。
   主张必须是「视觉泛化更广，因为呼吸声并非总是存在」，而不是「只有视觉能救援」。
2. **视觉可能有害**。VideoFDB（arXiv:2605.30256, NVIDIA）实测：加入视频流使 7 个系统中 **6 个**的时序对齐率下降 0–5 个百分点；
   MiniCPM-o 4.5 的 Conversational Flow 在 AV 模式下（3.54）**低于**纯音频模式（3.76）。
   → 这正是本方案坚持「乘性调制 + 缺省 1.0 + 恒等性测试」的原因。

## 状态

调研与计划已完成，尚未开始实现。下一步见 `plan/implementation-plan.md` §执行顺序建议。
