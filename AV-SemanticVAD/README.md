# AV-SemanticVAD — 融合视觉模态的语义 VAD

> Audio-Visual Semantic VAD ｜ 目标会议 **IROS** ｜ 语言 **中 + 英**
>
> 把 complete/incomplete 判决从纯音频扩展到视听 ——
> 建语料（C1）、改模型（C2）、上真机（C3）。

**首次接手请先读 [`../START-HERE.md`](../START-HERE.md)。**

---

## 本目录的文件

| 文件 | 内容 | 何时读 |
|---|---|---|
| [`plan/implementation-plan.md`](plan/implementation-plan.md) | **★ 唯一权威计划**：Phase 0–5 全部任务、Gate 表、指标定义、风险单、待核实项 | 一直 |
| [`plan/architecture.md`](plan/architecture.md) | 架构设计与设计论证；附录 B.1 是 Kurata'23 全文摘要 | 写模型前 |
| [`docs/server-setup.md`](docs/server-setup.md) | GPU 服务器部署：环境、权重下载、磁盘估算、已知坑 | 上服务器时 |
| [`docs/code-anchors.md`](docs/code-anchors.md) | **★ 第三方参考代码位置的唯一权威来源**（含行号） | 写代码时 |
| [`research/evidence-table.md`](research/evidence-table.md) | 证据速查表：每篇工作的模态/标签/流式/关键数字；§A.1 标注轴对照 | 写论文、答审稿人 |
| [`research/visual-semantic-vad-survey.md`](research/visual-semantic-vad-survey.md) | 完整文献综述（791 行，背景资料） | 需要时查 |
| [`research/methodology.md`](research/methodology.md) | 检索方法论、来源可信度评级、未核实项 | 需要时查 |
| `research/kurata23_interspeech.pdf` | Kurata et al. Interspeech'23 全文（最接近的先行工作） | 写 Related Work |

---

## 方案一句话

**base** = `x-square-robot/X2-Turn-4B-0812` 权重**全部冻结**
（不是裸 Voxtral —— X2-Turn 的 hidden 里已含话轮表征）。

**范式** = SoulX-Duplug 的「冻结前端 + 可训 Projector + LoRA」。

**输出** = 在 `vad_lm_head` 的空闲 token id 上**判别式**输出逐帧 complete/incomplete。

详细论证见 [`plan/architecture.md`](plan/architecture.md) §3。

---

## 三大贡献

| # | 贡献 |
|---|---|
| **C1** | **AVSC-Corpus**：首个带**语义完整性**标注的视听语料（中/英，双轴标注） |
| **C2** | **AV-X2-Turn**：冻结骨干上的轻量视觉扩展（Projector ≈0.8 M + LoRA r=32 + 零参数判别头） |
| **C3** | **真机系统评测**：错误打断率 / 响应延迟 / 固定阈值 Pareto / 降级安全 |

**C1 的论证要点**（完整版见 `plan/implementation-plan.md` §0.1.1）：

最接近的先行工作 Kurata'23 确实是视听 + 静音点触发，但它标注的是**行为结果**
（`continue`/`end utterance`，由对方是否接话决定），会把两种相反情形合并：

| 话语 | 语义完整性 | Kurata 标签 | 系统接话 |
|---|---|---|---|
| "我想去巴黎…（800 ms）…明年夏天" | complete | continue | **合法**（真 TRP） |
| "我想去…（800 ms）…巴黎" | incomplete | continue | **硬错误** |

现有格局二选一：GRASS [2504.09980] 有语义完整性标注但纯音频且仅 95 分钟；
Kurata'23 有视听但只有行为轴且语料未公开。**C1 落在两者交集上。**

---

## 核心调研结论

**存在**视觉+音频联合判断话轮边界的模型（AV-Dialog、MM-F2F、MM-VAP、Kurata'23 等），
**但不存在**「视觉引导的 Semantic VAD」——
即没有工作把视觉接入一个**流式、逐帧输出「语义完整/不完整」判决**的头。

原因是两个正交的标注轴：

| 轴 | 问的问题 | 典型标签 | 谁在做 |
|---|---|---|---|
| **行为结果轴** | 接下来实际发生了什么？ | shift/hold、SOT/SOB、continue/end | 几乎所有视听工作 |
| **语义完整性轴** | 这句话闭合了吗？（只看之前） | complete/incomplete | 仅 GRASS、SoulX（**均无视觉**） |

> ⚠️ 原 `SoulX-Duplug/research/turn-taking-landscape.md` 中
> 「检索未发现任何视觉引导工作」的结论**不成立**，勘误见 `research/methodology.md` §5。

---

## ★ 视觉增益的量化现实（必须正视）

| 工作 | 条件 | 纯音频 | +视觉 | Δ |
|---|---|---|---|---|
| AV-Dialog | **干净语音** | 73.2% | 74.5% | **+1.3** |
| AV-Dialog | 背景噪声 | 70.2% | 78.3% | **+8.1** |
| AV-Dialog | 干扰说话人 | 65.8% | 78.8% | **+13.0** |
| MM-VAP | 互静默期 shift/hold | 79% | 83% | **+4** |
| MM-VAP (noise) | 10 dB 音乐噪声 | 52% | 72% | **+20** |
| MM-F2F | 三分类 macro-F1 | 0.811 | 0.823 | **+1.2** |
| Kurata'23 | A → A+V | 0.887 | 0.917 | **+3.0** |
| Kurata'23 | A+L → A+V+L | 0.896 | 0.920 | **+2.4** |

**读法**：视觉增益随音频退化程度单调放大；**干净音频下只有 +1.2~1.3**。
视觉目前主要扮演**噪声鲁棒性**角色，**尚未被证明能改善语义完整性判断本身**。

**⚠️ 增益与基线强度反相关**：Kurata'23 的 +3.0 建立在 wav2vec2-base（AUC 0.887）
这一**弱基线**上，我们的基线是 X2-Turn 4B ——
**不能把 +3.0 当预期值**，这正是必须先跑 Phase 0 探针 A 的原因。

**对 IROS 的含义**：机器人工况恰好在「音频退化」这一侧（风扇/电机噪声、远场麦克风、
旁边有人说话），AV-Dialog 的 **+13%** 才是真实条件。
因此可正当聚焦噪声/远场，但**必须单独诚实报告 clean 增益**。

---

## ⚠️ 两条不能回避的对照

1. **声学门控已能部分救援**。X2-Turn 的 `acoustic_vad_max_hold_frames=8`
   在停顿时有呼吸/填充音能量的情况下已能撑住 640 ms。
   主张必须是「视觉泛化更广，因为呼吸声并非总是存在」，**而非**「只有视觉能救援」。

2. **视觉可能有害**。VideoFDB [2605.30256] 实测：加视频使 **7 个系统中 6 个**
   时序对齐率下降 0–5 个百分点；MiniCPM-o 4.5 的 Conversational Flow
   在 AV 模式（3.54）**低于**纯音频（3.76）。
   → 这是本方案坚持「零初始化 + 视觉 dropout + 恒等性单测 + 降级保证」的原因。

---

## 状态

调研与计划已完成，**代码尚未开始**。
下一步见 [`../START-HERE.md`](../START-HERE.md) §4 或
`plan/implementation-plan.md` 的「立即执行的三件事」。
