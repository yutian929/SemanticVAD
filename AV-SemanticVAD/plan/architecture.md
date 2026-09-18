# AV-SemanticVAD 架构设计（v5 · 多人逐人 · 最简版）

> # ✅ 现行权威架构（2026-09-18）
> **输入 = 多人混合音频 + 单目 RGB 视频；对画面内每个人，逐 chunk 判「在不在说 · 说了什么(ASR) · 说完没(完整性)」。**
> 目标会议 **CVPR 2027**。骨干走 **Route A（热启动 Voxtral/X2-Turn，非从零训）**。
>
> - 总览图（可交互，含 X2-Turn/SoulX 参考图）：[`architecture-diagram.html`](architecture-diagram.html)
> - 方向转向与竞品调研：[`implementation-execution/phase2.md`](implementation-execution/phase2.md)
> - 三点贡献与竞品定位：[`implementation-plan.md`](implementation-plan.md) §0.1 / §0.1.1 / §5.2.3
> - 旧 v4（冻结单流，已否决端到端）已归档：[`architecture-legacy-v4.md`](architecture-legacy-v4.md)
>
> **⚠️ 尚未定：模型 forward 的输出形态**（见 §5）。本文其余部分已定；输出形态定稿后回填 §5 与图。
>
> **相对旧 v4 的变化**：单流→**多人逐人（per-face）**；纯音频完整性→**视觉做归属、音频做内容/完整性**；
> 旁挂一路视觉→**端到端联合**。**本版相对早期草案的简化**：输入只留**多人混合音频 + 单目 RGB**；
> 视觉只用**一个 RGB 编码器**（删去几何/唇双分支、addressee、身份 enroll、DoA、body-pose）；先做最简。

---

## 目录
- [第 0 部分 · 核心命题与任务定义](#第-0-部分--核心命题与任务定义)
- [第 1 部分 · 相较现有模型多了什么](#第-1-部分--相较现有模型多了什么)
- [第 2 部分 · 架构（到 forward 为止）](#第-2-部分--架构到-forward-为止)
- [第 3 部分 · 参考骨干：X2-Turn / SoulX 的 forward 输出](#第-3-部分--参考骨干x2-turn--soulx-的-forward-输出)
- [第 4 部分 · 从 logits 到结果（系统层，另做）](#第-4-部分--从-logits-到结果系统层另做)
- [第 5 部分 · ★ 待定：我们 forward 的输出形态](#第-5-部分--待定我们-forward-的输出形态)
- [第 6 部分 · 训练](#第-6-部分--训练)
- [第 7 部分 · 降级与竞品差异](#第-7-部分--降级与竞品差异)
- [第 8 部分 · 待定口子（TODO）](#第-8-部分--待定口子todo)
- [附录 · 复用的代码锚点](#附录--复用的代码锚点)

---

# 第 0 部分 · 核心命题与任务定义

## 0.1 核心命题（模态分工，探针实证）

| 证据 | 结论 |
|---|---|
| 探针 A（干净单人） | 音频完整性 AUC **0.99** → 音频判"说完没"够强 |
| 噪声 gate（竞争说话人） | 剔时长后 0.77→**0.47** → 混合/重叠下音频**塌** |
| 探针 B（视觉→完整性） | **0.42≈随机** → 视觉**不能直接读完整性** |
| 探针 B（视觉→说话人身份） | acc **0.649** → 视觉**能做归属** |

> **架构第一原则**：**视觉做归属（谁在说 / 声音归到哪张脸），音频做内容与完整性（说了什么 / 说完没）。**
> 视觉把"多人混合"还原成"可归属到某张脸的流"；完整性判断仍由强音频骨干承担。

## 0.2 任务定义（输入/能力）
- **输入**（流式、因果、80ms）：① 多人**混合音频**（16kHz）；② 单目 **RGB 视频**（多人在画面内）。
  **假设人一定在画面内**（不做画面外；不用麦阵/DoA/朝向要求）。
- **能力（系统交付）**：对**画面内每个人**逐 chunk 给出 `{在不在说 · ASR 转写 · complete/incomplete}`。
- **注意**：这是**系统层交付**；模型 forward 本身只吐 logits（见 §2/§4/§5）。

---

# 第 1 部分 · 相较现有模型多了什么

| | 现有模型（X2-Turn / SoulX） | 我们 |
|---|---|---|
| 输入 | 单人音频流（chunk） | **多人混合音频 + 单目 RGB** |
| 隐含假设 | 只有一个说话人 | 画面里有多人、可能同时说 |
| 能力 | ASR 转写 + 完整性 | **对每个人**：在不在说 · ASR · 完整性 |

**多出来的三个功能**（都由 RGB 里"谁的嘴在动"驱动）：
1. **视频归属**——把声音绑定到画面里正在说话的那个人（现有模型无"谁"的概念）。
2. **重叠鲁棒的 per-speaker ASR**——同时说话时靠各自唇动分给对的人分别转写（现有把混音转成一团糊）。
3. **per-speaker 完整性**——分别判每个人说完没（现有的完整性在混音上没有意义）。

---

# 第 2 部分 · 架构（到 forward 为止）

```
  多人混合音频 ─►[音频编码器(热启动)]─►[音频-LLM 骨干(热启动, 80ms/帧, 因果, LoRA)]─► hidden H[t]
                                                                                        │ K,V
  单目 RGB 视频 ─►[视觉编码器: 整帧RGB→内部人脸检测→每人 visual tokens]── 每人 token 作 Q ─┤
                                                                                        ▼
                                              ┌ per-face cross-attn 读出 (每人 Q attend H[t]) ┐
                                              │  变长 K · 每人独立 · 把混音里属于他的部分拎出来 │
                                              └───────────────────────┬───────────────────────┘
                                                                      ▼
                                     forward 输出 = 每人一组 logits（对 H[t] 的读出）  ← 模型到此为止
```

**四块**：
1. **音频编码器 + 音频-LLM 骨干**（热启动 X2-Turn/Voxtral）→ 逐帧 `H[t]`，已编码语言内容（供 ASR 与完整性；探针 A 证明完整性信息在 hidden 里）。
2. **视觉编码器**：吃**整帧 RGB**，内部定位人脸 → 每人一组 visual token。（输入就是 RGB，不是外部喂的 crop/几何。）
3. **per-face cross-attn 读出**：每人 visual token 作 query 去 attend `H[t]`（视觉"去 attend 音频的哪一段"= 归属）。变长 K、每人独立、共享权重。
4. **forward 输出 = 每人一组 logits**。模型 forward **到此为止**；如何把 logits 变成"在说?/转写/完整性"见 §4，且**我们的具体输出形态尚未定**（§5）。

---

# 第 3 部分 · 参考骨干：X2-Turn / SoulX 的 forward 输出

两个单流音频-LLM 是我们的直接参照（详图见 HTML）：

- **X2-Turn**（`VoxtralMTP.forward`）：**双头**，forward 返回两个整词表 logits：
  `logits` = `lm_head(H)` [B,T,131072]（ASR）；`vad_logits` = `vad_lm_head(H)` [B,T,131072]，
  **仅 id 35–40** 有意义（idle/noidle/speaking/turn_end/backchannel/uncertain）。`vad_lm_head` 是 `lm_head` 的拷贝。
- **SoulX-Duplug**（`State_Prediction_Model.forward`）：**单头**，forward 返回 `logits` [B,T,V_llm]；
  complete/incomplete 是**词表里的槽位**（生成式），混在同一 token 流里。

**共同点**：forward 只吐"逐位置的词表 logits"；都是**单流、无"人"的维度**。

---

# 第 4 部分 · 从 logits 到结果（系统层，另做）

**logit 本身只是"下一个 token 的打分"**。之所以能读出 complete/incomplete：① 完整性信息本就编码在 `H[t]`（探针 A 0.99）；② **训练用标签把某个词表槽位"指派"成该类**，CE loss 逼高对应槽位 → 之后"该槽 logit 高"= "模型认为闭合"。

| 模型 | logits → 转写 | logits → 完整性/话轮 |
|---|---|---|
| X2-Turn | 对 `logits` 自回归 generate → 去 PAD(32)/WORD(33)/id35–40 → detokenize | 每帧对 `vad_logits` 的 id 35–40 softmax → 6 类（turn_end/uncertain 承载完整性） |
| SoulX | 解同一条 token 流 → detokenize | 判决位置读 complete/incomplete 槽位 logits，取大 |
| **我们（待定）** | 每人 token 流（若做 per-face ASR） | 每人 hidden 读出槽位 —— **判别式 or 生成式，见 §5** |

---

# 第 5 部分 · ★ 待定：我们 forward 的输出形态

**这是当前唯一悬置的架构决策**（其余已定）。两个正交的岔路：

1. **粒度**：
   - (a) **K 套 per-face**：每张脸各自一套（ASR logits 流 + 状态） —— 最贴"逐人"，但 ASR 成本 ×K；
   - (b) **单流 ASR + 指派头**：一条 ASR + 一个"这帧属于哪张脸"的指派 —— 省 K 倍 ASR。
2. **状态读出**：
   - **判别式**（挂 hidden 上读保留槽位，仿 X2-Turn，不 generate；低延迟、可降级）；
   - **生成式**（状态当 token，仿 SoulX）。

**暂定倾向（未定稿）**：状态走**判别式**（与我们"可降级/低延迟"一致）；粒度 (a)/(b) 取决于是否必须逐人转写还是逐人只需状态+目标转写。**定稿后回填本节、§2 的 forward 输出框与 HTML 主图终点。**

---

# 第 6 部分 · 训练
- **S1**：冻骨干，训 视觉编码器 + per-face 读出 + 头 → 验证归属/状态可读、ASR 不坏。
- **S2**：骨干开 **LoRA r=32**（数据足可全微调）→ 一次前向联合训练。
- 损失：`asr_loss + Σ_k per-face 状态损失`；`incomplete→误判 complete`（抢话）代价加权。
- 视觉 dropout（p≈0.3 整段置零）→ 降级从"结构成立"升级为"统计成立"。
- 必测单测：恒等性（无视觉→逐比特等于纯音频骨干）、帧对齐脉冲响应、零初始化。

---

# 第 7 部分 · 降级与竞品差异
- **降级**：视觉失效（黑暗/无脸）→ 退化为纯音频骨干行为。
- **重叠的诚实边界**：两人重度重叠时混音 `H[t]` 退化（噪声 gate 0.47），视觉归属是"路由"机制，能否完全捞回完整性是**经验问题** → 低置信输出 **uncertain**，由恢复实验量化（phase2.md §2.2）。
- **与竞品**：MuVAP 把 N 人塌成 2 态、只判行为；AV-Dialog 单目标 + 行为事件 token；我们 **变长 K 真 per-face + 语义完整性**。

---

# 第 8 部分 · 待定口子（TODO）
1. **★ forward 输出形态**（§5）——当前唯一悬置的架构决策。
2. **视觉编码器选型**：整帧 RGB 编码器具体用什么（含内部人脸检测/跟踪的选型、是否复用预训练视觉/唇塔）。
3. **骨干适配**：LoRA vs 全微调——等数据量（自建 + AVCocktail）定。
4. **重叠段策略**：uncertain 起步；视觉引导分离作条件升级（依恢复实验）。
5. **"该回应谁"**：最简版交下游简单策略；addressee（跟不跟我说）作后续扩展（可加为第 4 个逐人输出）。

---

# 附录 · 复用的代码锚点
（前缀 `X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/`，只读；详见 [`../docs/code-anchors.md`](../docs/code-anchors.md)）
- `modeling.py:37-188` `VoxtralMTP`：共享骨干 + 双头范式（推广为 per-face 多头）。
- `modeling.py:60-69`：`vad_lm_head` 复制 `lm_head` 的手法。
- `inference.py:86` `_predict_turn`：对保留 id softmax 的判别读出。
- `inference.py:157-168`：一次前向读逐帧 logits 的循环。
- `inference.py:165`：`prediction_index = prefix_length + frame_index − 1` 帧对齐偏移。
- `modeling.py:125-145` `train_vad_head_only`：S1"冻骨干只训新头"的现成模式。
- SoulX `model/model.py:17/50/57/145-166`：Projector / 80ms token / 冻结前端 / 融合。
