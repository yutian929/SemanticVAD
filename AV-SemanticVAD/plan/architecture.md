# AV-SemanticVAD 架构设计（v5 · 多人 per-face 端到端）

> # ✅ 现行权威架构（2026-09-18）
> **多人入境 · 端到端联合 AV 融合 · chunk 级 per-face 判「谁在说 × 对不对我说 × 说完没」。**
> 目标会议 **CVPR 2027**。骨干走 **Route A（热启动 Voxtral/X2-Turn，非从零训）**。
>
> - 方向转向与竞品调研：[`implementation-execution/phase2.md`](implementation-execution/phase2.md)
> - 三点贡献与竞品定位：[`implementation-plan.md`](implementation-plan.md) §0.1 / §0.1.1 / §5.2.3
> - 旧 v4（冻结单流，已否决端到端）已归档：[`architecture-legacy-v4.md`](architecture-legacy-v4.md)
> - 架构总览图（可交互）：[`architecture-diagram.html`](architecture-diagram.html)
>
> **从 v4 继承仍有效的论证**：判别式读出优于生成式；80ms 前视在该骨干上"免费"；时间粒度不可退让；
> Kurata 视觉线索消融序（眼>嘴>头姿）；参数量-数据量耦合。
> **相对 v4 的根本变化**：单流→**多人 per-face**；冻结→**LoRA/微调**；纯音频完整性→**视觉做归属/addressee、音频做完整性**；
> "旁挂一路视觉判 complete"→**端到端联合、按脸独立输出**。

---

## 目录
- [第 0 部分 · 核心命题与任务定义](#第-0-部分--核心命题与任务定义)
- [第 1 部分 · 总体架构](#第-1-部分--总体架构)
- [第 2 部分 · 组件规格](#第-2-部分--组件规格)
- [第 3 部分 · 时间对齐与流式](#第-3-部分--时间对齐与流式)
- [第 4 部分 · 参数预算](#第-4-部分--参数预算)
- [第 5 部分 · 训练](#第-5-部分--训练)
- [第 6 部分 · 降级保证与模态分工](#第-6-部分--降级保证与模态分工)
- [第 7 部分 · 与竞品的结构性差异](#第-7-部分--与竞品的结构性差异)
- [第 8 部分 · 待定口子（TODO）](#第-8-部分--待定口子todo)
- [附录 · 复用的代码锚点](#附录--复用的代码锚点)

---

# 第 0 部分 · 核心命题与任务定义

## 0.1 核心命题（模态分工，探针实证）

四个探针（见 phase0/phase1）连成的因果链决定了架构分工：

| 证据 | 结论 |
|---|---|
| 探针 A（干净单人） | 音频完整性 AUC **0.99** → 音频判"说完没"够强 |
| 噪声 gate（竞争说话人） | 剔时长后 0.77→**0.47** → 混合/重叠下音频**塌** |
| 探针 B（视觉→完整性） | **0.42≈随机** → 视觉**不能直接读完整性** |
| 探针 B（视觉→说话人身份） | acc **0.649** → 视觉**能做归属** |

> **架构第一原则**：**视觉做归属与 addressee（谁在说 / 对不对我说），音频做内容与完整性（说了什么 / 说完没）。**
> 每个模态只承担被实测证明擅长的轴。视觉把"多人混合"还原成"可归属的单人流"，完整性判断继续由强音频骨干承担。

## 0.2 任务定义 = I/O 契约（这也是 C1 基准的标注 schema）

**输入**（流式、因果、80ms 主时钟）：
- 全局单路**混合音频**（16kHz 波形）。
- **上游人脸跟踪器**给出的 K 条（变长）**per-face 轨迹**，每条含：唇/脸 crop + 头姿 + 注视 + 身姿朝向 + bbox 几何。
- （可选）每脸身份嵌入（家庭成员 enroll，绑定下游记忆）；（可选）麦阵 DoA。

**输出**（每 80ms 帧 × 每张脸 k，K 变长）：
- **核心 4 态**：`{IDLE / SPEAKING_TO_OTHER / ADDRESSING_ME_INCOMPLETE / ADDRESSING_ME_COMPLETE}`
- **分解辅助头**（供监督/消融，对应 L0/L1/L3）：`p_speak`、`p_addressee|speak`、`p_complete|speak`、`overlap/uncertain` 置信
- **目标说话人转写**（L2）：对"正在 addressing me"的那张脸出 text
- **稳定 per-face track-id**（供记忆绑定）
- **派生 L0 唤起闸** = `max_k p_addressee_k`

四态到机器人动作的映射：IDLE→忽略；SPEAKING_TO_OTHER→不打断/不应；ADDRESSING_ME_INCOMPLETE→继续听、别抢话；ADDRESSING_ME_COMPLETE→可应。

---

# 第 1 部分 · 总体架构

```
                         每 80ms 帧 (12.5Hz 主时钟, 因果 + delay-token 前视)
  混合音频 ─►[Voxtral 音频编码 (热启动)]─┐
                                         ├─►[Mistral decoder 26层 H=3072 (热启动)]──► 顶层 hidden H[t]
 (E2) 目标脸唇嵌入 ─►[cross-attn adapter ─┘   （已编码语言内容；ASR 路径）        │
       注入中段几层, 仅 addressing 时激活]                                        │
                                                                                 │
 ┌───────────────── 视觉前端 (新建/训练) ─────────────────┐                       │
 │ 几何分支: MediaPipe blendshape(口/眼/注视/眉)+头姿+身姿+bbox │─► g_k[t] (≈探针B 24维→MLP) │
 │ 唇分支:  预训练唇前端(AV-HuBERT 视觉塔, 冻结/轻调)         │─► l_k[t] (25Hz→重采样12.5Hz)│
 └────────────────────────────┬────────────────────────────┘                     │
                               ▼                                                  │
  每张脸 k:  v_k = [g_k ⊕ l_k]  ─ 作 Q ─►┌ face-query 读出块 (共享权重, 因果) ┐    │
                                          │  Q = v_k[t-W:t]  K,V = H[t-W:t] ◄──────┘
                                          └───────────────┬────────────────────┘
                                                          ▼ s_k[t]
   判别式多头 (不 generate):
     · 4态 softmax
     · 分解辅助头: p_speak · p_addressee|speak (主吃 g_k) · p_complete|speak (主吃 attend 到的 H)
     · overlap/uncertain 置信
   派生 L0 唤起闸 = max_k p_addressee_k ；  L2 = lm_head 在 E2 条件下出目标脸转写
```

**两个视觉入口，只有 E2 碰骨干内部**（沿用 X2-Turn"共享 hidden 上多读一个判别头"的哲学）：
- **E1 = per-face 状态读出**（不改骨干结构）：K 张脸的状态全部由挂在顶层 `H[t]` 上的 face-query 读出块产生。
- **E2 = 目标 ASR 条件化**（唯一碰骨干处）：被 addressing 的那张脸的唇流经 cross-attn adapter 注入中段几层，使 ASR 路径 target-speaker-aware。

---

# 第 2 部分 · 组件规格

## 2.1 骨干（Route A · 热启动）
- **VoxtralRealtime / X2-Turn-4B-0812 权重热启动**：音频编码器 → audio embeds 交织进 Mistral decoder（26 层，H=3072），`audio_length_per_tok=8` → **80ms/帧**。
- ASR 由原生 `lm_head` 承担；完整性从同一 `H` 读出（探针 A：完整性信息已在 hidden 里）。
- **不从零训**（从零训 ASR 对 CVPR 2027 时间线不可行）。S2 以 **LoRA r=32** 适配（数据足时可全微调，见 §8）。

## 2.2 视觉前端（新建/训练）

### 几何分支 → L0/L1（addressee / active-speaker）
- **MediaPipe FaceLandmarker**（探针 B 已用）：21 blendshape（口/眼/注视/眉）+ 3 头姿(yaw/pitch/roll) = 24 维 + **bbox 几何**（中心 x,y、尺度→距离/角度）+（增强项）**身姿朝向**（需 body-pose 估计器）。
- 小 MLP → `g_k[t]`。**CPU 可跑、可解释**；第一人称摄像头下"朝不朝镜头"= addressee 强信号。

### 唇分支 → L2/L3（target-ASR / 完整性微线索）
- **复用预训练唇前端**（首选 **AV-HuBERT 视觉塔**，冻结、可 S2 轻调），输出唇嵌入 25Hz → 重采样至 12.5Hz → `l_k[t]`。
- 唇同步给 E2 做"哪把声音属于这张脸"的软分离；唇微动辅助完整性。

## 2.3 face-query 读出块（C2 核心新模块）
- **输入**：`v_k[t-W:t] = [g_k ⊕ l_k]`，因果窗 W≈1.6s（20 帧）。
- **机制**：小型**因果 Transformer** 读出块，`Q = v_k[t-W:t]`，`K,V = H[t-W:t]`（骨干顶层 hidden）→ 融合状态 `s_k[t]`。视觉"去 attend 音频的哪一段"= 归属/消歧。
- **变长 K**：同一套**共享权重**对每张脸并行跑；**绝不塌成 floor-holder**（与 MuVAP 的结构性差异）。
- **脸间交互**（各脸互相 attend，建模"同一时刻单 floor"）：**默认关**，作消融项。

## 2.4 per-face 判别头（不 generate）
- 主：4 态 softmax。
- 辅：`p_speak`、`p_addressee|speak`（主吃 `g_k`）、`p_complete|speak`（主吃 attend 到的 `H`）、`overlap/uncertain` 置信。
- **判别式**（logits+softmax）→ 低延迟、可降级；沿用 `inference.py:86` 的读出模式。

## 2.5 目标说话人 ASR（L2）
- 被判 `ADDRESSING_ME` 的脸 → 其唇流经 E2 条件化骨干 → `lm_head` 出该脸转写。
- 默认**只转目标脸**（省算力）；转所有活跃脸为可选（见 §8）。

---

# 第 3 部分 · 时间对齐与流式
- **主时钟 80ms / 12.5Hz**：几何(≤30fps)池化到 12.5Hz；唇(25Hz)重采样到 12.5Hz；对齐骨干帧。VideoFDB：视觉 ≤12.5Hz 最优。
- **因果 + delay-token 前视**：沿用 X2-Turn `num_delay_tokens`（默认 6=480ms）做延迟-精度权衡；AV-HuBERT ~2 帧前视落在预算内。
- **逐帧读出索引**：沿用 `prediction_index = prefix_length + frame_index − 1`（next-token 偏移，`inference.py:165`）；错一帧=全局 80ms 偏移，须单测（脉冲响应）。

---

# 第 4 部分 · 参数预算

| 模块 | 参数 | 训练 |
|---|---|---|
| Voxtral 骨干 | 4B | 热启动；S2 开 **LoRA r=32 (~15–25M)** |
| 几何 MLP | ~0.1M | 训 |
| 预训练唇前端（AV-HuBERT 视觉塔） | ~20–30M | **冻结**（S2 可轻调） |
| face-query 读出块（共享） | 数 M | 训 |
| E2 唇注入 adapter | 小（LoRA 式） | 训 |
| per-face 头 | ~0 | 训 |

→ **可训参数 ~ 数十 M**；与 §0.4（≥30K 事件 ↔ LoRA r=32）耦合仍成立。H20 单卡富余。

---

# 第 5 部分 · 训练

## 5.1 两阶段
- **S1**：冻骨干，训 视觉前端 + face-query + per-face 头 → 验证 L0/L1 可读、ASR 不坏。
- **S2**：骨干开 LoRA + 挂 E2 唇注入 → **一次前向联合训练**。

## 5.2 损失
```
loss = asr_loss(目标脸)                       # 热启动 ASR，轻权保能力
     + Σ_k [ CE(speak_k) + CE(addressee_k) + CE(complete_k) ]
```
- **代价非对称**：`incomplete→误判 complete`（=抢话打断）罚重于反向（=多等）——进 loss，不只调阈值。
- **视觉 dropout**（p≈0.3 整段置零 `visual_valid`）→ 把降级从"结构成立"升级为"统计成立"。

## 5.3 必测单测
- **恒等性**：`visual_valid` 全零 → 逐比特等于 −视觉臂（纯音频骨干）。
- **帧对齐脉冲响应**：只在第 k 帧非零的视觉输入 → 输出变化恰在预期帧。
- **零初始化**：E1/E2 末层零初始化 → step 0 输出等于热启动骨干。

---

# 第 6 部分 · 降级保证与模态分工
- **降级保证**：视觉失效（无脸/遮挡/暗光）→ face-query 无有效 Q → 退化为纯音频骨干行为（恒等性单测保证）。
- **重叠说话的诚实边界**：混合音频的 `H` 在重叠时退化（噪声 gate 0.47）。E1 的视觉条件 attention 是"路由/归属"机制，**能否把声学上被叠掉的完整性捞回来是经验问题** → 输出 `overlap/uncertain`，并由 **恢复实验**（phase2.md §2.2：AMI/AVCocktail 单人 vs 重叠分档）量化。捞不回则重叠段走 uncertain，或升级视觉引导分离（备选）。

---

# 第 7 部分 · 与竞品的结构性差异
- **vs MuVAP**：它为套 VAP 把 N 人塌成"当前vs下一 floor-holder"2 态、只判行为；我们 **face-query 变长 K、真 per-face、判语义完整性 + addressee**。
- **vs AV-Dialog**：它单目标说话人 + 行为事件 token（`<SOT>/<SOB>`）；我们**多人 per-face + 语义完整性判决**。
- **vs MM-VAP/AVCocktail**：它们 VAP 未来语音活动/无完整性/无 ASD 任务；我们把 **addressee × active × completeness** 合成 per-face 联合判决。

---

# 第 8 部分 · 待定口子（TODO）
1. **唇前端选型**：AV-HuBERT 视觉塔 vs Auto-AVSR/VSR 唇编码器——按可得性/许可 + 因果流式支持定。
2. **身姿来源**：先做 脸+注视+头姿+bbox；身姿朝向（需 body-pose 估计器，如 MediaPipe Pose）作增强项后加。
3. **骨干适配**：LoRA vs 全微调——等数据量（自建+AVCocktail）定；先按 LoRA 设接口。
4. **ASR 转写范围**：默认只转目标脸；转所有活跃脸为可选。
5. **脸间交互**：默认独立读出；inter-face attention 作消融。
6. **重叠段策略**：uncertain 起步；视觉引导分离作条件升级（依恢复实验结果）。

---

# 附录 · 复用的代码锚点
（前缀 `X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/`，只读；详见 [`../docs/code-anchors.md`](../docs/code-anchors.md)）
- `modeling.py:37-188` `VoxtralMTP`：共享骨干 + 双头范式（我们推广为多 per-face 头）。
- `modeling.py:60-69`：`vad_lm_head` 复制 `lm_head` 的手法（判别头 warm-start 参考）。
- `inference.py:86` `_predict_turn`：对保留 id 做 softmax 的判别读出（我们的 per-face 头照此形态）。
- `inference.py:157-168`：一次前向读逐帧 logits 的循环（per-face 读出挂在此形态）。
- `inference.py:165`：`prediction_index = prefix_length + frame_index − 1` 帧对齐偏移。
- `modeling.py:125-145` `train_vad_head_only`：S1"冻骨干只训新头"的现成模式。
