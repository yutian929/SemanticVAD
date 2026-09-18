# 实施计划单

> **本文件是项目的唯一权威计划。** 与其他文档冲突时以本文件为准。
>
> **⚠️ 方向转向（2026-09-18，CVPR 主目标确认）**：本项目已从「单说话人 · 冻结 X2-Turn 旁挂一路视觉 · 判 complete/incomplete」
> 转向 **「多人入境 · 端到端联合 AV 融合 · chunk 级 per-face 判「谁在说 × 语义完整性」」**，目标会议改为 **CVPR 2027**。
> 依据见 [`implementation-execution/phase2.md`](implementation-execution/phase2.md)（方向转向 + 深度竞品调研 + AV-Dialog 深挖）。
> **下方 Phase 1–5 的具体架构（冻结骨干 / 单 ci 头 id 41,42 / SoulX 范式 / MM-F2F 英文主训练）多数将被重修** ——
> 放开冻结范式、从头构建+训练、数据集自建；**架构与 Phase 细节待 [`architecture.md`](architecture.md) 讨论后更新**。
> 在此之前，§0.1 三点贡献与竞品定位（§0.1.1 / §5.2.3）**已按新方向重写、为最新权威**；Phase 1–5 旧文暂留作历史参照。
>
> **旧方案（历史，待重修）**：以 X2-Turn-4B-0812 冻结为 base，套 SoulX-Duplug 范式侧接一路视觉，在 `vad_lm_head` 空闲 id 上判别式输出逐帧 complete/incomplete。
>
> | | |
> |---|---|
> | 目标会议 | **CVPR 2027**（截稿约 2026-11；原 IROS 已切换） |
> | 算力 | **双卡 H20（96 GB/卡）** |
> | 数据 | **自建多人 per-face 双轴 AV 语料**（active-speaker × 语义完整性）＋ 复用/扩展 AVCocktail 做评测（可行性待核，见 phase2.md 待办） |
> | 架构细节 | [`architecture.md`](architecture.md)（**待按新方向重写**） |
> | 服务器部署 | [`../docs/server-setup.md`](../docs/server-setup.md) |
> | 代码位置索引 | [`../docs/code-anchors.md`](../docs/code-anchors.md) |
>
> **最后更新**：2026-09-18
> **执行 changelog**：
> - **2026-09-18** — ★ **方向转向 CVPR 2027**（多人 per-face AV：谁在说 × 语义完整性）。放开冻结范式、从头训练、数据集自建。C1/C2/C3 与竞品表（§0.1 / §0.1.1 / §5.2.3）已重写；深度竞品调研 + AV-Dialog 全文深挖见 phase2.md。Phase 1–5 架构待 architecture.md 讨论后重修。
> - **2026-09-14** — Phase 0 探针跑完，对 C2 卖点有重大影响，见下方「执行发现」。
> - **2026-09-16** — ⚠️ 核实原始来源订正三处：① **MM-F2F 是英文语料，非中文**（README/论文原文 "in-the-wild online **English** conversation videos"）；② MM-F2F **不发布媒体**，只发标注 CSV + YouTube 链接 + 脚本，视频须按 `video_id` 自下（生 YouTube，**非去标识化**）→ 原「用它可绕开人脸合规风险」的结论**不成立**；③ 决策：**英文一条线先跑通**（MM-F2F 主），中文暂无现成大规模源，只能靠自采/Full-Duplex-Bench-zh，待英文跑通后再定。

---

## ⚠️ 执行发现与 framing 更新（changelog 2026-09-14）

> 本节记录 Phase 0 探针的**实测结果**及其对三大贡献的影响。
> 详细过程见 [`implementation-execution/phase0.md`](implementation-execution/phase0.md)；
> benchmark 转向的草案见 [`benchmark-proposal.md`](benchmark-proposal.md)（提案，非权威）。

**已实测**：
- **探针 A**（冻结 X2-Turn hidden 线性探针，Easy-Turn-en 318+299）：complete/incomplete
  **AUC ≈ 0.99**（去时长混淆后仍 ≥0.77，上界 0.99）→ **干净孤立语句上音频基线近天花板**。
- **Gate**（Full-Duplex-Bench-zh 句中停顿）：**X2-Turn 未被这些停顿骗到**——停顿前文是
  词法明显未完的悬空虚词（"比较…""是否…"），强音频 LM 直接判对，只在真语义结束点 turn_end。

**对三大贡献的诚实影响**：
- **C2（视觉扩展）**：在**干净条件的语义完整性**这条线上，卖点**显著减弱**
  （与 AV-Dialog clean +1.3% / MM-F2F +1.2% / VideoFDB 一致）。**不宜**作为主贡献。
- **★ 噪声鲁棒性 gate（2026-09-14 已跑）**：竞争说话人噪声下，音频完整性信号（剔除时长后）
  **单调崩塌 0.77→0.67→0.60→0.47（SNR 10/5/0）**，接近随机；仅时长基线稳在 0.863 不动
  → 下滑是噪声毁语义、非长度伪迹。**必要条件满足**：音频在声学退化下确实失效，视觉有潜在主场。
  （这正是 AV-Dialog 视觉 +13% 的干扰说话人场景。）
- **C1（AVSC-Corpus / benchmark）**：仍成立；靶心应从"思考停顿"收紧到
  **"声学退化下的完整性判断"**（音频失效、视觉可能补回的区域）。
- **C3（真机系统评测）**：不依赖视觉输赢，仍成立。

**新 framing（现在有实证支撑）**：不是"视觉帮语义判断"，而是
**"当音频被竞争说话人/噪声干扰、完整性信号塌掉时，视觉维持话轮完整性判断"**。

**待决策（未执行）**：
- **下半场实验**：取**带视频的 AV 数据**（MM-F2F 子集 / 自采），在噪声条件下验证
  **视觉能否补回**音频丢失的完整性信号（必要条件已过，这是充分性验证）。
- 若视觉在噪声下**也补不回** → 才退回「数据集 + negative result」下限。

> **状态**：necessary condition（音频会失效）✅ 已验；sufficiency（视觉能补回）⏳ 待 AV 数据。
> 在 sufficiency 未验前，不宣称 C2 成立，也不宣称项目失败——两头都还没到。

---

## 目录

- [第 0 部分 · 概览](#第-0-部分--概览)（章节号 `§0.x`）
- [Phase 0 · 前置探针与环境（W1–2）](#phase-0--前置探针与环境w1-2)（章节号 **`§P0.x`**）
- [Phase 1 · 数据集：AVSC-Corpus（W1–8）](#phase-1--数据集avsc-corpusw1-8)
- [Phase 2 · 模型结构改造（W3–8，与 Phase 1 并行）](#phase-2--模型结构改造w3-8与-phase-1-并行)
- [Phase 3 · 训练（W8–13）](#phase-3--训练w8-13)
- [Phase 4 · Benchmark 与评测（W13–19）](#phase-4--benchmark-与评测w13-19)
- [Phase 5 · 真机系统与论文（W19–24）](#phase-5--真机系统与论文w19-24)
- [附录 A · 里程碑 Gate 总表](#附录-a--里程碑-gate-总表)
- [附录 B · 风险单](#附录-b--风险单)
- [附录 C · 目录结构](#附录-c--目录结构)
- [附录 D · 待核实项状态](#附录-d--待核实项状态)

---

# 第 0 部分 · 概览

## 0.1 论文三大贡献

| # | 贡献 | 形态 | 可信度支点 |
|---|---|---|---|
| **C1** | **多人 per-face 语义完整性视听基准**：首个在多人同框视频上、对**每张脸**联合标注**双轴**（active-speaker「谁在说」× semantic completeness「语义闭合没」）的 chunk 级流式基准；概念上首次把「语义完整性 vs 行为结果 vs VAD」三轴区分落到 **video / 多脸** | HF Dataset + 标注工具 + Datasheet | **无任何现成语料覆盖此交集**（GRASS 纯音频/德/95min；AVCocktail 无完整性/无 ASD 任务）→ 自建必需；测试集全人工 + 双人 κ + 双轴（§1.3） |
| **C2** | **per-face 联合 AV 模型**：端到端联合融合，chunk 级流式输出**每张脸独立**的 {静默 / 说话-incomplete / 说话-complete}；视觉作归属消歧、音频判完整性（探针实证的模态分工） | 代码 + 权重 | 与 **MuVAP 的结构性差异**（不塌成 floor-holder，真 per-face、任意 N）；同构消融（mask 视觉即 −视觉臂） |
| **C3** | **实证发现 + 分析（punchline）**：现成 SOTA AV 话轮模型（VAP 家族 MuVAP / MM-VAP）在多人**语义完整性**上**近随机**，本方法恢复 X%；模态分工 + 条件切片（重叠/噪声/遮挡/侧脸）分析 | 指标表 + 消融（+ 真机 demo 作**补充材料**） | 复用 AVCocktail 公开 VAP checkpoint 作 baseline；探针 A/B 证据支撑「视觉→谁、音频→完整性」 |

> **★ 三条互锁**：C1 造出别人没有的地（数据 + 任务），C2 是唯一能在这地上跑的方法（真 per-face，非塌态），C3 用数字证明旧 SOTA 在这地上垮掉、我们不垮。
> **CVPR 语境下 C3 从「真机系统」改为「实证发现 + 分析」**（真机 demo 降为补充材料，非承重贡献）。

## 0.1.1 ★ C1 的精确措辞与论证（2026-09-04 核实后重写）

**背景**：核实 Kurata et al. Interspeech'23 全文后发现，它**确实是视听 + 静音点触发**，
但标注的是**行为结果**，不是语义完整性。C1 因此成立，但**措辞必须换轴**。

### 关键区分：语义完整性 vs 行为结果

Kurata §3 原文：

> "the speaker **either continues speaking after IPU or gives the turn to the interlocutor**.
> We assigned the label of *"continue utterance"* in the former case and
> that of *"end utterance"* in the latter case."

标签由「接下来实际发生了什么」决定 → **行为结果轴**，与「这句话闭合了没有」正交。

用双轴表看它的坍缩：

| | 说话人继续 | 对方接话 |
|---|---|---|
| **语义完整点停顿** | `complete` × hold | `complete` × change |
| **语义不完整点停顿** | `incomplete` × hold | `incomplete` × trail-off |

```
Kurata "continue utterance" = { complete×hold , incomplete×hold }   ← 两类被合并
Kurata "end utterance"      = { complete×change , incomplete×trail-off }  ← 同样被合并
```

**为什么这个合并致命**：

| 话语 | 语义完整性 | Kurata 标签 | 系统若接话 |
|---|---|---|---|
| "我想去巴黎…（800 ms）…明年夏天" | **complete** | continue | 合法（那是真 TRP），只是说话人选择继续 |
| "我想去…（800 ms）…巴黎" | **incomplete** | continue | **真打断错误** |

同一个标签，一个是可接受的时序选择，一个是硬错误。
**这正是「错误打断率」这个主指标要区分的东西**，而行为标注无法提供监督信号。

GRASS [2504.09980] 的统计说明这不是边缘情况：
`incomplete-hold` 占全部 turn-hold 的 **≈39%**（⚠️ 见 V12，投稿前需复核原文数字）。

**行为标注的第二个固有弱点**：标签继承了对话伙伴的个人习惯。
Kurata 是教师-学生在线面试，老师的耐心/礼貌会系统性偏置「对方是否接话」。

### Related Work 定位表（论文里直接用）

| 语料/工作 | 语义完整性标注 | 视频 | 多人 / per-face | 公开 | 规模 | 语言 |
|---|---|---|---|---|---|---|
| **GRASS** [2504.09980] | ✅ κ=0.875 (PCOMP) | ❌ | ❌ | ✅ | 95 分钟 | 奥地利德语 |
| **Kurata'23** [Interspeech'23] | ❌ **行为结果** | ✅ | ❌ | ❌ NEDO 内部 | 21.7 K 片段 | 日本人说英语（面试） |
| MM-F2F [2505.12654] | ❌ KEEP/TURN/BC | ✅ | ❌ 双人 | ✅（仅标注+链接） | 51 K turn | **英文** |
| AV-Dialog [2511.11124] | ❌ 行为 `<SOT>`/`<SOB>` | ✅ 唇动 | ❌ **单目标** | ⚠️ 仅项目页 | — | 英语 |
| MuVAP [2606.16731] | ❌ VAP 行为 | ✅ 人脸轨迹 | ⚠️ speaker-aware 但**塌成 2 态** | ✅ 附 31h 语料 | 31 小时 | — |
| AVCocktail [Interspeech'25 / 2609.17056] | ❌ VAP shift/hold | ✅ per-face 224² | ✅ 真多人(≤8) 但**实验只取双人** | ✅ | ~9.7 小时 | 待核 |
| **本项目基准（我们）** | ✅ **per-face 双轴** | ✅ | ✅ **多人 · 每脸独立** | ✅ | 待定 | 中 / 英（待定） |

> **交集全空**：语义完整性（只有 GRASS）× 真 per-face（无人做，MuVAP 都塌成 2 态）× 多人 AV —— 三者合一无先例。

### 定稿措辞

> **AVSC-Corpus** is, to our knowledge, the first **audio-visual** corpus annotated for
> **semantic completeness** (complete/incomplete) at within-turn pause points.
> Prior work presents an either-or: GRASS [2504.09980] provides such annotation but is
> audio-only and limited to 95 minutes; Kurata et al. [Interspeech'23] provides audio-visual
> data at IPU boundaries but labels the **behavioral outcome** (*continue* vs *end utterance*,
> determined by whether the interlocutor actually took the turn), which collapses pauses at
> semantically complete points together with pauses at incomplete points — the latter
> accounting for ≈39% of turn-holds in natural dialogue [2504.09980].

**关键词是「语义完整性 vs 行为结果」，不是「首个视听」。**
后者容易被一篇没检索到的论文推翻，前者有具体机制和数字支撑。

**⚠️ 仍然不要写**「首个视觉引导语义 VAD」（模型层面的首创声明）——
差异化落在**轻量 / 免重训 / 零前视 / 可部署 / 有配套数据**（见 §5.2.3）。

## 0.2 四条贯穿全程的纪律

| 纪律 | 出处 | 违反的后果 |
|---|---|---|
| **同构对照**：一切增益声明必须来自「同权重、同数据、唯一 mask 掉视觉」 | arch §3.13 | 增益不可信 |
| **WER 门禁** ΔWER ≤ +0.5% | arch §3.2 证据 A / V8 | 转写坏了话轮无从谈起 |
| **降级保证**：视觉失效 → 逐比特等于 −视觉臂 | arch §3.11 | VideoFDB 显示视觉常有害，审稿人必问 |
| **判别式输出**：logits 直出，不做 `generate()` | arch §3.6 / 2606.05713 | 生成式 >2× 误差且不可降级 |

## 0.3 时间线

```
W1  ─┬─ Phase 0 探针（决定 framing 与配置）
     └─ Phase 1 数据（伦理审批 + 许可核验，最长串行依赖，第一天启动）
W3  ─── Phase 2 模型改造（与 Phase 1 并行）
W8  ─── Phase 1/2 合流 → Phase 3 训练
W13 ─── Phase 4 Benchmark
W19 ─── Phase 5 真机 + 论文
W24 ─── 投稿（含 3 周缓冲）
```

**算力**：**双卡 H20（96 GB/卡，共 192 GB）**。
显存预算：冻结 4B bf16 前向 ≈8 GB ＋ LoRA r=32 及优化器 ≈4 GB ＋ 激活 ≈3 GB
= **≈16 GB / 96 GB → 单卡即可完成全部训练**。估计总训练 <100 GPU·h。
外部对照：2606.05713 在 7B omni 上做同类判别任务，RTX 5090 32 GB 即可完成。

**两卡用法**：不引入 DDP —— GPU 0 训练，GPU 1 并行跑探针/评测/数据流水线。
仅当 batch 需推到 32+ 时才 `torchrun --nproc_per_node 2`（HF Trainer 自动处理）。

**由算力确定的三项配置**（详见 `../docs/server-setup.md` §0）：

| # | 配置 | 说明 |
|---|---|---|
| 1 | **LoRA `r=32, α=64`** | 已锁定，不随数据量调档（前提见 §0.4） |
| 2 | **三个实验臂 A/B/C** | C 臂解冻 `vad_lm_head` 作上界对照，成本极低（§2.3） |
| 3 | 噪声增广**训练时即时合成** | 省磁盘、允许更多变体；测试集仍预计算固定种子（§1.5） |

**★ 真正的瓶颈是人力标注，不是算力**（见 §1.4）。

## 0.4 ★ 数据量与可训参数的硬耦合

**这是全计划最容易被忽略的硬约束，必须在 Phase 1 立项时就锁定。**

| 停顿事件数 | 可训参数 | LoRA 配置 | 能否走 SoulX 范式 |
|---|---|---|---|
| 2–3 K | 0.5–1 M | **不能加** 或 r=4 | ❌ 退化为纯冻结，主干读不懂视觉 |
| ~10 K | 3–5 M | r=8 | ⚠️ 勉强 |
| **≥ 30 K（本项目目标）** | 15–25 M | **r=32** | ✅ |

> **⚠️ 这个约束不是显存，是过拟合。**
> H20 的 96 GB 对 r=32 绰绰有余（实际占用 ≈16 GB），
> 所以**算力充足不能免除 ≥30 K 的数据要求**。
> 若最终数据量不足 30 K，**必须下调 r**，理由是过拟合而非显存。

**解法**（不增加人力）：

```
弱标注：全量铺开 → 训练集（目标 ≥30 K 事件）
人  审：只做测试集（500–800）+ κ 抽检（300）
```

依据：MM-F2F 有 **51 K turn 实例**、169 K utterance，静音段候选是**数万级**，
瓶颈从来在人审而非弱标注。

---

# Phase 0 · 前置探针与环境（W1–2）

**目的**：用最低成本回答两个决定论文 framing 的问题，并解锁待核实项。
**不需要**：训练循环、LoRA、视觉分支、标注数据（用小规模弱标注即可）。

> **⚠️ 本节章节号用 `P0.x` 前缀**，以区别于第 0 部分概览的 `§0.x`。
> 引用时请写全，例如「见 §P0.3」而非「见 §0.3」。

**执行顺序**（★ 注意依赖关系，不能只按编号）：

```
P0.1  V1 + V2          ← 无依赖，只需下 2 个小文件，5 分钟
  ↓
P0.2  环境 + 基线 + V3 + V7   ← V3/V7 必须模型能载入/能跑推理，故归在此
  ↓
P0.3  探针 A     P0.4  探针 B    ← 本 Phase 的真正目的
```

> **⚠️ 待核实项被拆成两处，这是刻意的**：
> V1/V2 只读配置文件，**不需要环境和完整权重**，第一天就能做完；
> **V3 需要 `print(model)`、V7 需要跑推理**，两者都依赖 P0.2 的环境就绪。
> 把它们放在一起会造成「纯读取」的错觉，实际会卡住。

## P0.1 无依赖的待核实项（**第一天即可完成**）

只需从 HF 下载两个小文件，**不需要环境、不需要完整权重**：

```bash
huggingface-cli download x-square-robot/X2-Turn-4B-0812 \
    --include "config.json" "tekken.json" \
    --local-dir /tmp/x2turn-meta
```

| ☐ | 项 | 方法 | 影响 |
|---|---|---|---|
| ☐ | **V1** `hidden_size`/`vocab_size`/层数 | 读 `config.json` | Projector 与 LoRA 参数量定档 |
| ☐ | **V2** id 41+ 是否空闲 | 查 `tekken.json` 词表 41–50 | **H-A 零参数方案成立与否；不成立则退 H-B（+6 K）** |

**V2 是这两项里更重要的**：它决定 ci 头能否复用 `vad_lm_head` 而零新增参数（方案 H-A，见 §2.4）。

## P0.2 环境、基线复现，以及需要模型在手的待核实项

| ☐ | 任务 | 交付物 | 验收 |
|---|---|---|---|
| ☐ | P0.2.1 环境配置 | **官方路径**：`conda env create -f X2-Turn/environments/environment-transformers.yml` + `pip install -e "./voxtral-realtime[transformers]"` + `pip install -e "./turn-demo"`（`install.sh` 是 fork 自制，仅作备选） | turn-demo 跑通 |
| ☐ | P0.2.2 加载 base | `load_mtp_checkpoint("x-square-robot/X2-Turn-4B-0812")` | bf16 载入成功，显存 ≈8 GB |
| ☐ | P0.2.3 复现帧级输出 | `scripts/run_backbone.py` | `infer_asr_turn` 输出 80 ms/帧；3.4 s 音频 ≈53 帧 |
| ☐ | P0.2.4 固化 golden | `tests/golden/backbone.json` | 固定样本的 turn_frames 序列，后续改动的回归基线 |
| ☐ | **P0.2.5 解锁 V3** | decoder layer 0 的模块路径 | `print(model)` 后确认 hook 挂点（**需模型已载入**） |
| ☐ | **P0.2.6 解锁 V7** | turn head 是否吃 delay tokens | 对比不同 `delay_ms` 下 turn 帧时序（**需能跑推理**）；决定视觉前视约束能否放松到 80 ms |

## P0.3 ★ 探针 A：X2-Turn 的 hidden 里已有多少完整性信息？

X2-Turn 训过 `turn_end`/`uncertain`，hidden 中很可能**已经**含有大量完整性信息。
这决定我们的 −视觉基线有多强，从而决定视觉能有多大空间。

```
冻结 X2-Turn → 抽 hidden_states → 线性探针 → 预测 complete/incomplete
```

| ☐ | 步骤 | 说明 |
|---|---|---|
| ☐ | 取 ~1 K 弱标注停顿事件（Phase 1.2 的早期产物即可） | 不需要人审 |
| ☐ | **拿到 hidden**（⚠️ 见下方，比预想麻烦） | 索引用 `prefix_length + frame_index − 1`（arch F3） |
| ☐ | 训 logistic regression / 单层 MLP | CPU 分钟级 |

> ### ⚠️ 取 hidden 需要额外一步（已核实代码）
>
> `inference.py` 的 `infer_asr_turn()` **只返回 `output.vad_logits`，拿不到 hidden**。
> `hidden_states` 仅存在于 `modeling.py` 内部（`:128,139,160,163`）。
>
> 三个可选做法：
>
> | 做法 | 说明 |
> |---|---|
> | **A（推荐）** | 对 `model.base_model.model(...)` 单独前向，取 `outputs.last_hidden_state` —— 这正是 `modeling.py:127-128` 的做法 |
> | B | 在 `AVVoxtralMTP` 包装层里让 forward 额外返回 hidden（Phase 2 反正要写这个包装） |
> | C | 挂 forward hook 抓取 | 
>
> **不要改 `X2-Turn/` 里的代码**（只读约定）。做法 A 只是调用，不算修改。

**判据与行动**：

| AUC | 含义 | 论文 framing |
|---|---|---|
| **> 0.85** | 音频基线很强，视觉空间小 | **必须走分层汇报**（思考停顿 × 噪声条件切片），不报全集 F1 |
| 0.70–0.85 | 有真实空间 | 按主线推进 |
| < 0.70 | hidden 里话轮信息不足 | 需考虑解冻 `vad_lm_head` 或加大 LoRA |

## P0.4 ★ 探针 B：视觉单独携带多少信息？

```
MediaPipe 24 维特征（同一批事件）→ 小 GRU → 预测 complete/incomplete
```

对照基准：**MM-F2F 的 Video-only 三分类是 0.559**（近随机），这是警示线。

| 仅视觉 AUC | 行动 |
|---|---|
| **> 0.62** | 视觉携带独立的 hold-intent 证据 → 核心卖点 |
| 0.55–0.62 | 弱信号 → 定位「噪声/远场鲁棒性」 |
| ≈ 0.50 | 视觉在 clean 下无用 → 只做噪声条件，或转 negative result |

**同时训第三个对照**：视觉 + 音频韵律 → 看是否有交互增益。

**Gate（W2）**：两个探针都出 AUC 数字（无论结果如何）；V1/V2/V3 有明确答案。

> **⚠️ 不要跳过探针直接进入 Phase 2/3。**
> 现有全部外部证据都指向「视觉增益来自噪声鲁棒性，不来自语义判断」
> （AV-Dialog clean +1.3%；MM-F2F +1.2%；VideoFDB 6/7 模型时序变差）。
> Kurata'23 的 +3.0 建立在 wav2vec2-base（AUC 0.887）弱基线上，
> **不能当作本项目的预期值**（见 [`architecture.md`](architecture.md) §B.1）。

---

# Phase 1 · 数据集：AVSC-Corpus（W1–8）

> **W1 第一天同时启动伦理审批与许可核验** —— 这是全计划唯一不可压缩的串行外部依赖。

## 1.1 数据来源与许可

| 语料 | 规模 | 视听 | 角色 | 风险 |
|---|---|---|---|---|
| **MM-F2F** | **210 h**，773 视频，**英文**，169 K utterance，51 K turn | ✅ | **主训练集（英文）** | ⚠️ **只发标注+YouTube链接+脚本，媒体须自下（生 YouTube，非去标识）**；再分发条款待核 |
| Seamless Interaction (InterAct) | 大规模双人视听（Meta, 2506.22554） | ✅ | 备选/补充 | 许可待核 |
| 自录真机子集 | 15 被试 × 20 轮 | ✅ | **仅测试集** | 需伦理审批 |
| SoulX-Duplug-Eval | — | ❌ 纯音频 | **跨域回归检查** | 无 |

| ☐ | 任务 | 验收 |
|---|---|---|
| ☐ | 1.1.1 **核对 MM-F2F 再分发条款** | 明确回答：衍生标注能否随论文发布？ |
| ☐ | 1.1.2 备选预案 | 若受限 → 发「标注索引 + 转换脚本」不发媒体；或换 InterAct |
| ☐ | 1.1.3 伦理审批（自录部分） | 提交申请，取得受理回执 |

**★ 语言策略（2026-09-16 决策）**：**英文一条线先跑通** —— 主训练/开发/探针 B 全用 MM-F2F（英文）。
**中文侧目前没有现成的大规模 AV 语料**（MM-F2F 是英文；Full-Duplex-Bench-zh 仅评测规模小样），
故中文只能靠**自采**（§1.1.3，需伦理审批）或 Full-Duplex-Bench-zh 补测试集。
是否投入中文自采，待英文线跑通、sufficiency 有结论后再定，不阻塞当前进度。

**⚠️ 更正（2026-09-16 核实仓库）**：论文虽描述了去标识化（>10,000 张合成人脸替换 + 声纹扰动，
去标识消融 0.836→0.823 影响很小），**但作者出于原视频上传者隐私，决定不发布处理后的媒体**
（README 原文："we have decided not to release the processed data directly"）。
实际发布的只有**标注 CSV + 原始 YouTube 链接 + 处理脚本**，视频须按 `video_id` 自行从 YouTube 下载。
→ **我们拿到的是生 YouTube 视频（非去标识），原「用它可绕开人脸合规风险」的结论不成立**；
人脸/版权合规须自行处理，且**我们同样不能再分发媒体**（只能仿照 MM-F2F 发标注+脚本）。

## 1.2 标注流水线

```
视听语料
   ↓
① ASR + 词级时间戳（Whisper-Large / Paraformer）
   ↓
② 静音段切分（≥200 ms），产出候选停顿事件（预计数万级）
   ↓
③ LLM 弱标注【轴 1】：给定静音点前文转写 → complete / incomplete + 置信度
   ↓
④ 高置信过滤 → 训练集（≥30 K 事件）
   ↓
⑤ 分层抽样 → 人工全标【轴 1】→ 测试集（500–800 事件）
   ↓
⑥ 双人独立标 300 条 → Cohen's κ
   ↓
⑦ ★【轴 2】自动抽取行为结果：VAD + 说话人分离 → hold / change（零人力）
   ↓
⑧ 对齐 MediaPipe 视觉特征 @12.5 Hz
   ↓
⑨ 输出四格交叉分布统计（§1.3 交叉表）→ 论文 Related Work 用
```

| ☐ | 任务 | 交付物 | 验收 |
|---|---|---|---|
| ☐ | 1.2.1 ASR + 时间戳对齐 | `avsvad/data/align.py` | 词级时间戳误差 <50 ms（抽检） |
| ☐ | 1.2.2 静音段切分 | `avsvad/data/segment.py` | 检出 ≥200 ms 静音段 + 起止时间戳 |
| ☐ | 1.2.3 LLM 弱标注（轴 1） | `avsvad/data/labeler.py` | 固定 prompt + 低温度 + 保留置信度字段 |
| ☐ | 1.2.4 视觉特征抽取 | `avsvad/visual/extractor.py` | 24 维 @12.5 Hz，含 `valid`/`confidence` |
| ☐ | **1.2.5 行为轴自动标注（轴 2）** | `avsvad/data/outcome.py` | 每事件产出 `hold`/`change`；抽检 100 条准确率 ≥95% |
| ☐ | **1.2.6 四格分布统计** | `results/phase1/crosstab.json` | 四格计数 + 占比；报告 `incomplete-hold` 占 hold 的比例 |
| ☐ | 1.2.7 打包 | `avsvad/data/dataset.py` | 见 §1.6 字段表 |

## 1.3 ★ 双轴标注体系 v2.0

> **v1.0 → v2.0 的变更**：只标语义完整性 → **两轴同时记录**。
> 动机见 §0.1.1：Kurata'23 只有行为轴，GRASS 有语义轴但无视频。
> 我们两轴都记，**副轴零人力成本**，换来三项论文收益（见本节末）。

### 轴 1 · 语义完整性（**预测目标**）

在静音点处回答：**说话人这句话语义/句法闭合了吗？**
判定**只看静音点之前**的内容，与之后发生什么无关。

| 标签 | 定义 | 映射 |
|---|---|---|
| `complete` | 语义/句法闭合，此处接话合法 | 1 |
| `incomplete` | 未闭合，说话人显然还要继续 | 0 |
| `ambiguous` | 标注员无法判断 | 剔除（不进训练/测试） |

### 轴 2 · 行为结果（**不是预测目标**，仅记录）

在静音点之后**实际发生了什么**。**由 VAD + 说话人分离自动得出，零人力成本。**

| 标签 | 定义 |
|---|---|
| `hold` | 同一说话人继续说 |
| `change` | 对方接话 |

**⚠️ 明确声明**：轴 2 **不作为训练监督**。它是 Kurata'23 的标签轴，
我们记录它只为做交叉分析与对照实验，避免把对话伙伴的个人习惯学进模型（§0.1.1）。

### 交叉表（命名对齐 GRASS [2504.09980] 便于对比）

| | `hold` 说话人继续 | `change` 对方接话 |
|---|---|---|
| `complete` | **①** complete-hold<br>*完整处的思考停顿* | **②** complete-change<br>*正常话轮交接* |
| `incomplete` | **③** incomplete-hold<br>*★ 不完整处的思考停顿* | **④** incomplete-change<br>*trail-off / 被打断* |

**Kurata'23 的标签映射**：`continue = ①∪③`，`end = ②∪④`。

### ★ 四个格子对系统的意义完全不同

| 格 | 系统判 `complete`（接话） | 系统判 `incomplete`（等待） |
|---|---|---|
| ① | **不算错误** —— 合法 TRP，说话人恰好选择继续 | 也可以，稍显迟疑 |
| ② | ✅ 正确 | ❌ 该接不接，延迟惩罚 |
| ③ | ❌ **真打断错误（主要惩罚项）** | ✅ 正确 |
| ④ | ✅ 正确（说话人已让出） | ❌ 冷场 |

**这张表直接修正了 §4.2 的两个主指标定义**：

```
错误打断率      = P(判 complete │ 格 ③)          ← 只在 ③ 上算，不含 ①
思考停顿存活率  = P(判 incomplete │ 格 ③)        ← ③ 是正样本来源
```

**格 ① 是关键**：若按 Kurata 的行为标签评测，格 ① 上判 complete 会被算成错误；
但那其实是**合法的**话轮选择。**混淆①③是行为标注固有的评测偏差，
双轴让我们能把它择出来** —— 这是方法论层面的贡献，写进论文。

### 三项收益（相对只标单轴）

| # | 收益 | 用在哪 |
|---|---|---|
| 1 | 用**自己的数据**给出四格人口分布，实证 Kurata 方案的坍缩程度，不必只引 GRASS 的 39% | §0.1.1 / Related Work |
| 2 | 可训一个**行为结果对照头**（预测 hold/change），证明两轴不可互替 | 新增消融 §4.5.7 |
| 3 | 评测可按格切片，主指标定义有据 | §4.2 / §4.3 |

**必须写进 guideline 的边界案例**（针对轴 1，2 页文档）：

| 场景 | 判定 |
|---|---|
| 自我修正（"我想去…不对，是"） | incomplete |
| 列举中途（"有苹果、香蕉、"） | incomplete |
| 口头禅/填充音后停顿（"就是说…嗯…"） | incomplete |
| 完整句 + 追加补充（"我要一杯咖啡。加奶。"） | 第一处 complete |
| 疑问句尾 | complete |
| 被对方打断 | 按打断前内容判 |

**★ 训练时监督铺满整段静音**（arch §3.10）：

```
frames:   ... speaking speaking │ sil sil sil sil sil │ speaking ...
ci_label: ...   -100     -100   │  1   1   1   1   1  │   -100
                                 └──── 同一事件标签 ────┘
```
单事件从 1 个监督点变 3–13 个，**数据量再放大近一个数量级**，且与流式推理形态一致。

## 1.4 人审协议（★ 论文可信度所在）

| ☐ | 任务 | 验收 |
|---|---|---|
| ☐ | 1.4.1 标注 UI | 播放静音点前后 ±3 s 音频 + 视频片段，4 键标签 |
| ☐ | 1.4.2 试点 50 条 | 发现 guideline 漏洞 → 修订 v1.1 |
| ☐ | 1.4.3 双人独立标 ≥300 条 | **报告 Cohen's κ** |
| ☐ | 1.4.4 分歧仲裁 | 第三方裁定，记录分歧类型分布 |
| ☐ | 1.4.5 测试集全人工 | 500–800 事件，分层覆盖 |
| ☐ | 1.4.6 弱标签质量抽检 | 随机 200 条 LLM 标签人工复核，报告一致率 |

**人力预估**：(300 × 2 + 800) × 1.5 min ≈ **35 人时**
（关键在于**只人审测试集**，训练集靠弱标注全量铺开 —— 见 §0.4）

## 1.5 测试集分层设计

测试集必须覆盖评测矩阵的所有格子，否则 Phase 4 无法做分层汇报：

| 维度 | 层 |
|---|---|
| **★ 四格（§1.3）** | ① complete-hold / ② complete-change / **③ incomplete-hold** / ④ incomplete-change |
| 停顿时长 | 200–400 / 400–800 / 800 ms+ |
| 音频条件 | clean / 噪声 / 远场 / 干扰说话人（后三者可合成增广）|
| 视觉条件 | 正脸 / 侧脸 / 遮挡 / 暗光 / 无人脸 |
| 语言 | 中文 / 英文 |

**★ 格 ③ 必须过采样**：它是「错误打断率」与「思考停顿存活率」的唯一正样本来源，
但在自然语料中占比不高。测试集中 **格 ③ 不少于 150 条**，否则主指标置信区间过宽。

**★ 格 ① 也必须够量**（≥100 条）：它是证明「行为标注会误判」的对照证据（§1.3）。

**合成增广配方**（参考 AV-Dialog）：20% clean / 40% MUSAN 背景噪声 /
40% 1–4 个干扰说话人，SNR −8~8 dB。

**★ 训练集增广在 collator 内即时合成**，不预计算落盘 ——
省磁盘，且每个 epoch 见到不同噪声实例。
**测试集必须预计算并固定随机种子**，否则评测不可复现。

## 1.6 数据字段

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `audio` | float32 [T] @16 kHz | 原始波形 |
| `lang` | str ∈{`zh`,`en`} | 语种 |
| `transcript` | str | ASR 转写 |
| `word_timestamps` | list | 词级时间戳 |
| `pause_events` | list of dict | `{start_ms, end_ms, label, source, confidence, outcome, cell}` |
| `visual_feats` | float32 [L, 24] | MediaPipe @12.5 Hz |
| `visual_valid` | bool [L] | 人脸可用性 |
| `ci_labels` | int [L] ∈{0,1,−100} | **轴 1**，铺满静音段（训练监督） |
| `outcome_labels` | int [L] ∈{0,1,−100} | **轴 2**，同样铺满；**仅用于对照实验 §4.5.7，主模型不监督** |
| `condition` | dict | 音频/视觉条件标签（测试集用） |

`pause_events` 内两个新字段：

| 字段 | 取值 | 来源 |
|---|---|---|
| `outcome` | `hold` / `change` | 1.2.5 自动 |
| `cell` | `1`..`4`（§1.3 交叉表格号） | `label` × `outcome` 派生 |

## 1.7 Datasheet

按 Datasheets for Datasets 格式：来源 / **双轴标注流程与两轴的明确分工** / κ /
**四格交叉分布** / 已知偏差 / 许可与 MM-F2F 衍生关系 / 去标识说明 / 中英语言与人群覆盖。

**必须在 Datasheet 里显式声明**：轴 2（行为结果）**不是**预测目标，
以免下游使用者误把它当标签训练（这正是 Kurata'23 的做法，§0.1.1）。

**Gate（W8）**：
- 训练集 ≥30 K 事件（弱标注，高置信过滤后）
- 测试集 500–800 事件全人工，分层覆盖达标，**格 ③ ≥150 条、格 ① ≥100 条**
- **κ ≥ 0.65**（低于则先改 guideline 再继续，不带病推进）
- **四格交叉分布已产出**（`results/phase1/crosstab.json`）
- 许可问题有决定性结论

---

# Phase 2 · 模型结构改造（W3–8，与 Phase 1 并行）

## 2.1 目标结构（详见 arch §3.3）

```
video → 冻结视觉前端 → VisualProjector(可训) → ×α(零初始化)
                                                    │
  ┌────────── X2-Turn-4B-0812（冻结）───────────────┼──────┐
audio → Voxtral 音频编码器 → audio_embeds ────────►(+)     │
  │                                                 ▼      │
  │                          Mistral decoder + LoRA r=32   │
  │                                 │ hidden_states        │
  │              ┌──────────────────┼──────────────────┐   │
  │              ▼                  ▼                  ▼   │
  │          lm_head           vad_lm_head        vad_lm_head
  │          (冻结)             id 35-40            id 41,42 ★
  └──────────────┼──────────────────┼──────────────────┼───┘
                ASR             6类话轮        complete/incomplete
```

## 2.2 新增模块清单

| 模块 | 文件 | 参数 | 说明 |
|---|---|---|---|
| `VisualFeatureExtractor` | `avsvad/visual/extractor.py` | 0（冻结） | MediaPipe → 24 维 @12.5 Hz，零前视 |
| `VisualEncoder` | `avsvad/visual/encoder.py` | ≈17 K | 因果 GRU(hidden=64)，窗口 1.6 s |
| `VisualProjector` | `avsvad/model/projector.py` | ≈0.8 M | `D_v → 256 → H`，**末层零初始化** |
| `AVVoxtralMTP` | `avsvad/model/av_mtp.py` | — | 包装 `VoxtralMTP`，注入视觉 + 新 ci 监督 |
| `CIHead` | `avsvad/model/ci_head.py` | **0**（H-A）/ 6 K（H-B）/ 0.8 M（H-C） | 见 §2.4 |

**⚠️ Projector 宽度不能照抄 SoulX**（`→2048→2048→1024` 会得到 10.5 M）。
用窄版 `D_v→256→H` ≈0.8 M。理由见 arch §3.9。

## 2.3 三个实验臂

| 臂 | 融合 | 主干 | ci 头 | 定位 |
|---|---|---|---|---|
| **A（下界）** | 晚融合：`concat(hidden, visual)` 只进 ci 头 | 全冻结 | H-C | 消融基线；ASR 路径**物理隔离**，WER 门禁天然通过 |
| **B（主力）** | 相加：`audio_embeds + α·visual_embeds` | **+ LoRA r=32** | H-A | SoulX 范式，出数字的那个 |
| **C（上界）** | 同 B | **+ LoRA r=32 且解冻 `vad_lm_head`** | H-A | 看头部解冻是否比纯 LoRA 更好；**必过 WER 门禁** |

**不设「早融合 + 全冻结」臂** —— 这是最差组合：视觉注入了但冻结主干不会用它
（依据见 arch §3.2 证据 E）。

**臂 C 的成本**：只多解冻一个 `H×6` 的小头（≈0.1 M），前向不变，对应 §3.1 的 S3 阶段。

**⚠️ 臂 C 的风险**：解冻 `vad_lm_head` 会直接改动原生 6 类话轮输出，因此：
1. WER 门禁必测
2. 原生话轮能力的漂移必须量化
3. **§4.7 的「同权重一次前向读不同 id 组」对照对 C 臂失效**，须独立加载原始权重跑基线

若 C 优于 B 但破坏了原生话轮，**仍以 B 为主线**，C 只作为上界报告。

## 2.4 ci 头三方案

| 方案 | 做法 | 新增参数 | 适用臂 |
|---|---|---|---|
| **H-A** | 复用 `vad_lm_head`，`complete/incomplete` 放 **id 41,42**，单独 softmax | **0** | B |
| H-B | `nn.Linear(H, 2)` | 6 K | B（V2 不成立时的退路） |
| H-C | `MLP(H + D_v → 256 → 2)` | 0.8 M | A |

**H-A 的实现只需改两处**：
1. 训练侧：`ci_loss = CE(vad_logits[..., [41,42]], ci_labels)`
2. 推理侧：`inference.py:86` 的 `TURN_CLASS_IDS` 旁边加一组 `CI_CLASS_IDS = (41, 42)`

## 2.5 视觉注入点（臂 B）

三种做法，风险递减，**推荐第 3 种**：

| 做法 | 风险 |
|---|---|
| 传 `inputs_embeds` | 最脆 —— 依赖库内部音频/文本嵌入合并逻辑 |
| `base_model.model` 挂 forward pre-hook | 中 |
| **decoder layer 0 挂 forward pre-hook** | **最稳** —— 不依赖任何内部构造细节 |

## 2.6 时间对齐（最易写错处）

```
audio token  │ a0 │ a1 │ a2 │ ...   80 ms / 12.5 Hz
visual_enc   │ v0 │ v1 │ v2 │ ...   重采样至 12.5 Hz
                   ↕ 逐位置相加
hidden 读取：prefix_length + i − 1     ← inference.py:165（next-token 偏移）
```

- 25 fps → 每 2 帧池化；30 fps → 每 2.4 帧插值
- **视觉帧率上限 12.5 Hz**（VideoFDB：2 FPS 达峰，过高反而更差）
- **80 ms 前视是免费的**：X2-Turn 音频通路本身有 480 ms 前视（`default_num_delay_tokens=6`）
  ⚠️ 取决于 V7 —— 若 turn head 严格因果，则前视约束重新生效

## 2.7 必须写的单测

| ☐ | 测试 | 断言 |
|---|---|---|
| ☐ | **恒等性** | `visual_valid` 全零 → 输出**逐比特**等于 −视觉臂 |
| ☐ | **零初始化** | 训练 step 0 时输出等于原始 X2-Turn |
| ☐ | **帧对齐脉冲响应** | 只在第 k 帧非零的视觉输入 → ci_logits 变化恰在预期位置 |
| ☐ | 形状 | 各张量形状符合 arch §3.12 |

> **帧对齐单测是必须的**：错一帧 = 全局 80 ms 系统性偏差，
> 且在指标上表现为「视觉略有帮助」，**极难察觉**。

**Gate（W8）**：三个单测全绿；`visual_valid=0` 恒等性逐比特通过。
不通过禁止进入 Phase 3。

---

# Phase 3 · 训练（W8–13）

## 3.1 三阶段（对齐 SoulX 的 freeze 策略）

```
S1  只训 VisualProjector + VisualEncoder（主干全冻结，LoRA 关闭）
      对应 SoulX 的 freeze_projector 开关（model.py:64-69）
      目的：验证视觉信号能否被读出
      ↓
S2  开 LoRA r=32，联合训 Projector + Encoder + LoRA        ← 臂 B 终点
      目的：主干学习解释视觉（arch §3.2 证据 E）
      ↓
S3  额外解冻 vad_lm_head，以 1/10 lr 微调                   ← 臂 C 终点
      （对应 §2.3 的臂 C，是固定对照臂而非可选项）
      必过 WER 门禁；且须单独量化原生 6 类话轮的漂移
```

**参考实现已在库内**：`SoulX-Duplug-training/`（上游 `training-code` 分支 `@928b065`）。

**写 Trainer 前的建议顺序**：

| 顺序 | 读什么 | 目的 |
|---|---|---|
| 1 | `SoulX-Duplug-training/example_data_fisher.jsonl` | **官方数据格式**，对照它定 §1.6 字段，避免格式返工 |
| 2 | `SoulX-Duplug-training/finetune.py` | 同范式同任务的训练循环 |
| 3 | `SoulX-Duplug-training/launch.sh` | lr / batch / 阶段划分的实际取值 |
| 4 | 才动手写 `avsvad/train/trainer.py` | — |

可选借用：`utils/ema/`（EMA 三种实现）、`utils/epoch_shuffle.py`（变长组批）、
`scripts/export_weights.py`（权重导出）。

## 3.2 损失

```
loss = 0   · asr_loss                            # 冻结骨干，ASR 不训
     + 0.1 · vad_loss                            # 保留原权重，防 6 类话轮能力漂移
     + λ_ci · CE(ci_logits[mask], ci_labels)     # λ_ci ≈ 1.0 起试
```

**代价非对称**：`ci_loss` 用 `pos_weight`，
把 incomplete 判成 complete（= 抢话打断）的代价 **>** 反向（= 多等一会儿）。
这是产品语义，必须进损失而非只调推理阈值。

**保留 `0.1·vad_loss` 的额外好处**：原生 6 类话轮不漂移，
于是同一模型自带对照（arch §3.13），Phase 4 零成本多一组实验。

## 3.3 视觉 dropout（防退化）

训练时以 **p ≈ 0.3** 随机整段置零 `visual_valid`（参考 MM-F2F 的 RMDT）。

作用：让模型学会没视觉也能干活 ——
把降级保证从「结构上成立」升级为「**统计上也成立**」。

## 3.4 超参与配置

| 项 | 值 | 备注 |
|---|---|---|
| GPU | **H20 单卡训练**（另一卡跑探针/评测） | DDP 仅在 batch ≥32 时启用 |
| 精度 | bf16 | 骨干冻结前向 ≈8 GB，总占用 ≈16 GB |
| 梯度检查点 | **可关** | 96 GB 显存充裕，关掉换速度；显存紧张时再开 |
| LoRA | **r=32, α=64（已锁定）** | 不再按数据量升降档（§0.4） |
| lr | sweep {1e-4, 3e-4, 1e-3} | 只作用于新参数；`vad_lm_head`（臂 C）用 1/10 |
| batch | **8 起，按 V5 结论上调** | 变长 padding；显存不是瓶颈 |
| 噪声增广 | **训练时即时合成** | 测试集预计算固定种子（§1.5） |
| 早停 | 按 val ci-AUC | 防过拟合 |

## 3.5 训练流程

| ☐ | 任务 | 验收 |
|---|---|---|
| ☐ | 3.5.1 Trainer + Dataset | HF `Trainer` 包装 `AVVoxtralMTP`，`compute_loss` 加 ci 项。**先读 SoulX `upstream/training-code`** |
| ☐ | 3.5.2 10% 数据冒烟 | ci-AUC 单调上升且不过拟合 |
| ☐ | 3.5.3 lr sweep | 选定配置 |
| ☐ | 3.5.4 S1 完整训练 | Projector 收敛 |
| ☐ | 3.5.5 S2 完整训练（臂 B） | 加 LoRA 后 ci-AUC 优于 S1 |
| ☐ | **3.5.6 S3 完整训练（臂 C）** | 解冻 `vad_lm_head`；**同时量化原生 6 类话轮漂移** |
| ☐ | 3.5.7 **WER 门禁** | 臂 B / 臂 C 均必测；ΔWER ≤ +0.5% 否则该臂作废 |
| ☐ | 3.5.8 **三臂各出一个 checkpoint** | 供 Phase 4 |

**Gate（W13）**：冒烟 ci-AUC 上升；WER 门禁通过（B/C 两臂）；**三臂 checkpoint 就绪**。

---

# Phase 4 · Benchmark 与评测（W13–19）

## 4.1 评测集清单

| 集 | 来源 | 用途 |
|---|---|---|
| **AVSC-test** | Phase 1，500–800 全人工，分层 | **主评测** |
| **SoulX-Duplug-Eval** | HF，纯音频 | **跨域回归检查**：确认我们没搞坏音频能力 |
| 真机集 | Phase 5.1 采集 | 系统级验证 |
| （可选）MM-F2F 原生 turn 标签 | 已有 | 与 KEEP/TURN/BC 体系交叉验证 |

## 4.2 指标定义

| 指标 | 定义 | 方向 |
|---|---|---|
| **ci-F1 / AUC** | 帧级 complete/incomplete（轴 1） | ↑ |
| **事件级判决准确率** | 每个停顿事件一票（多数投票或首帧） | ↑ |
| **错误打断率** | `P(判 complete │ 格 ③ incomplete-hold)` | ↓ **主** |
| **思考停顿存活率** | `P(判 incomplete │ 格 ③)` | ↑ **主** |
| **响应延迟中位** | 格 ② / ④ 上，用户说完到系统判 complete 的时长 | ↓ **主** |
| **★ 行为轴偏差量** | 若按轴 2 评测会在**格 ①** 上误记为错误的事件数/占比 | 报告用 |
| **TOR-Alignment** | VideoFDB 式时序对齐率（人类上界 90% / 1400 ms） | ↑ |
| **算法延迟** | 视觉前端前视 + 处理（平台无关） | ↓ |
| **ΔWER** | 相对原始 X2-Turn | ≤ +0.5% |
| **降级安全性** | 视觉失效条件下相对 −视觉臂的变化 | **必须 ≥ 0** |

**★ 指标必须按格切片**（依据 §1.3）：
主指标只在**格 ③** 上计算，不把**格 ①**（合法 TRP 上说话人恰好继续）计入错误。
「行为轴偏差量」这一行专门用来量化 Kurata'23 式评测会高估多少错误 ——
**这是我们相对既有视听工作的方法论差异，必须报告。**

## 4.3 主表：同构消融 × 条件矩阵

**臂**：`−视觉` / `臂A(晚融合冻结)` / `臂B(相加+LoRA r=32)` / **`臂C(+解冻 vad_lm_head)`** / `原始 X2-Turn 6 类话轮`

**⚠️ 臂 C 的对照不再"零成本"**：臂 A/B 的 `−视觉` 对照可由同权重 mask 视觉得到，
但**臂 C 解冻了 `vad_lm_head`，其原生 6 类话轮已被改动**，
因此「原始 X2-Turn」这一列对臂 C 必须用**独立加载的原始权重**跑，不能同权重复用。

**注意**：`−视觉` 臂**不需要重训** —— 同一权重把 `visual_valid` 全零即可（arch §3.13）。

| | clean | 风扇/电机噪声 | 远场 ≥2 m | 干扰说话人 |
|---|---|---|---|---|
| 正脸 | ✅ | ✅ | ✅ | ✅ |
| 侧脸 | ✅ | — | — | — |
| 遮挡 | ✅ | — | — | — |
| 暗光 | ✅ | — | — | — |
| 无人脸 | ✅ | ✅ | — | — |

**★ 必须单独报告 clean 条件的增益。** 若 ≈0（AV-Dialog 是 +1.3%，MM-F2F 是 +1.2%），
须诚实把工作定位为鲁棒性增强，否则会被 AV-Dialog Table 3 直接反驳。

## 4.4 ★ 固定阈值 Pareto（防「调大等待常数就行了」）

**这是回答审稿人「为什么不直接把 `silence_end_frames` 调大」的唯一有力方式。**

> ### ⚠️ 上游已经先调过一次阈值（2026-09-14 核实，务必读）
>
> 上游 `01af067`（2026-09-05，"Make v4 dialogue experience reproducible"）把规则控制器
> 默认值改成了 **`silence_end_frames = 10`（800 ms）**，注释原文：
> *"live mic: tolerate natural pauses"* —— **即上游用纯规则调参，对「容忍自然停顿」
> 这个问题做了一次部分缓解**，而这正是本项目声称要解决的问题。
>
> 两条硬性后果：
>
> | # | 后果 |
> |---|---|
> | 1 | **基线必须用 v4 默认值**（见下表）。拿旧值 `silence_end_frames=3` 当基线是**打稻草人**，审稿人一句「你跟未调优的基线比」即可废掉主表 |
> | 2 | **这其实是收益** —— 上游替我们把「纯调阈值能走多远」这个对照做实了。图上标出 v4 点，即可直接展示「阈值拉到 800 ms 后延迟代价多大、错误打断率只降到哪」，而我们的曲线在其外侧 |
>
> 本地 `X2-Turn/` 的 `config.py` / `controller.py` 已同步至 v4，
> 溯源与逐参数对照见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §1.2 A / §1.3。

**规则控制器基线的权威取值**（上游 v4，`turn/controller.py:44-56`）：

| 参数 | v4 值 | 等效时长 |
|---|---|---|
| `silence_end_frames` | **10** | 800 ms |
| `tail_min_frames` / `tail_max_frames` / `tail_stable_frames` | **1 / 1 / 1** | 80 ms |
| `short_tail_min_frames` / `short_tail_max_frames` | **3 / 5** | 240 / 400 ms |
| `acoustic_vad_max_hold_frames` | **3** | 240 ms veto 上限 |

```
横轴：响应延迟中位
纵轴：错误打断率
散点：扫 silence_end_frames ∈ {2..16} × tail_max_frames ∈ {1..15} 的固定阈值组合
      → 连成 Pareto 前沿
◆    ：★ 上游 v4 默认值（10, 1）—— 必须单独标注并在正文点名
星号：我们的自适应模型
```

> **扫描范围已从 `{2..10}×{3..15}` 放宽到 `{2..16}×{1..15}`** ——
> 否则 v4 的 `tail_max_frames=1` 落在原范围之外，图上标不出这个关键参照点。

**验收**：
- 自适应模型落在固定阈值 Pareto 前沿**外侧**
- **且相对 v4 默认值这一具体点有改进**（不只是相对前沿的某个极端点）

## 4.5 消融

| ☐ | 消融 | 目的 |
|---|---|---|
| ☐ | 4.5.1 视觉线索分组 | 注视 / 口型 / 头姿 / 眉毛 逐一关闭 → **对照 Kurata'23 的消融序**（去眼 0.684 < 去嘴 0.739 < 去头姿 0.758 < 全量 0.801，即眼>嘴>头姿） |
| ☐ | 4.5.2 视觉前端三选项 | MediaPipe / AV-HuBERT / omni vision tower → **容量阶梯曲线** |
| ☐ | 4.5.3 融合方式 | 臂 A vs 臂 B → 验证 LoRA 的必要性（arch 证据 E） |
| ☐ | 4.5.4 停顿时长分桶 | 200–400 / 400–800 / 800 ms+ → 视觉在长停顿上是否更有用 |
| ☐ | 4.5.5 视觉 dropout p | 0 / 0.3 / 0.5 → 对降级安全性的影响 |
| ☐ | 4.5.6 λ_ci / pos_weight | 代价非对称的敏感性 |
| ☐ | **4.5.7 ★ 双轴不可互替验证** | 同结构训两个头：**轴 1 头**（complete/incomplete）vs **轴 2 头**（hold/change，= Kurata 的目标）。交叉评测：轴 2 头在格 ③ 上的错误打断率应显著劣于轴 1 头 → **证明标注轴的选择本身有因果影响，而非措辞差异** |
| ☐ | **4.5.8 语言拆分** | 中文 / 英文分别报告 → 视觉线索是否跨语言迁移（Kurata 仅日本人说英语，MM-F2F 仅英文）；**⚠️ 中文侧目前无现成 AV 源，须自采/Full-Duplex-Bench-zh，英文跑通前此拆分可能只有英文** |

**4.5.7 是 C1 的实验支撑，优先级等同主表。**
没有它，「语义完整性 vs 行为结果」只是措辞主张；有了它就是可测的结论。
成本很低：数据里 `outcome_labels` 已经免费具备（§1.6），只需多训一个同结构头。

**4.5.2 的意义**：若 MediaPipe 就够 → **这本身是很好的结论**
（端侧可部署、可解释、零 GPU 视觉开销）。论文里写成 finding 而非退让。

## 4.6 降级安全测试

| ☐ | 测试 | 断言 |
|---|---|---|
| ☐ | 确定性 | `visual_valid` 全零 → 输出**等于** −视觉臂（单测已保证，此处复验） |
| ☐ | 真实遮挡片段 | 指标 **不劣于** −视觉臂 |
| ☐ | 暗光 / 侧脸 / 无人脸 | 同上 |
| ☐ | 视觉流突然中断 | 无异常抖动 |

## 4.7 外部对照

| 对照 | 说明 |
|---|---|
| 原始 X2-Turn 6 类话轮控制器 | 同权重、同一次前向读不同 id 组 |
| SoulX-Duplug-0.6B | 在其自带评测集上跑通，**引文献数字，不跨数据集强行比较**（标签体系不同） |
| 固定阈值网格 | §4.4 |

**Gate（W19）**：噪声/远场条件下显著优于 −视觉臂；Pareto 图成立；降级测试全过。

---

# Phase 5 · 真机系统与论文（W19–24）

## 5.1 真机集成

| ☐ | 任务 | 说明 |
|---|---|---|
| ☐ | 5.1.1 平台搭建 | 摄像头 + 麦克风；**必须真机采集**，否则 IROS 质疑 "why robotics" |
| ☐ | 5.1.2 推理链路 | 优先 transformers 路径（vLLM patched 链路深、时间风险大）|
| ☐ | 5.1.3 视觉前端实时进程 | MediaPipe CPU <15 ms/帧，帧对齐缓冲 |
| ☐ | 5.1.4 **ci 判决接入规则控制器** | 见下 |
| ☐ | 5.1.5 采集诱发脚本 | 开放式提问 / 需回忆 / 列举任务 → **刻意诱发思考停顿** |
| ☐ | 5.1.6 端到端延迟测量 | 视觉前端 + 模型 + 控制器 |

**ci 判决 → 控制器的联动**（arch §3.13）：

```
ci_logits → p(continue) → 调制 FrameTurnConfig：
                            silence_end_frames (v4 默认 10 = 800 ms)
                            tail_max_frames    (v4 默认 1  =  80 ms)
                          factor = 1 + g·conf·(2p−1)，clamp
```

> ⚠️ **基准值已更新为上游 v4**（2026-09-14）。旧版本此处写的是 240 ms / 400 ms，
> 那是上游 `01af067` 之前的默认值，**已废弃**。
> 用旧值设计调制范围会导致：模型在 v4 基线上「往长了调」的空间被严重高估
> （800 ms 已经很长），而「往短了调」才是主要收益方向 —— **这会改变 `g` 的取值与 clamp 边界**。
> 逐参数对照见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §1.3。

**模型负责判决，控制器负责时序。** 照抄仓库现有的外部信号注入模式：
`AcousticVoiceGate`（`server.py:102` 每帧求值 → `server.py:115-120` 注入 `on_frame`），
接入点全仓库仅一处。**其 veto 上限逻辑（`controller.py:110-118`）正是纪律 3
「外部信号只能延缓、不能无限阻断」的现成写法，直接复用。**
完整锚点见 [`../docs/code-anchors.md`](../docs/code-anchors.md) §1。

## 5.2 论文

**Title（草案）**：*Who Is Speaking and Are They Done? Per-Face Audio-Visual Semantic Endpointing in Multi-Party Conversation*
（旧 IROS 标题 *Visually-Grounded Semantic Endpointing for Robot Speech Interaction* 已弃）

| ☐ | 任务 |
|---|---|
| ☐ | 5.2.1 三点贡献（C1 多人 per-face 双轴基准 / C2 per-face 联合 AV 模型 / **C3 实证发现+分析**；真机 demo 作补充材料） |
| ☐ | 5.2.2 Related Work：补 X2-Turn(2608.10878)、SoulX(2603.14877) 全文表格 + **§0.1.1 的语料定位表** |
| ☐ | 5.2.3 差异化表述（见下） |
| ☐ | 5.2.4 Limitation 如实写 clean 增益、**仅中英两语**、单一 base、**轴 2 为自动标注（未人工复核全量）** |
| ☐ | 5.2.5 Demo 视频：思考停顿不被打断 vs 基线抢话 |
| ☐ | 5.2.6 发布：HF(model+dataset) + GitHub(代码+标注工具) |

**5.2.3 必须处理的外部声称（2026-09-18 按新方向重写，深挖见 phase2.md §2.1/§2.3）**：

| 竞品 | 声称 | 我们的表述（差异化） |
|---|---|---|
| **MuVAP** [2606.16731]（最像） | AV 单麦+单摄 · 多人 · HRI 话轮 | 只预测**行为 Shift-Hold / next-speaker**（VAP），**不判语义完整性**；把 N 人**塌成"当前 vs 下一 floor-holder"2 态**，非 per-face。我们：正交的**语义完整性轴** + **真 per-face 独立输出**。**卖点绝不写成"AV 多人话轮"** |
| **AV-Dialog** [2511.11124]（最危险） | "**semantically grounded** turn-boundary detection" + AV + 流式 | 全文核实：输出为**行为事件 token `<SOT>`/`<SOB>`**（PairwiseTurnGPT 行为分类），"grounded"仅指**以转写语义为条件**，**从不判 complete/incomplete**；**单目标说话人**（干扰当噪声抑制）、唇动、**dyadic**。我们：**per-face 多人 × 语义完整性判决**。⚠️ **必须点名拆解"semantically grounded"**以防审稿误判撞车 |
| **MM-VAP** [2607.07294] | AV 话轮预测 | 硬编码 2 人（256 态未来语音活动）；"semantic consistency loss"仅为 VAP 状态**正则项**，非完整性目标 |
| **AVCocktail** [2609.17056 / Interspeech'25] | AV 多人 cocktail-party 话轮测试床 | 沿用 **VAP shift/hold**、**无 ASD 任务**（吃预切脸）、**无完整性标注**、实验只取双人 → **可复用其数据/VAP checkpoint 作 baseline 与 C3 对照**，我们补齐 per-face 双轴 |
| **★ Kurata'23** [Interspeech'23] | 视听 + "end-of-utterance prediction" | 标注**行为结果**（continue/end，由对方是否接话决定），坍缩语义完整/不完整（§0.1.1）；语料未公开、IPU 触发非帧同步、非 LLM。**引用其视觉线索消融序（眼>嘴>头姿）作设计依据** |
| Qwen3.5-Omni [2604.15804] / MiniCPM-o [2604.27393] | "native turn-taking" / "减少 VAD 依赖" | 权重仅 API 或评测不含真话轮/端点任务，无法作可扩展基座；均非多人 per-face 语义完整性 |

**首创声明限定**：只说 **「首个多人 per-face 语义完整性（active-speaker × completeness）视听基准 / 模型」**（§0.1.1）。
**不要**写"首个 AV 流式话轮模型"（AV-Dialog 已占"首个 AV 流式全双工对话系统"），也不要写"首个视觉引导语义 VAD"。
差异化护城河 = **语义完整性轴 + 真 per-face + 多人视频**，辅以轻量 / 可解释。

---

# 附录 A · 里程碑 Gate 总表

| 周 | 里程碑 | Gate（不过不得推进） |
|---|---|---|
| W1 | 启动 | **伦理审批已提交**；MM-F2F 许可核验启动 |
| **W2** | **Phase 0 探针** | 两个 AUC 数字出炉（无论结果）；V1/V2/V3 有答案 |
| W4 | 标注 guideline | v1.0 定稿，试点 50 条通过 |
| **W8** | **数据 + 模型双 Gate** | ① 训练集 ≥30 K，测试集全人工，**κ≥0.65**，**四格分布已产出（格③≥150、格①≥100）** ② **恒等性单测逐比特通过** |
| W10 | 冒烟训练 | ci-AUC 单调上升 |
| **W13** | **训练完成** | **WER 门禁通过（B/C）**；三臂 checkpoint |
| **W19** | **评测完成** | 主表显著；**Pareto 图成立**；**降级测试 ≥0**；**§4.5.7 双轴对照出结论** |
| W22 | 论文成稿 | **V12 已复核**（GRASS 承重数字） |
| W24 | 投稿 | 含 3 周缓冲 |

---

# 附录 B · 风险单

| 风险 | 概率 | 对策 |
|---|---|---|
| **clean 条件增益 ≈ 0** | **高** | 探针 B 提前预判；定位为噪声/远场鲁棒（机器人真实工况）；如实报 clean |
| **数据量不达 30 K** | 中 | 弱标注全量铺开 + 监督铺满静音（双重放大）；不足则降 LoRA 档位 |
| MM-F2F 再分发受限 | 中 | 发标注+脚本不发媒体；或换 InterAct |
| 弱标签毒化训练 | 中 | 高置信过滤 + 测试集全人工 + 抽检报一致率 |
| 视觉引入退化 | 中高 | 零初始化 + 视觉 dropout + 恒等性单测 + 降级测试 |
| **帧对齐错 1 帧** | 中 | **脉冲响应单测**（否则表现为「视觉略有帮助」，极难察觉） |
| 早融合伤 ASR | 中 | WER 门禁；失败回退臂 A |
| 探针 A AUC 过高（>0.9） | 中 | 音频基线太强 → 改走分层汇报 + 系统级指标（错误打断率/延迟） |
| 伦理审批延误 | 中高 | 主训练数据用已去标识的 MM-F2F 规避；自录只做测试集 |
| 无真机可用 | 中 | 最低配置：摄像头+麦克风桌面平台，但必须真实采集 |
| 「为啥不直接调阈值」质疑 | **高** | §4.4 Pareto 图正面回应 |
| 被质疑与 AV-Dialog / Qwen3.5-Omni 重复 | 中 | §5.2.3 四条表述 |
| **C1 首创声明被某篇未检索到的工作推翻** | 中 | 措辞已限定在「语义完整性 × 视听」交集（§0.1.1），且用 "to our knowledge"；**即使被推翻，C2/C3 与双轴标注体系仍独立成立** |
| **GRASS 承重数字转述有误** | 中 | **V12：投稿前核对原文**；若 39% 不实，改用 V13 自有数据的四格分布论证 |
| 轴 2 自动标注不准（VAD/分离出错） | 中 | 抽检 100 条 ≥95%；轴 2 不进主监督，出错不污染主模型 |
| 上游 X2-Turn 更新 | 低 | 只读 + 包装，golden 测试会显式报错 |

---

# 附录 C · 目录结构

```
AV-SemanticVAD/
├── plan/
│   ├── architecture.md          # 架构设计 v4
│   └── implementation-plan.md   # 本文件
├── docs/
│   └── server-setup.md          # ★ GPU 服务器部署清单（双卡 H20）
├── research/                    # 调研报告（已完成）+ Kurata'23 PDF
├── avsvad/
│   ├── upstream.py              # 免 websockets 加载上游 modeling/controller
│   ├── visual/
│   │   ├── extractor.py         # MediaPipe → 24 维 @12.5Hz
│   │   ├── features.py          # 特征定义与归一化
│   │   └── encoder.py           # 因果 GRU(64)
│   ├── model/
│   │   ├── projector.py         # VisualProjector（窄版，末层零初始化）
│   │   ├── ci_head.py           # H-A / H-B / H-C
│   │   ├── fusion.py            # 晚融合(臂A) / 相加+hook(臂B)
│   │   └── av_mtp.py            # AVVoxtralMTP 包装
│   ├── data/
│   │   ├── align.py             # ASR + 词级时间戳
│   │   ├── segment.py           # 静音段切分
│   │   ├── labeler.py           # 轴1 LLM 弱标注（语义完整性）
│   │   ├── outcome.py           # ★ 轴2 行为结果自动标注（VAD+分离）
│   │   ├── crosstab.py          # ★ 四格交叉分布统计
│   │   ├── augment.py           # 噪声/远场/干扰说话人合成
│   │   └── dataset.py           # Dataset + collator
│   ├── train/
│   │   ├── trainer.py           # HF Trainer 包装，ci_loss 组合
│   │   └── cli.py
│   └── eval/
│       ├── metrics.py           # ci-F1 / 错误打断率 / 延迟 / TOR
│       ├── pareto.py            # §4.4 固定阈值扫描
│       └── runner.py            # 分层评测
├── annotate/                    # 人审 UI
├── scripts/                     # 各阶段 CLI 入口
├── configs/
├── tests/
│   ├── golden/                  # 基线指纹
│   ├── test_identity.py         # ★ 恒等性
│   └── test_alignment.py        # ★ 帧对齐脉冲响应
├── data/                        # gitignore
└── results/                     # 各 phase 产物 JSON
```

**原则**：X2-Turn 目录**只读**，不 fork 源码树。

**在工作区中的位置**（父仓库 `SemanticVAD/` 已建 git，两个基座为 submodule）：

```
SemanticVAD/                  ← git root
├── THIRD_PARTY.md            ← 第三方溯源清单（SHA / 许可 / 只读约定）
├── AV-SemanticVAD/           ← 本项目（上面的树）
├── X2-Turn/                  ← 只读，base 权重来源     上游 @8992c7c
├── SoulX-Duplug/             ← 只读，范式参考（推理服务）      main @45bd237
└── SoulX-Duplug-training/    ← 只读，★ 训练代码参考   training-code @928b065
```

**三份第三方代码已直接纳入仓库，不用 submodule** ——
服务器上 `git clone <地址>` 一条命令拉齐，后续见 [`../docs/server-setup.md`](../docs/server-setup.md)。

> ⚠️ `SoulX-Duplug/` 与 `SoulX-Duplug-training/` **是互补的两棵树，不是包含关系**：
> 前者含 `service/model.py`（complete/incomplete 判决实现），
> 后者含 `finetune.py` 与 `example_data_fisher.jsonl`（训练循环与数据格式）。

---

# 附录 D · 待核实项状态

| # | 项 | 状态 | 归属 |
|---|---|---|---|
| V1 | `hidden_size`/`vocab_size`/层数 | ⬜ | Phase P0.1 |
| V2 | id 41+ 是否空闲 | ⬜ | Phase P0.1 |
| V3 | decoder layer 0 模块路径 | ⬜ | Phase P0.2.5（需模型已载入）|
| V4 | `−1` 偏移语义 | ⬜ | Phase 2.7 单测 |
| V5 | batch>1 与变长 padding | ⬜ | Phase 3.4 |
| V6 | 两趟推理能否合并 | ⬜ | Phase 5.1 |
| V7 | turn head 是否吃 delay tokens | ⬜ | Phase P0.2.6（需能跑推理）|
| V8 | LoRA 是否伤 ASR | ⬜ | Phase 3.5.6 WER 门禁 |
| **V9** | **Qwen3.5-Omni 粒度与规模** | ✅ **已解决** | arch §2.3.1：权重仅 API / 数千亿参数 / 视觉非流式 / 视频 1 FPS |
| V10 | Qwen2.5-Omni-3B vision tower 许可与可加载性 | ⬜ | Phase 4.5.2（⚠️ 只能用 2.5 版，3.5 权重不开放） |
| V11 | 该 tower 每帧 patch token 数与池化 | ⬜ | Phase 4.5.2 |
| **D1** | **是否已存在「视听 + 语义完整性标注」的语料/模型** | ✅ **已解决（2026-09-04）** | 核实 Kurata'23 全文：视听 ✅ 但标注为**行为结果**，非语义完整性 → **C1 成立**，措辞按 §0.1.1 换到「语义完整性 vs 行为结果」轴。法语 ALLIES 已排除（仅在说话人切换点标 TRP，无 hold 样本，训不出「该不该等」；且我们做中/英） |
| **V12** | **GRASS 的两个统计量原文复核** | ⬜ **投稿前必做** | `incomplete-hold` 占 turn-hold ≈39%、turn-change 中句法不完整 ≈17%、κ=0.875、95 分钟 —— 这四个数字在 §0.1.1 定稿措辞里承重，**必须核对 2504.09980 原文**，不得凭二手转述 |
| **V13** | 我们自己数据的四格分布 | ⬜ | Phase 1.2.6；若格 ③ 占 hold 的比例远低于 39%，需在论文中解释场景差异，并**下调 C1 论证的强度** |

---

## 立即执行的三件事

1. **W1 D1**：提交伦理审批 + 核对 MM-F2F 再分发条款（最长串行依赖）
2. **W1**：配环境、载 X2-Turn 权重、查 V1/V2/V3
3. **W2**：跑探针 A 与探针 B —— **两个 AUC 数字决定整篇论文的 framing**
