# 实施计划单

> **v3.2 — 2026-09-04**，对齐 [`architecture.md`](architecture.md) v4。
> 服务器部署见 [`../docs/server-setup.md`](../docs/server-setup.md)。
>
> **方案**：以 **X2-Turn-4B-0812 权重**为 base 全部冻结，套 **SoulX-Duplug 范式**
> （冻结视觉前端 + 可训 Projector + LoRA）侧接一路视觉，
> 在 `vad_lm_head` 空闲 token id 上**判别式**输出逐帧 complete/incomplete。
>
> **目标会议**：IROS ｜ **周期**：约 6 个月 ｜ **算力**：**双卡 H20（96 GB/卡）** ｜ **语言**：中 + 英
>
> v2 → v3 的三处实质变更：
> 1. base 明确为 **X2-Turn 权重**（v2 误写为 Voxtral）
> 2. 数据量目标 **2–3 K → 万级**（否则撑不起 LoRA，见 §1.0）
> 3. 删除「早融合 + 全冻结」臂；新增**前置探针**作为 Phase 0
>
> **v3 → v3.1（核实 Kurata'23 Interspeech 全文后）**：
> 1. **C1 换轴**：「首个视听 complete/incomplete 语料」→
>    **「首个带*语义完整性*标注的视听语料」**，论证从「有没有视听」改为
>    **「标的是语义完整性还是行为结果」**（新增 §0.1.1）
> 2. **标签体系 v1.0 → 双轴 v2.0**（§1.3）：轴 1 语义完整性（预测目标）
>    ＋ 轴 2 行为结果（零人力自动，仅对照）→ 四格交叉表
> 3. **主指标按格切片**（§4.2）：错误打断率只在格 ③ 上算，
>    新增「行为轴偏差量」量化 Kurata 式评测的高估
> 4. 新增消融 **§4.5.7 双轴不可互替验证**（C1 的实验支撑）与 **§4.5.8 语言拆分**
> 5. 新增 **V12/V13**（GRASS 承重数字复核、自有四格分布）；**D1 已关闭**
>
> **v3.1 → v3.2（算力确定为双卡 H20）**：
> 1. **LoRA 锁定 `r=32, α=64`**，删除按数据量升降档（§0.4）
>    —— ⚠️ 但 ≥30 K 数据要求**不变**，因为该约束来自过拟合而非显存
> 2. **新增臂 C**（相加 + LoRA + 解冻 `vad_lm_head`），原 S3「可选阶段」升级为固定对照臂（§2.3 / §3.1）
> 3. 噪声增广改为**训练时即时合成**，测试集仍预计算固定种子（§1.5）
> 4. 两卡策略：**不用 DDP**，GPU 0 训练 / GPU 1 跑探针评测（§0.3）
> 5. 新增 [`../docs/server-setup.md`](../docs/server-setup.md)：环境、权重下载、Phase 0 验收、代码锚点、已知坑

---

## 目录

- [第 0 部分 · 概览](#第-0-部分--概览)
- [Phase 0 · 前置探针与环境（W1–2）](#phase-0--前置探针与环境w1-2)
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
| **C1** | **AVSC-Corpus**：首个带**语义完整性**（complete/incomplete）标注的**视听**语料，中/英双语 | HF Dataset + 标注工具 + Datasheet | 弱标注全量铺开 + 测试集全人工 + 双人 κ + **双轴标注**（§1.3） |
| **C2** | **AV-X2-Turn**：冻结骨干上的轻量视觉扩展（Projector + LoRA + 征用空闲 id 的判别头） | 代码 + 权重 | 同构消融（同权重 mask 掉视觉即 −视觉臂） |
| **C3** | **真机系统评测**：错误打断率 / 响应延迟 / 固定阈值 Pareto / 降级安全 | demo 视频 + 指标表 | 与原始规则控制器同台，人类参考上界 |

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

| 语料 | 语义完整性标注 | 视频 | 公开 | 规模 | 语言 |
|---|---|---|---|---|---|
| **GRASS** [2504.09980] | ✅ κ=0.875 | ❌ | ✅ | 95 分钟 | 奥地利德语 |
| **Kurata'23** [Interspeech'23] | ❌ **行为结果** | ✅ | ❌ NEDO 内部 | 21.7 K 片段 | 日本人说英语（面试） |
| MM-F2F [2505.12654] | ❌ KEEP/TURN/BC | ✅ | ✅ | 51 K turn | 中文 |
| AV-Dialog [2511.11124] | ❌ `<SOT>`/`<SOB>` | ✅ | ⚠️ 仅项目页 | — | 英语 |
| **AVSC-Corpus（我们）** | ✅ **+ 双轴** | ✅ | ✅ | ≥30 K 事件 | **中 + 英** |

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

**★ 算力宽裕带来的三项决策变更**（详见 `docs/server-setup.md` §0）：

| # | 决策 | 相对原计划 |
|---|---|---|
| 1 | **LoRA 直接锁 `r=32, α=64`** | 删除「按数据量升降档」的条件判断（§0.4） |
| 2 | **新增臂 C**：相加 + LoRA + **解冻 `vad_lm_head`** | 上界对照，成本极低（§2.3） |
| 3 | 噪声增广改为**训练时即时合成** | 省磁盘，允许更多变体（§1.5） |

**真正的瓶颈是人力标注，不是算力**（见 §1.4）。这一点没有因为算力升级而改变。

## 0.4 ★ 数据量与可训参数的硬耦合（v3 的核心修正）

这是 v2 最大的内在矛盾，必须在 Phase 1 立项时就解决：

| 停顿事件数 | 可训参数 | LoRA 配置 | 能否走 SoulX 范式 |
|---|---|---|---|
| 2–3 K（v2 原定） | 0.5–1 M | **不能加** 或 r=4 | ❌ 退化为纯冻结，主干读不懂视觉 |
| ~10 K | 3–5 M | r=8 | ⚠️ 勉强 |
| **≥ 30 K（v3 目标）** | 15–25 M | **r=32（已锁定）** | ✅ |

> **⚠️ 注意约束的性质变了。**
> 这张表的约束**从来不是显存**（H20 96 GB 绰绰有余），而是**数据量与参数量的过拟合关系**。
> 所以算力升级**不能**免除 ≥30 K 的数据要求 —— 它只是让我们不必再纠结 r=16 还是 r=32，
> **直接取 r=32**。若最终数据量不足 30 K，仍须下调 r，理由是过拟合而非显存。

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

## 0.1 环境与基线复现

| ☐ | 任务 | 交付物 | 验收 |
|---|---|---|---|
| ☐ | 0.1.1 环境配置 | conda env（按 X2-Turn `environments/`） | turn-demo 跑通 |
| ☐ | 0.1.2 加载 base | `load_mtp_checkpoint("x-square-robot/X2-Turn-4B-0812")` | bf16 载入成功，显存 ≈8 GB |
| ☐ | 0.1.3 复现帧级输出 | `scripts/run_backbone.py` | `infer_asr_turn` 输出 80 ms/帧；3.4 s 音频 ≈53 帧 |
| ☐ | 0.1.4 固化 golden | `tests/golden/backbone.json` | 固定样本的 turn_frames 序列，后续改动的回归基线 |

## 0.2 ★ 探针 A：X2-Turn 的 hidden 里已有多少完整性信息？

X2-Turn 训过 `turn_end`/`uncertain`，hidden 中很可能**已经**含有大量完整性信息。
这决定我们的 −视觉基线有多强，从而决定视觉能有多大空间。

```
冻结 X2-Turn → 抽 hidden_states → 线性探针 → 预测 complete/incomplete
```

| ☐ | 步骤 | 说明 |
|---|---|---|
| ☐ | 取 ~1 K 弱标注停顿事件（Phase 1.2 的早期产物即可） | 不需要人审 |
| ☐ | 抽 `hidden_states[prefix_length + i − 1]` | 注意 −1 偏移（arch F3） |
| ☐ | 训 logistic regression / 单层 MLP | CPU 分钟级 |

**判据与行动**：

| AUC | 含义 | 论文 framing |
|---|---|---|
| **> 0.85** | 音频基线很强，视觉空间小 | **必须走分层汇报**（思考停顿 × 噪声条件切片），不报全集 F1 |
| 0.70–0.85 | 有真实空间 | 按主线推进 |
| < 0.70 | hidden 里话轮信息不足 | 需考虑解冻 `vad_lm_head` 或加大 LoRA |

## 0.3 ★ 探针 B：视觉单独携带多少信息？

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

## 0.4 解锁待核实项

| ☐ | 项 | 方法 | 影响 |
|---|---|---|---|
| ☐ | **V1** `hidden_size`/`vocab_size`/层数 | 读 `config.json` | Projector 与 LoRA 参数量定档 |
| ☐ | **V2** id 41+ 是否空闲 | 查 Tekken 词表 41–50 | H-A 零参数方案成立与否 |
| ☐ | **V3** decoder layer 0 模块路径 | `print(model)` | 相加注入的 hook 挂点 |
| ☐ | **V7** turn head 是否吃 delay tokens | 对比不同 `delay_ms` 下 turn 帧时序 | 视觉前视约束能否放松到 80 ms |

**Gate（W2）**：两个探针都出 AUC 数字（无论结果如何）；V1/V2/V3 有明确答案。

---

# Phase 1 · 数据集：AVSC-Corpus（W1–8）

> **W1 第一天同时启动伦理审批与许可核验** —— 这是全计划唯一不可压缩的串行外部依赖。

## 1.1 数据来源与许可

| 语料 | 规模 | 视听 | 角色 | 风险 |
|---|---|---|---|---|
| **MM-F2F** | **210 h**，773 视频，955 人，169 K utterance，51 K turn，**已去标识化** | ✅ | **主训练集** | ⚠️ 再分发条款待核 |
| Seamless Interaction (InterAct) | 大规模双人视听（Meta, 2506.22554） | ✅ | 备选/补充 | 许可待核 |
| 自录真机子集 | 15 被试 × 20 轮 | ✅ | **仅测试集** | 需伦理审批 |
| SoulX-Duplug-Eval | — | ❌ 纯音频 | **跨域回归检查** | 无 |

| ☐ | 任务 | 验收 |
|---|---|---|
| ☐ | 1.1.1 **核对 MM-F2F 再分发条款** | 明确回答：衍生标注能否随论文发布？ |
| ☐ | 1.1.2 备选预案 | 若受限 → 发「标注索引 + 转换脚本」不发媒体；或换 InterAct |
| ☐ | 1.1.3 伦理审批（自录部分） | 提交申请，取得受理回执 |

**MM-F2F 已去标识化**（>10,000 张合成人脸替换 + 声纹扰动），
且其消融显示去标识对性能影响很小（全原始 0.836 vs 全去标识 0.823）
→ **主训练数据用它可绕开人脸合规的大部分风险。**

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
（v2 的 100–150 人时是因为要人审全部训练集；v3 只审测试集，人力反而下降）

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

**★ 训练集增广改为即时合成**（算力升级后的变更）：
训练时在 collator 内在线加噪，不预计算落盘 —— 省磁盘，且每个 epoch 见到不同噪声实例。
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
| **C（上界，算力升级后新增）** | 同 B | **+ LoRA r=32 且解冻 `vad_lm_head`** | H-A | 看头部解冻是否比纯 LoRA 更好；**必过 WER 门禁** |

~~早融合 + 全冻结~~ **已删除** —— 最差组合（注入了但主干不会用，见 arch §3.2 证据 E）。

**臂 C 的成本**：只多解冻一个 `H×6` 的小头（≈0.1 M），前向不变。
它同时替代了原 §3.1 里「S3 可选阶段」的角色，**把可选项升级为固定对照臂**。

**⚠️ 臂 C 的风险**：解冻 `vad_lm_head` 会直接改动原生 6 类话轮输出，
因此 ①WER 门禁必测 ②原生话轮能力的漂移必须量化（§4.7 的同权重对照会失效，需单独跑基线）。
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
      （算力升级后由「可选」升级为固定对照臂，见 §2.3）
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

**★ 指标按格切片是 v3 的实质变更**（依据 §1.3）：
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

```
横轴：响应延迟中位
纵轴：错误打断率
散点：扫 silence_end_frames ∈ {2..10} × tail_max_frames ∈ {3..15} 的固定阈值组合
      → 连成 Pareto 前沿
星号：我们的自适应模型
```

**验收**：自适应模型落在固定阈值 Pareto 前沿**外侧**。

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
| ☐ | **4.5.8 语言拆分** | 中文 / 英文分别报告 → 视觉线索是否跨语言迁移（Kurata 仅日本人说英语，MM-F2F 仅中文）|

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
                            silence_end_frames (240 ms)
                            tail_max_frames    (400 ms)
                          factor = 1 + g·conf·(2p−1)，clamp
```

**模型负责判决，控制器负责时序。** 照抄仓库现有的外部信号注入模式：
`AcousticVoiceGate`（`server.py:102` → `on_frame(..., acoustic_active)`），
接入点全仓库仅一处（`server.py:115-120`）。

## 5.2 论文

**Title**：*Visually-Grounded Semantic Endpointing for Robot Speech Interaction*

| ☐ | 任务 |
|---|---|
| ☐ | 5.2.1 三点贡献（C1 数据 / C2 方法 / C3 真机） |
| ☐ | 5.2.2 Related Work：补 X2-Turn(2608.10878)、SoulX(2603.14877) 全文表格 + **§0.1.1 的语料定位表** |
| ☐ | 5.2.3 差异化表述（见下） |
| ☐ | 5.2.4 Limitation 如实写 clean 增益、**仅中英两语**、单一 base、**轴 2 为自动标注（未人工复核全量）** |
| ☐ | 5.2.5 Demo 视频：思考停顿不被打断 vs 基线抢话 |
| ☐ | 5.2.6 发布：HF(model+dataset) + GitHub(代码+标注工具) |

**5.2.3 必须处理的四条外部声称**：

| 竞品 | 声称 | 我们的表述 |
|---|---|---|
| **AV-Dialog** [2511.11124] | 视听 + "semantically grounded turn-boundary detection" | 其输出为 `<SOT>/<SOB>` 事件，**无 complete/incomplete 判决**；需 8B + 128×A100 重训；AV-HuBERT 引入 120 ms 前视；**仅唇部特征**（作者自列 limitation） |
| **★ Kurata'23** [Interspeech'23] | 视听 + "end-of-utterance prediction" | **最接近的先行工作，必须正面且尊重地处理**。它确实是视听 + 静音点触发，但标注的是**行为结果**（continue/end，由对方是否接话决定），坍缩了语义完整/不完整（§0.1.1）；此外：语料未公开（NEDO 内部）、IPU 触发非帧同步、非 LLM（wav2vec2+BERT+X3d 拼接）、场景为教师-学生在线面试。**我们引用其视觉线索消融结论作为设计依据**（眼>嘴>头姿），并用 §4.5.7 实证标注轴的影响 |
| **Qwen3.5-Omni** [2604.15804] | "native turn-taking intent recognition" | **215 项评测中零个话轮/端点任务，该能力未获实验支撑**；第三方评测 [2606.26083] 显示该系列存在感知—行动脱节；**权重仅 API，无法作为可扩展研究基座** |
| **MiniCPM-o 4.5** [2604.27393] | "减少对外部 VAD 模块的依赖" | 其全双工评测在**无音频**的 LiveSports-3K-CC 上；自身消融显示 chunk 缩到 0.2 s 即崩塌 |

**不要**声称「首个视觉引导语义 VAD」。差异化落在
**轻量 / 免重训 / 零前视 / 可部署 / 有配套数据**。
C1 的首创声明限定在**「语义完整性标注 + 视听」的交集**上（§0.1.1 定稿措辞）。

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
├── X2-Turn/                  ← 只读，base 权重来源            @53d3b9a
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
| V1 | `hidden_size`/`vocab_size`/层数 | ⬜ | Phase 0.4 |
| V2 | id 41+ 是否空闲 | ⬜ | Phase 0.4 |
| V3 | decoder layer 0 模块路径 | ⬜ | Phase 0.4 |
| V4 | `−1` 偏移语义 | ⬜ | Phase 2.7 单测 |
| V5 | batch>1 与变长 padding | ⬜ | Phase 3.4 |
| V6 | 两趟推理能否合并 | ⬜ | Phase 5.1 |
| V7 | turn head 是否吃 delay tokens | ⬜ | Phase 0.4 |
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
