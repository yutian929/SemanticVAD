# Benchmark 设计提案：AVSC-Hard（暂定名）

> ## ⚠️ 文档状态：提案 / 待验证 / **非权威**
>
> - **性质**：设计草案,记录一个**尚未被证实**的 framing pivot,便于日后接续,不代表已定论。
> - **不是权威**：任务/Gate/指标的唯一权威仍是 [`implementation-plan.md`](implementation-plan.md);
>   架构论证在 [`architecture.md`](architecture.md);文献数字在 [`../research/evidence-table.md`](../research/evidence-table.md)。
>   本文与它们冲突时,**以它们为准**。
> - **生效条件**：本提案的全部立论都挂在一个**决定性 gate 实验**上（见 §7）。
>   **gate 通过前不得据此改动上述权威文档**;gate 通过后再把结论一次性同步进权威并打 changelog。
> - **来源**：探针 A 结果（见 [`implementation-execution/phase0.md`](implementation-execution/phase0.md) §P0.3
>   与「★ Framing 决策」）+ 用户 2026-09-14 的判断。
> - **最后更新**：2026-09-14

---

## 1. 为什么会有这份提案（触发）

探针 A（在 SoulX-Duplug-Eval / Easy-Turn-en 上）实测：**冻结 X2-Turn 的判决点 hidden 线性探针，
预测 complete/incomplete 的 AUC = 0.994**（去除时长混淆后仍落在 [0.77, 0.99]）。

含义：**在干净、孤立的语句上，纯音频语义 VAD 已接近天花板。**
因此"给音频模型加一路视觉、在干净集上报聚合增益"这条路几乎必然只有 +0~1%，
且会被 AV-Dialog（clean +1.3%）、MM-F2F（clean +1.2%）、VideoFDB（加视频 6/7 变差）同时反驳。

**结论**：贡献点不能是"加了一个模态"。应把**承重贡献换成一个 benchmark**——
专门隔离**音频解决不了**的话轮完整性判断情形，并证明 SOTA 音频模型在其上崩溃。
这与计划原有的 **C1（AVSC-Corpus）** 同源，探针只是把 C1 从"数据贡献"抬成了主贡献。

---

## 2. 核心现象：音频不足的完整性判断

目标情形（用户原话的锐化）：**人在思考、一下想不起词——此时可能几乎无声，
声学上"像说完了"，但从语义/视觉看他还没说完，不该判 complete。**

用计划的**双轴**（见 plan §1.3 / §0.1.1）定位，这落在 `incomplete × hold` 格：

|  | 说话人继续(hold) | 对方接话(change) |
|---|---|---|
| **语义完整点停顿** | complete × hold（合法的 TRP，可接） | complete × change |
| **语义不完整点停顿** | **`incomplete × hold` ← 本 benchmark 的靶心** | incomplete × trail-off |

`incomplete × hold` 不是边缘情况：GRASS [2504.09980] 称其占全部 turn-hold **≈39%**
（⚠️ V12，投稿前须核原文）。**这是"错误打断率"这个主指标真正要区分、而行为标注无法监督的东西。**

**音频为什么会在这里失败（假设，待 §7 验证）**：思考停顿的声学特征（静音、拖长、填充词"嗯…"）
与真正说完的静音高度相似；区分二者需要"下文语义是否闭合"或"说话人是否仍在组织语言"的证据，
后者恰恰是**视觉**（注视回避、口型仍在动、思考表情）可能携带、而音频缺失的。

---

## 3. Benchmark 是什么（一句话 + 任务定义）

**AVSC-Hard**：一个**视听、带语义完整性（双轴）标注**的评测集，
**专门 stress 音频不足的话轮完整性判断**（思考停顿 / 想不起词 / 语义未闭合但声学像闭合）。

- **任务**：给定停顿点前的音（视）频，**判别式**输出该点 `complete / incomplete`（logits 直出，不 `generate()`）。
- **单位**：停顿点事件（≥200ms 静音切出），逐帧对齐（80ms/帧）。
- **标注**：轴 1 语义完整性（complete/incomplete）+ 轴 2 行为结果（hold/change），双人 κ 抽检。
- **模态**：音频；视觉 24 维 MediaPipe 特征（注视/口型/头姿，@12.5Hz，含 valid/confidence）。

---

## 4. 与现有 benchmark 的差异化（niche）

（依据 plan §0.1.1 的 related-work 表，补入我们实测）

| 语料/benchmark | 语义完整性标注 | 视频 | 思考停顿(hold)靶心 | 规模/语言 |
|---|---|---|---|---|
| Easy-Turn(-en) | ✅ complete/incomplete | ❌ | ❌ 刻意"简单"、孤立句 | 617 / 英 |
| MM-F2F | ❌ KEEP/TURN/BC | ✅ | ❌ | 51K turn / 英 |
| Full-Duplex-Bench | ❌ 交互场景 | 部分 | ❌ | — |
| GRASS | ✅ κ=0.875 | ❌ 纯音频 | ⚠️ 有但音频 only、95min、德语 | 95min / 德 |
| Kurata'23 | ❌ **行为结果** | ✅ | ❌ | 21.7K / 日式英语 |
| **AVSC-Hard（本提案）** | ✅ **+ 双轴** | ✅ | ✅ **靶心即 incomplete-hold** | 目标 ≥? / 中+英 |

**差异化落点**：语义完整性 + **思考停顿靶心** + 视听 + 双轴。
**不写**"首个视觉引导语义 VAD"（模型层面首创声明易被推翻）；卖点在**数据/评测的缺口**本身。

---

## 5. 设计细节（草案）

### 5.1 硬情形分类（audio-insufficient taxonomy）
构造/筛选时按"为什么音频不足"打类型标签，用于分层汇报：
- T1 思考停顿 / 词检索失败（"我想去…嗯…"）
- T2 语义未闭合但声学像闭合（句末降调但句子没说完）
- T3 拖长 / 犹豫（trailing）
- T4 填充词后停顿（"那个…"、"就是…"）
- （对照）E0 真正说完的静音（complete×hold 的合法 TRP）

### 5.2 声学条件切片（stratified）
clean / 加性噪声 / 远场混响 / 重叠说话人。噪声**训练时即时合成**，测试集**固定种子预计算**（plan §1.5）。

### 5.3 数据来源与构造
- 验证阶段（gate）：**GRASS**（纯音频，带 completeness，够验证"音频是否下跌"）。
- 视听主体：**MM-F2F 子集**（**英文** AV，挖 hold 事件）+ **自采中文小样**（有视频，直达 incomplete-hold）。
- 流水线复用 plan §1.2：ASR+时间戳 → 静音切分 → LLM 弱标注(轴1) → 视觉特征 → 行为轴自动标注(轴2) → 人审测试集。
- ⚠️ 许可/伦理：MM-F2F 再分发条款待核（plan §1.1.1）；自采需伦理审批（plan §1.1.3）。

### 5.4 规模
承袭 plan §0.4 的硬耦合：若要支撑 LoRA r=32 训练需 ≥30K 事件；但**benchmark 的测试集**只需
数百条人审 + κ 抽检即可。弱标注全量铺开做训练/开发集。

---

## 6. 指标与要报的基线

**主 punchline 指标**：同一 SOTA 音频模型在 **Easy 集 vs AVSC-Hard** 的 AUC 落差
（我们已有 Easy 端 X2-Turn = 0.994；期望 Hard 端显著下跌）。

**分层指标**：按 §5.1 类型 × §5.2 声学条件切片报 AUC / 错误打断率 / 响应延迟 / 固定阈值 Pareto（plan §4.4）。

**必报基线**：
- chance、**length-only**（时长混淆基线，探针 A 已证其强，必须永远同台报）
- 音频 only（X2-Turn / SoulX）、视觉 only、AV 融合
- **frozen turn head vs 可训读出头**（探针 A：0.778 vs 0.994，说明信息在 hidden 但需训小头 → 支持 H-A）
- 人类参考上界

**纪律（plan §0.2，不可违反）**：同构对照（唯一 mask 掉视觉）、WER 门禁 ΔWER≤+0.5%、
降级保证（视觉失效逐比特等于 −视觉臂）、判别式输出。

---

## 7. ★ 决定性 gate 实验（本提案的全部立论都挂在这里）

**假设**：音频在 §5.1 的硬情形上会**显著失败**。目前**只证明了音频在简单情形赢，尚未证明它在硬情形输。**

**实验**：取一小批真正的思考停顿 / incomplete-hold 事件，**用探针 A 同一套代码重跑纯音频探针**。

| 结果 | 判定 |
|---|---|
| 音频 AUC 显著下跌（如 0.99 → 0.6~0.7） | **pivot 成立**：该 gap 同时是 benchmark 立身之本 + 视觉唯一可为空间 → 正式改权威、建 benchmark、上视觉（探针 B） |
| 音频仍高（韵律/呼吸/填充词泄露答案） | **pivot 削弱**：benchmark 论点不成立，须重想切片或换角度 |

这是投入前**最便宜的一步**（CPU 分钟级）。**在它通过前，本提案不升为权威。**

---

## 8. Kill criteria（诚实的自毁条件）

- 若 gate 显示音频在硬情形**也不失败** → benchmark 没有存在理由，回到"数据集 + negative result"下限。
- 若视觉在硬情形的增益**仅来自噪声鲁棒性、而非语义**（与 AV-Dialog/VideoFDB 一致）→
  卖点收缩为"噪声/远场鲁棒性"，而非"视觉懂语义"。
- 若 `incomplete-hold` 在我们数据里占比**远低于 39%**（V13）→ 下调 C1 论证强度、解释场景差异。

---

## 9. 风险与注意

- **Easy-Turn 的教训**：数据一"简单"，音频就近满分。构造 AVSC-Hard 时必须主动**对抗 length/声学捷径**
  （length-matched、剔除声学可分的平凡样本），否则又造一个音频能解的集。
- 标注主观性：complete/incomplete 在 hold 情形边界模糊，必须双人 κ + 明确标注手册。
- 前视预算：探针 P0.2.6/V7 已证 turn head 吃 delay tokens（默认 6 token=480ms）；
  benchmark 的延迟-精度 Pareto 要把前视量作为受控变量。

---

## 10. 命名候选（待定）

`AVSC-Hard`（与 plan 的 AVSC-Corpus 一致，稳）｜ `HOLD-Bench`（点出 incomplete-hold 靶心）｜
`ThinkPause` ｜ `SILENT-INTENT`。默认 `AVSC-Hard`。

## 11. gate 通过后要同步进权威的清单（预备）

1. `implementation-plan.md` §0.1 三大贡献重排（C1 benchmark 升主贡献，C2 视觉降为"解法之一"）、
   §0.1.1 related-work 补入实测、新增 benchmark 交付物与 Gate。
2. `architecture.md` 模型角色改为"benchmark 上的解法之一"。
3. `research/evidence-table.md` 记入 X2-Turn 0.994（Easy）与硬停顿 AUC。
4. `START-HERE.md` 30 秒版 + 状态表。
5. 顶部加带日期 changelog（= 版本管理的正确形态，不开平行文档集）。
