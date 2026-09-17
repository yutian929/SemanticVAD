# Phase 0 执行日志

对应 [`../implementation-plan.md`](../implementation-plan.md) §P0.1–P0.4。
执行顺序（计划规定，不能只按编号）：**P0.1 → P0.2 → P0.3/P0.4**。

> ## 📌 Phase 0 结论摘要（TL;DR，给赶时间的人）
>
> 1. **环境 + 4 个待核实项全解锁**：H=3072/V=131072/26 层；**ci 头零参数方案 H-A 成立**（id 41/42 空闲）；
>    视觉融合挂点 = `base_model.model.language_model.layers.0`；turn head 吃 480ms 前视。
> 2. **探针 A（干净）**：X2-Turn 音频 hidden 判 complete/incomplete **AUC≈0.99**（去时长混淆后 [0.77,0.99]）
>    → **干净条件音频近天花板，视觉在语义线上几乎无空间。**
> 3. **Gate（词法明显的句中停顿）**：**音频没被骗**——强 LM 直接读懂"比较…""是否…"这类悬空未完。
> 4. **★ 噪声 gate（竞争说话人）**：音频完整性信号（剔除时长）**单调崩塌 0.77→0.67→0.60→0.47**（SNR 10/5/0）
>    → **音频在声学退化下确实失效，视觉有潜在主场（necessary condition 满足）。**
> 5. **新 framing（有实证支撑）**：不是"视觉帮语义判断"，而是
>    **"音频被干扰、完整性信号塌掉时，视觉维持话轮判断"**。
> 6. **尚未证明**：视觉**能否补回**（sufficiency）——需带视频 AV 数据。**在此之前既不宣称 C2 成立、也不宣称失败。**
>
> 详见 [`../implementation-plan.md`](../implementation-plan.md) 顶部 changelog 与本文件各节。

状态图例：✅ 完成 ｜ 🔄 进行中 ｜ ⛔ 受阻 ｜ ⏳ 待做

| 任务 | 状态 |
|---|---|
| P0.1 V1+V2（读配置） | ✅ H-A 零参数成立 |
| P0.2 环境+基线+V3+V7 | ✅ 全解锁 |
| P0.3 探针 A（干净基线） | ✅ AUC≈0.99（去混淆 [0.77,0.99]）→ >0.85 |
| Gate：词法明显停顿 | ✅ 音频没被骗（pivot 在此不成立） |
| Gate：竞争说话人噪声 | ✅ 音频崩塌 0.77→0.47 → necessary condition 满足 |
| P0.4 探针 B（视觉充分性） | ⏳ 待 AV 数据（下一步） |

---

## P0.1 — V1 / V2（纯读配置）✅ 2026-09-13

**做法**：`AV-SemanticVAD/scripts/p0_1_verify_config.py`，读
`X2-Turn/models/X2-Turn-4B-0812/{config.json,tekken.json}`。
日志：`AV-SemanticVAD/results/p0_1_config.log`。

### V1 — 骨干规格（定 Projector / LoRA 参数量）

| 字段 | 值 |
|---|---|
| `text_config.hidden_size` **H** | **3072** |
| `text_config.vocab_size` **V** | **131072** |
| `num_hidden_layers` | **26** |
| `num_attention_heads` / `head_dim` | 32 / 128 |
| `audio_config.hidden_size` | 1280 |
| `audio_length_per_tok` / `downsample_factor` | 8 / 4 |
| **`default_num_delay_tokens`** | **6** ← 与 V7 前视相关 |
| `dtype` | bfloat16 |
| `architectures` | `VoxtralRealtimeForConditionalGeneration` |

**结论**：VisualProjector 目标结构 `D_v→256→H(=3072)`（arch §3.9 窄版）；LoRA 挂在 3072 维、26 层解码器上，`r=32` 参数量约 15–25M（符合 §0.4 与 ≥30K 数据要求匹配）。

### V2 — token id 41–50 是否空闲（决定 ci 头能否零参数）★

- tekken v7，`default_num_special_tokens=1000`；special_tokens 里
  ids 35–40 = `<SPECIAL_35..40>`，ids 41–50 = `<SPECIAL_41..50>`，**全是未命名占位符**。
- 交叉核实：`modeling.py:15` `TURN_CLASS_IDS = (35,36,37,38,39,40)` —— X2-Turn 正是
  在代码里**给占位符 35–40 赋予话轮语义**，词表本身不区分。
- 关键：`modeling.py:61` `self.vad_lm_head = nn.Linear(hidden=3072, vocab=131072, bias=False)`
  —— 输出覆盖**全词表**，故 id 41/42 的输出行**已存在**（初始化自 `base_model.lm_head`）。

**结论（V2 定档）**：`CI_CLASS_IDS = (41, 42)` 空闲，**方案 H-A（零新增参数，复用
`vad_lm_head` 在 id 41/42 的输出行）成立**。ci 判决 = 在逐帧循环里对 `vad_logits[...,(41,42)]`
做 softmax（挂在 `inference.py:86` 现有 turn softmax 旁边），无需新建 head。
不必退 H-B。

**附录 D 更新**：V1 ✅、V2 ✅。

---

## P0.2 — 环境、基线复现、V3、V7 ✅ 2026-09-13

**环境**（P0.2.1）✅：见项目记忆 `env-setup`。conda env `x2-turn`
（torch 2.11.0+cu128，2×H20 可见），上游 `demo_turn`/`voxtral_realtime` 可编辑安装，
我们新增依赖（peft/mediapipe/sklearn/…）已装；权重软链在 `X2-Turn/models/X2-Turn-4B-0812`。

**脚本**：`AV-SemanticVAD/scripts/run_backbone.py`（`--device cuda:0`）。
**产物**：`tests/golden/backbone.json`、`results/p0_2_backbone.json`。

| 子任务 | 结果 | 验收 |
|---|---|---|
| P0.2.2 加载 base | bf16 载入 47.5s，**VRAM ≈ 9.67 GB** | ✅（计划估 ≈8GB） |
| P0.2.3 帧级输出 | **frame_ms=80**；sample_en.wav **53 帧**；转写 `"hello can you tell me what the weather is like today"` | ✅（期望 ≈53 帧、80ms/帧） |
| P0.2.4 固化 golden | `tests/golden/backbone.json`（含每帧 6 类概率） | ✅ 回归基线已建 |
| P0.2.5 **V3** | decoder layer 0 = **`base_model.model.language_model.layers.0`**；hidden 源 = `base_model.model(...).last_hidden_state`（内层类型 `VoxtralRealtimeModel`；另有 `audio_tower.layers.0`） | ✅ 视觉融合 hook 挂点确定 |
| P0.2.6 **V7** | delay_ms=80 与 480 的 turn 帧**不同**（480 更干净/更高置信） | ✅ **turn head 吃 delay tokens，前视不能免费放松到 80ms** |

**参考**：`TURN_CLASS_NAMES = (idle, noidle, speaking, turn_end, backchannel, uncertain)`，
对应 `TURN_CLASS_IDS=(35..40)`；完整性判决关注 `turn_end` / `uncertain`。
`infer_asr_turn` 逐帧读出用 `prediction_index = prefix_length + frame_index − 1`（`inference.py:165`，已核对）。

**V7 的设计含义**：X2-Turn 默认 `default_num_delay_tokens=6`（=480ms 前视）。turn 预测显著依赖前视，
因此我们视觉臂若要与音频基线同操作点对照，需匹配前视量；把前视压到 80ms 会牺牲话轮质量
——这条要写进 arch 的前视预算，并影响 §4.4 的延迟-精度 Pareto。

**附录 D 更新**：V1 ✅、V2 ✅、V3 ✅、V7 ✅。

---

## P0.3 — 探针 A 🔄（数据已解决，抽取中）

### 数据来源决策（★ 对计划的正向偏差）

计划设想探针用「~1K 弱标注事件（Phase 1.2 早期产物）」，需自建 ASR→静音切分→LLM 标注流水线。
实际发现更省的现成金标数据：**SoulX-Duplug-Eval / `Easy-Turn-Testset-en`**
（Apache-2.0，从 ModelScope 下，本机 huggingface.co 不通、hf-mirror 只 308 转发、ModelScope 可用）。

- **318 complete + 299 incomplete = 617 条 utterance**，每条含转写文本 + 24kHz wav，标签即**语义完整性**。
- **免 LLM 弱标注**，标签质量高于原计划的弱标注。
- ⚠️ **已识别混淆**：incomplete 片段系统性更短（中位 1.77s vs complete 3.09s）。探针 A 训练脚本
  加了 **length-only 基线**来量化"是否在借时长而非语义"。

数据落位 `data/SoulX-Duplug-Eval/`（已 gitignore）。中文 `Full-Duplex-Bench-zh.zip` 暂未取，留待需要时。

### 方法（不改第三方代码）

`scripts/probe_a_extract.py`：复刻 `inference.py` 前向但额外取 hidden——
encode→generate(ASR 对齐)→对 `model.base_model.model(...)` 单独前向取 `last_hidden_state`
→逐帧索引 `prefix_length+frame_index−1`→取**末帧(判决点)** hidden 与全帧 mean-pool，
并用 `vad_lm_head` 读该帧 6 类 turn 概率（frozen turn head 零训练基线）。
双卡分片（cuda:0/1 各 ~308 条，稳态 ~4.2s/条，约 22 分钟）。

`scripts/probe_a_train.py`：分层 5 折 CV 训 logistic regression，报告 feat_last / feat_mean /
turn_probs / turn_end / **length** 各特征集 AUC，套用 §P0.3 判据。

### 结果（2026-09-13）✅

617 条全部抽取完成（双卡 ~10 分钟）。分层 5 折 CV：

| 特征集 | AUC | 说明 |
|---|---|---|
| **feat_last（末帧 hidden，主指标）** | **0.9944** | 判决点 hidden 近乎完美线性可分 |
| feat_mean（全帧 mean-pool） | 0.9989 | — |
| turn_probs（frozen turn head 6 类） | 0.7779 | 模型现成读出头 |
| turn_end_prob（单特征） | 0.7162 | — |
| **length_framecount（时长混淆基线）** | **0.8633** | ⚠️ 混淆很强 |

**时长混淆控制**（`results/probe_a/probe_a_length_control.json`）：
- 长度匹配子集（band=[28,53] 帧，N=471，两类均衡）：feat_last 仍 **0.9949**（上界）。
- 线性剔除长度后的残差特征：feat_last 降到 **0.7725**（过度校正下界）。
- → **去混淆后完整性可分性落在 [0.77, 0.99]，两端均 ≥0.70，上界 ≫0.85。**

**判据命中 >0.85**：音频基线在**干净孤立语句**上已近天花板，视觉在 clean 条件几乎无语义增益空间。
→ **必须改走分层汇报**（真思考停顿 × 噪声 × 远场），**不报全集 F1**。这**印证了计划反复警示的风险**。

**两条重要注解**：
1. Easy-Turn 是刻意"简单"的 clean 孤立语句（complete vs 截断），**不含项目真正目标的话轮内思考停顿**（"我想去…嗯…"）。0.994 是**简单情形的天花板**；最终决定性判据仍需**硬停顿数据**（GRASS/自采）复核，但趋势已很明确：凡接近 clean 孤立语音，音频基线就赢。
2. **frozen turn head 仅 0.778 vs 可训线性探针 0.994** → 完整性信息**在 hidden 里，但现成读出头没充分暴露**。这正面支持 **H-A**（冻结 hidden + 训一个小读出头即可提取），也说明我们的 ci 判决头确实有活干。

**对路线的影响**（写给下一步）：
- C2（模型）的卖点**不能是** clean 条件的聚合 F1 增益（几乎必为 +0~1%，会被 AV-Dialog/MM-F2F/VideoFDB 反驳）。
- 探针 B 的解读门槛因此抬高：要看的是**视觉在音频失败切片上的条件增量**，不是 B 的绝对值。
- 下一步做**探针 B** 前，应先想清楚"音频在哪失败"——那才是视觉唯一可能有价值的地方（噪声/远场/被截断/重叠说话人）。

**附录 D 更新**：探针 A 的 AUC 数字已出（Gate W2 的一半达成）。

---

## ★ Framing 决策（2026-09-14，基于探针 A + 用户判断）

**触发**：探针 A 显示 clean 孤立语句上音频基线近天花板（0.994）。用户据此判断：
「贡献点不能是'加了一路视觉',应提出**自己的 benchmark**,专门针对音频解决不了的情形
——比如人在思考、一下想不起词,此时可能**无声**,但**看得出在思考**,就不该判 complete。」

**核对结论**：这与计划的 **C1（AVSC-Corpus）** 一致,且用户描述的正是双轴表里的
**`incomplete × hold`（思考停顿/hold）**格——GRASS 称其占 turn-hold **≈39%**（§0.1.1，待 V12 复核）。
探针 A 只是把 C1 从"数据贡献"抬成了**承重贡献**:既然音频在简单情形已赢,科学价值就在
**构造并隔离出音频会失败的那批情形**,并证明 SOTA 音频模型在其上崩溃。

**锐化后的论文结构（提案,待用户确认后同步进 plan）**：
1. **Benchmark（主贡献）**：AVSC-hard —— 专门 stress 音频不足的情形(思考停顿、想不起词、
   语义未闭合但声学上"像说完了"),带**双轴 + 视听**标注。相对 Easy-Turn/MM-F2F/Full-Duplex-Bench
   的差异化 = 语义完整性 + 思考停顿 + AV + 双轴(§0.1.1 related-work 表已论证此缺口)。
2. **Finding（punchline）**：现成音频模型在旧 benchmark 近满分(我们实测 X2-Turn 0.994),
   但在 AVSC-hard 上**跌到近随机**——这正是 benchmark 存在的理由。
3. **Method**：视觉(或其他)信号在该 benchmark 上**恢复** X%。

**★ 决定性实验(下一步的 gate,统一了"探针B"与"硬停顿复核")**：
先构造/获取一小批**思考停顿 / incomplete-hold** 事件,**重跑探针 A(纯音频)**。
- 若音频 AUC **显著下跌**(例如从 0.99 → 0.6~0.7)→ 这个 gap **同时**是 benchmark 的立身之本
  和视觉唯一的可为空间。**pivot 成立**。
- 若音频**仍然高**(韵律/呼吸/填充词泄露了答案)→ benchmark 论点削弱,须重想。

> ⚠️ **honest risk**：目前只证明了"音频在简单情形赢",**尚未证明"音频在硬情形输"**。
> 后者是假设,必须先用硬停顿数据验证,不能假定。这是投入前最便宜的一步。

**硬停顿数据候选**：GRASS[2504.09980]（含 completeness 标注,但纯音频德语→只够验证探针A是否下跌）；
MM-F2F 子集（**英文** AV，可挖 hold；⚠️ 只发标注+YouTube链接，媒体须自下）；自采中文小样（有视频,直达 incomplete-hold）。

**→ 完整的 benchmark 设计草案见 [`../benchmark-proposal.md`](../benchmark-proposal.md)**
（提案 / 待 gate 验证 / 非权威）。

---

## Gate 实验（硬停顿纯音频探针）— 2026-09-14

数据：Full-Duplex-Bench-zh（pause_handling 239 句中停顿 / turn_taking 155 完整话轮）。

### ⚠️ v1 作废（confounded，已撤回）
第一版把 incomplete 截到句中停顿(短)、complete 用整句(长)，导致 **length-only AUC=0.993**，
feat_last 0.999 基本在量长度；且判决帧读的是**尾部静音**，turn head 恒 idle=1.0。
教训：截断长度差 + 读静音帧 = 双重伪迹。**不作数。**（`results/probe_gate/gate_report.json` 保留作反面记录）

### v2 逐帧诊断（4 样本，定性但清晰）
直接看模型在**句中停顿**和**真正说完**两处逐帧输出 `turn_end`：

| 样本 | 句中停顿处是否误报 turn_end | 真正说完处 turn_end |
|---|---|---|
| pause/69「前两年过得比较…辛苦现在好多了」 | ❌ 停顿(1.21–1.91)全程 idle，不误报 | ✅ 2.8s P=0.99 |
| pause/34「售后服务是否…完善我有点担心」 | ❌ 停顿(1.09–1.61)idle，不误报 | ✅ 2.64s P=0.97 |
| turn/1（完整句） | — | ✅ 7.12s P=0.61（另 0.7s 处对"你知道吗"误报 0.57） |
| turn/10（完整句） | — | ✅ 3.6s P=0.99（中间 clause 边界也点 turn_end） |

### 结论：**gate 未通过（pivot 在此数据上不成立）**
- **X2-Turn 没有被这些停顿骗到**：pause_handling 的停顿前文都是**词法/句法上明显未完**
  的悬空虚词（"比较…""是否…""又…""有点…"），X2-Turn 的强语言模型能识别，停顿处保持 idle、
  只在**真正语义结束**处才点 turn_end。
- 副发现：模型会在**中间可完结的子句边界**（"你知道吗""奇怪不奇怪"）点 turn_end——
  那是 `complete×hold`（可完结但说话人继续），另一种现象，非"被静音骗"。

### 对路线的**修正**（重要）
"音频在思考停顿上会失败"这个假设，**在"词法明显未完"的停顿上不成立**——强音频 LM 已能搞定。
视觉真正的可为空间**比"思考停顿"更窄**：只在**词法+韵律都对完整性有歧义**的停顿
（停顿前文句法上像说完了、但说话人其实没完，如重启/追加/列举中断）。
→ benchmark **不能建在"悬空虚词"这类容易样本上**，否则又只是证明音频赢；
必须专门构造"音频歧义"的硬样本。这是对 `benchmark-proposal.md` §5.1 硬情形分类的收紧。

**下一步待定**：(a) 把上面 4 样本的诊断**全量化**（239+155 全跑，出"停顿处误报率/真端检出率"的确切数字）；
(b) 直接去构造/寻找"音频歧义"的更硬样本再验 gate。

---

## 噪声鲁棒性 gate（2026-09-14）✅ necessary condition 满足

**动机**：干净条件视觉没空间；测音频的**失效区**（竞争说话人，AV-Dialog 里视觉 +13% 的场景）。
**做法**：`scripts/probe_noise_extract.py` 把 Easy-Turn 加**竞争说话人**噪声（随机混另一条 utterance，
按 SNR 缩放，时长不变）→ 重跑纯音频探针；`probe_noise_eval.py` 出退化曲线。双卡，~35 分钟。

| 条件 | feat_last | 仅时长 | **剔除时长(真语义)** |
|---|---|---|---|
| 干净 | 0.994 | 0.863 | **0.772** |
| SNR 10 | 0.984 | 0.863 | **0.673** |
| SNR 5 | 0.954 | 0.863 | **0.598** |
| SNR 0 | 0.898 | 0.863 | **0.474** |

**结论**：剔除时长后的真语义信号 **单调崩到近随机（0.77→0.47）**；仅时长基线恒 0.863
→ 下滑是噪声毁语义、非长度伪迹。**音频在竞争说话人下确实失效 → 视觉有潜在主场（必要条件满足）。**

**边界（诚实）**：
- ✅ 证明"音频会失效"（necessary）；❌ **未**证明"视觉能补回"（sufficiency，需带视频 AV 数据）。
- Easy-Turn 孤立句 + 合成竞争说话人；最终结论需真 AV 数据 + 真噪声。

**新 framing**：从"视觉帮语义判断"改为**"音频被干扰、完整性信号塌掉时，视觉维持话轮判断"**——
探针 + 文献（AV-Dialog +13%）双支撑。已写入 `implementation-plan.md` changelog（2026-09-14）。

**下一步**：取带视频 AV 数据（MM-F2F 子集 / 自采），在噪声下验证**视觉能否补回**（充分性）。
sufficiency 未验前，不宣称 C2 成立、也不宣称项目失败。

---

## P0.4 — 探针 B ⏳（待探针 A 结论）

需带视频的同类事件（MediaPipe 24 维→GRU）。Easy-Turn 是纯音频无视频；
探针 B 数据留到确认要做视觉后再取（MM-F2F 子集）。见与用户确认的 A→B 顺序。
