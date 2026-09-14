# START HERE

> **本文件写给首次接手本项目的人或 AI agent。**
> 读完这一页，你应该知道：项目要做什么、现在到哪一步、下一步具体做什么。
>
> **最后更新**：2026-09-04 ｜ **状态**：调研与计划完成，**代码尚未开始**

---

## 1. 项目在做什么（30 秒版）

**问题**：语音对话系统判断「用户说完了没有」时，只用音频。
用户思考停顿（"我想去…嗯…"）常被误判为说完，于是系统抢话打断。

**做法**：在一个**已冻结**的音频语义 VAD 骨干（X2-Turn-4B）旁边，
用极少参数（Projector ≈0.8 M + LoRA r=32）接入一路**视觉**（注视/口型/头姿），
输出逐帧的 **complete / incomplete** 判决。

**三个贡献**：

| # | 内容 |
|---|---|
| **C1** | **AVSC-Corpus**：首个带**语义完整性**标注的**视听**语料（中/英，双轴标注） |
| **C2** | **AV-X2-Turn**：冻结骨干上的轻量视觉扩展 |
| **C3** | **真机系统评测**：错误打断率 / 响应延迟 / 固定阈值 Pareto / 降级安全 |

**目标**：IROS，约 6 个月，双卡 H20。

---

## 2. 按这个顺序读（总量可控）

### 必读（约 1.5 小时）

| 序 | 文件 | 读什么 | 行数 |
|---|---|---|---|
| 1 | 本文件 | 全部 | — |
| 2 | [`AV-SemanticVAD/plan/implementation-plan.md`](AV-SemanticVAD/plan/implementation-plan.md) | **第 0 部分概览（§0.1–0.4）+ Phase 0（§P0.1–P0.4）** | 前 ~250 行 |
| 3 | [`AV-SemanticVAD/docs/server-setup.md`](AV-SemanticVAD/docs/server-setup.md) | 全部（部署 + 已知坑） | 240 |
| 4 | [`AV-SemanticVAD/docs/code-anchors.md`](AV-SemanticVAD/docs/code-anchors.md) | 全部（参考代码在哪） | 90 |

### 动手对应 Phase 时再读

| 何时 | 文件 | 章节 |
|---|---|---|
| 写模型前 | [`plan/architecture.md`](AV-SemanticVAD/plan/architecture.md) | §3 全部（采纳的路线） |
| 做数据前 | `plan/implementation-plan.md` | §1（含 **§1.3 双轴标注体系**） |
| 写 Trainer 前 | `SoulX-Duplug-training/finetune.py` + `example_data_fisher.jsonl` | — |
| 写论文/答审稿人 | [`research/evidence-table.md`](AV-SemanticVAD/research/evidence-table.md) | 全部 |

### 可以不读（背景资料，需要时再查）

- `research/visual-semantic-vad-survey.md`（791 行，完整文献综述）
- `research/methodology.md`（检索方法论与可信度评级）
- `plan/architecture.md` §2（被否决的 Omni 路线及理由）

---

## 3. 现在的状态

| 项 | 状态 |
|---|---|
| 文献调研 | ✅ 完成 |
| 架构设计 | ✅ 完成（`plan/architecture.md`） |
| 实施计划 | ✅ 完成（`plan/implementation-plan.md`） |
| 目录骨架 | ✅ 已建（`AV-SemanticVAD/avsvad/**` 全是空的 `.gitkeep`） |
| **代码** | ❌ **一行都还没写** |
| 权重 | ❌ 未下载（不在库内，见 `server-setup.md` §3） |
| 数据 | ❌ 未获取 |

---

## 4. 下一步做什么（Phase 0，约 2 周）

### 第一批（§P0.1）：V1 + V2 —— 无依赖，第一天就能做完

**只需下载两个小文件，不需要环境、不需要完整权重**：

```bash
huggingface-cli download x-square-robot/X2-Turn-4B-0812 \
    --include "config.json" "tekken.json" \
    --local-dir /tmp/x2turn-meta
```

| 项 | 要查什么 | 怎么查 | 影响 |
|---|---|---|---|
| **V1** | `hidden_size` / `vocab_size` / 层数 | 读 `config.json` | Projector 与 LoRA 参数量定档 |
| **V2** ★ | **token id 41+ 是否空闲** | 查 `tekken.json` 词表 41–50 | 决定 ci 头能否**零参数**（方案 H-A）；不成立则退 H-B |

> ⚠️ **V3 和 V7 不在这一批** —— 它们需要模型已载入（`print(model)`）
> 或能跑推理（对比 `delay_ms`），因此归在第二批的 §P0.2.5 / §P0.2.6。
> 别被「待核实项」这个共同标签误导成可以一起做完。

### 第二批（§P0.2）：环境、基线复现，以及 V3 / V7

```bash
cd X2-Turn
conda env create -f environments/environment-transformers.yml   # 上游官方路径
conda activate x2-turn
python -m pip install -e "./voxtral-realtime[transformers]"
python -m pip install -e "./turn-demo"
# ⚠️ install.sh 是 fork 自制脚本、不是上游的，只作备选（THIRD_PARTY.md §1.2 B）
# 权重下载见 AV-SemanticVAD/docs/server-setup.md §3
```

验收：`turn-demo` 跑通；`infer_asr_turn` 输出 **80 ms/帧**（3.4 s 音频 ≈53 帧）；
固化 golden 到 `tests/golden/backbone.json`；**并顺带解锁 V3（§P0.2.5）与 V7（§P0.2.6）**
—— 此时模型已在手，正是查它们的时候。

### 第三批（§P0.3 / §P0.4）：★★ 两个探针 —— 这是 Phase 0 的真正目的

```
探针 A：冻结 X2-Turn → 抽 hidden → 线性探针 → 预测 complete/incomplete
探针 B：MediaPipe 24 维视觉特征 → 小 GRU → 预测 complete/incomplete
```

**两个探针都不需要**训练循环、LoRA、视觉分支、人工标注数据。
用 ~1 K 条弱标注样本即可，CPU 分钟级出结果。

> ### ⚠️ 探针 A 有个已知拦路石（已核实代码，别自己撞）
>
> **`inference.py` 的 `infer_asr_turn()` 拿不到 hidden** —— 它只返回 `output.vad_logits`。
> `hidden_states` 仅存在于 `modeling.py` 内部（`:128,139,160,163`）。
>
> **推荐做法**：对 `model.base_model.model(...)` 单独前向，取 `outputs.last_hidden_state`
> （这正是 `modeling.py:127-128` 自己的做法）。
> 逐帧索引用 `prefix_length + frame_index − 1`。
>
> 完整说明与另两个备选做法见 `plan/implementation-plan.md` §P0.3。
> **不要为此修改 `X2-Turn/` 里的代码**（只读约定），做法是「调用」不是「改」。

**判据（直接决定论文怎么写，务必看数字不要跳过）**：

| 探针 A 的 AUC | 含义 | 行动 |
|---|---|---|
| **> 0.85** | 音频基线已很强，视觉空间小 | **必须改走分层汇报**（按停顿类型 × 噪声条件切片），不报全集 F1 |
| 0.70–0.85 | 有真实空间 | 按主线推进 |
| < 0.70 | hidden 里话轮信息不足 | 考虑解冻 `vad_lm_head` 或加大 LoRA |

| 探针 B 的 AUC | 行动 |
|---|---|
| **> 0.62** | 视觉携带独立的 hold-intent 证据 → 核心卖点 |
| 0.55–0.62 | 弱信号 → 定位「噪声/远场鲁棒性」 |
| ≈ 0.50 | 视觉在 clean 下无用 → 只做噪声条件，或转 negative result |

> ### ⚠️⚠️ 最重要的一条：不要跳过探针直接训练
>
> 现有**全部**外部证据都指向「视觉增益来自噪声鲁棒性，不来自语义判断」：
>
> - AV-Dialog：clean 条件仅 **+1.3%**，干扰说话人 **+13.0%**
> - MM-F2F：clean **+1.2%**
> - VideoFDB：加视频后 **7 个模型中 6 个**时序对齐**变差**
> - Kurata'23 的 **+3.0** 建立在 wav2vec2-base（AUC 0.887）**弱基线**上，
>   我们的基线是 X2-Turn 4B，**不能拿 +3.0 当预期值**
>
> 如果不先用 2 周验证「视觉是否携带独立信息」就投入训练，
> **最可能的结局是在干净评测集上拿到 +0~1%，然后被 AV-Dialog 与 VideoFDB 同时反驳。**

**同时启动（最长串行依赖，第一天就要做）**：
核对 MM-F2F 再分发条款 + 提交伦理审批（见 `plan/implementation-plan.md` §1.1）。

---

## 5. 五条不能违反的纪律

摘自 `plan/implementation-plan.md` §0.2 与相关章节。**违反任何一条，结果不可信。**

| # | 纪律 | 违反的后果 |
|---|---|---|
| 1 | **同构对照**：增益声明必须来自「同权重、同数据、唯一 mask 掉视觉」 | 增益不可信 |
| 2 | **WER 门禁** ΔWER ≤ +0.5% | 转写坏了，话轮预测无从谈起 |
| 3 | **降级保证**：视觉失效 → 输出**逐比特**等于 −视觉臂 | VideoFDB 显示视觉常有害，审稿人必问 |
| 4 | **判别式输出**：logits 直出，不做 `generate()` | 生成式 >2× 误差且不可降级 |
| 5 | **第三方目录只读** | 无法声明「未修改基座」，也无法与上游 diff |

---

## 6. 三个最容易踩的坑

| 坑 | 说明 |
|---|---|
| **帧对齐错 1 帧** | = 全局 **80 ms** 系统性偏差。**危险在于它在指标上表现为「视觉略有帮助」，极难察觉。** 因此 `tests/test_alignment.py`（脉冲响应单测）是强制的 |
| **逐帧索引偏移 `−1`** | 正确写法 `prefix_length + frame_index − 1`（next-token 语义，见 `X2-Turn/…/transformers/inference.py:165`），写错则全局错位 |
| **误改第三方目录** | 三个目录已删内层 `.git`，改了**不会有任何 git 提示**。改造一律走 `AV-SemanticVAD/avsvad/` 包装 |

---

## 7. 有两个数字还没核实，但论文要靠它们

| # | 待核实 | 为什么重要 | 何时做 |
|---|---|---|---|
| **V12** | GRASS [2504.09980] 的 `incomplete-hold` 占 turn-hold **≈39%**、κ=0.875 | **C1 的论证承重**，目前是二手转述，**必须核原文** | 投稿前 |
| **V13** | 我们自己数据的四格分布 | 若格 ③ 占比远低于 39%，需解释场景差异并**下调 C1 论证强度** | Phase 1.2.6 |

完整待核实清单见 `plan/implementation-plan.md` 附录 D。

---

## 8. 遇到冲突怎么办

文档之间若描述不一致，按 [`README.md`](README.md) 的「文档权威性约定」判断：

- 任务/Gate/指标 → `plan/implementation-plan.md`
- 架构/设计论证 → `plan/architecture.md`
- 代码位置行号 → `docs/code-anchors.md`
- 文献数字 → `research/evidence-table.md`

**发现冲突请修正权威来源，并同步引用它的文档**，不要在两处各留一份。
