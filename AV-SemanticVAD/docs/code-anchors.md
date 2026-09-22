# 参考代码位置索引

> **这是第三方代码位置的唯一权威来源。** 其他文档不重复列举，只链接到本文件。
>
> 三个目录**一律只读**（见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md)）。
> 行号基于纳入时的提交，若与实际不符请以文件内搜索为准，并回来更正本文件。

---

## 1. X2-Turn（base，上游 `@8992c7c`）

**角色**：本项目的骨干权重来源。⚠️ **不再全部冻结** —— v6 走 S1 冻骨干训新模块 → S2 LoRA r=32（见 [`../plan/architecture.md`](../plan/architecture.md) §4）。

> **行号已在上游 `@8992c7c` 上复验**（2026-09-14）—— `modeling.py` 与 `inference.py`
> 与上游**逐字节一致**，下表全部锚点仍成立。
> 溯源与 fork 差异见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §1.1–1.2。
> 两个文件的完整路径前缀均为
> `X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/`，下表简写为 `…/`。

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| **`TURN_CLASS_IDS = (35,…,40)` 定义处** | **`…/modeling.py:15`** | 6 类话轮的 id 组。⚠️ v6 **不再占用空闲 id**，改为判别式小头，并从第 35/37/38 行**热启动**（`architecture.md` §3.1） |
| `TURN_CLASS_NAMES` | `…/modeling.py:16` | 6 类的名字顺序 |
| **`TURN_CLASS_IDS` 的 softmax 使用处** | **`…/inference.py:86`**（在 `_predict_turn()`，定义于 `:83`） | **只对这 6 个 id 做 softmax** —— v6 的三态判决改走独立 `Linear(3072,3)` 头，此处只作判别式读出写法的参考 |
| `_predict_turn()` 的调用点 | `…/inference.py:168` | 逐帧循环内 |
| `VoxtralMTP` 类与双头 | `…/modeling.py`（共 264 行，全文可读） | 我们要包装的类；`base_model.lm_head` + `self.vad_lm_head` |
| **冻结开关 `train_vad_head_only`** | **`…/modeling.py:125-145`** | **现成的「骨干 `no_grad` + 只训 vad 头」实现，正是我们 S1 要的模式** |
| 正常前向（不冻结）路径 | `…/modeling.py:160-165` | 与上者对照 |
| **逐帧读出的索引偏移** | **`…/inference.py:165`**：`prediction_index = prefix_length + frame_index - 1` | **next-token 语义。写错则全局错位 80 ms** |
| `prefix_length` 定义 | `…/inference.py:133` | `= input_ids.shape[1]` |
| `frame_count` 计算 | `…/inference.py:135` | `= num_audio_tokens - prefix_length` |
| 取 `vad_logits` 并逐帧判决 | `…/inference.py:162-168` | 我们的 ci 判决要挂在同一循环里 |
| 安装脚本 | `X2-Turn/install.sh` | ⚠️ **fork 自制，非上游**（见下方警告）；官方路径见 `server-setup.md` §2.1 |
| 流式客户端 | `X2-Turn/turn-demo/stream_client.py` | 真机链路参考 —— ⚠️ **fork 自制，非上游** |

> ⚠️ **归属修正（2026-09-14 核实）**：`install.sh`、`turn-demo/stream_client.py`、
> `turn-demo/requirements-client.txt` 在上游 `X-Square-Robot/X2-Turn` 的历史中
> **从未存在**（`git log --all -- <file>` 返回空），它们来自 `yutian929` 的 fork。
> **可以用，但论文与文档中不得称其为上游实现。**
> 完整增删清单见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §1.2 B。

> ⚠️ **`hidden_states` 只在 `modeling.py` 内部出现**（`:128,139,160,163`），
> `inference.py` 拿到的已经是 `output.vad_logits`。
> 若要抽 hidden 做**探针 A**，需在 `modeling.py` 的 forward 返回值上取，
> 或对 `base_model.model` 单独前向 —— 不要指望 `inference.py` 能直接给你 hidden。

> ### ★ 规则控制器与外部信号注入点（Phase 5.1.4 要照抄的模式，**2026-09-14 已定位**）
>
> 路径前缀 `X2-Turn/voxtral-realtime/src/voxtral_realtime/`，下表简写 `§/`。
>
> | 内容 | 位置 | 为什么要读 |
> |---|---|---|
> | `class AcousticVoiceGate` 定义 | `§/audio.py:8` | 外部信号源的实现形态 |
> | import 与实例化 | `§/server.py:15` / `server.py:31` | 挂载方式 |
> | **★ 每帧求值** | **`§/server.py:102`**：`acoustic_active = self.acoustic_gate.update(pcm)` | **我们的 `p(continue)` 应在此并列产出** |
> | **★ 注入控制器** | **`§/server.py:115-120`**（`self.controller.on_frame(raw_turn, asr_text, frame_index, acoustic_active)`，`acoustic_active` 在 `:119`） | **全仓库唯一注入点，ci 判决照此接入** |
> | trace 落盘字段 | `§/server.py:135-137, 162-163` | 加 ci 字段时照此扩展，便于离线复盘 |
> | 控制器侧接收 | `§/turn/controller.py:94-104`（`on_frame` 签名 + `self.st.acoustic_active` 写入） | 形参顺序与状态写入 |
> | **veto 逻辑与上限** | **`§/turn/controller.py:110-118`** | `acoustic_active and acoustic_hold_run < cfg.acoustic_vad_max_hold_frames` → `tail_cancel_vad` —— 这就是「外部信号只能延缓、不能无限阻断」的 fail-safe 写法，**降级保证（纪律 3）可直接复用** |
>
> **★ `FrameTurnConfig` 默认值已于 2026-09-14 同步到上游 v4**
> （`silence_end_frames` 3→**10**、`tail_max_frames` 5→**1**、
> `acoustic_vad_max_hold_frames` 8→**3**，定义在 `§/turn/controller.py:44-56`，
> `§/config.py` 同步）。
> **C3 的规则控制器基线必须用这组新值** —— 理由见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) §1.3。

## 2. SoulX-Duplug（范式 · 推理侧，`main @45bd237`）

**角色**：「冻结前端 + 可训 Projector + LoRA」范式的成功先例，**且是同任务**
（它有显式 complete/incomplete，X2-Turn 折叠成了 `turn_end`/`uncertain`）。

> **行号已在 `main @45bd237` 上核实**（2026-09-04）。

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| `class EncoderProjector` | `SoulX-Duplug/model/model.py:17` | Projector 结构参考（**但宽度不要照抄**，见 arch §3.9） |
| `token_samples = int(0.08 * sampling_rate)` | `model/model.py:50` | **80 ms/token 的实现方式** |
| WhisperVQ 冻结（`requires_grad=False`） | `model/model.py:57` | 冻结前端的写法 |
| projector freeze 开关 | `model/model.py:68` | 对应我们 S1 阶段 |
| `forward()` audio/text embedding 混合 | `model/model.py:145-166` | 融合位置的参考 |
| **★ complete/incomplete 判决** | **`service/model.py:699-733`** | **判别式读出的完整实现，最重要的参考** |
| ↳ 取 logits | `service/model.py:699` | `logits = outputs.logits[0]` |
| ↳ 两个 logit 取值 | `service/model.py:711,714` | `complete_logit` / `incomplete_logit` |
| ↳ **判决式** | **`service/model.py:728`** | `if complete_logit > incomplete_logit + complete_bias` |
| ↳ `complete_bias` 的作用 | `service/model.py:717-720`（含注释） | **免重训移动决策边界**，我们的推理阈值可照此设计 |
| 状态 token 编码 | `service/model.py:106,109` | `<\|user_complete\|>` / `<\|user_incomplete\|>` |
| 状态分派 | `service/model.py:344` 起 | 各状态的处理分支 |
| 状态 token id（151675–151682） | `config/config.py:27-35` | 8 个状态 token 的注册方式 |
| per-token loss rate | `config/config.py:118-131` | 损失加权 |
| 远场 RMS 门控 | `service/model.py:252-256` | 兜底机制参考 |
| chunk / back / ahead size | `config/config.yaml:29-33` | 前视/回看配置 |

> **★ `complete_bias` 是个现成的好设计**：它在不重训的前提下平移决策边界
> （代码内注释即如此说明）。我们 §4.4 的固定阈值 Pareto 扫描可以直接借用这个思路。

## 3. SoulX-Duplug-training（范式 · 训练侧，`training-code @928b065`）

**角色**：**Phase 3 写 Trainer 的直接参考**。

> ⚠️ 这不是 §2 的超集，是**另一棵树**：上游把推理和训练放在不同分支，
> 该分支删掉了 `service/model.py` 等推理代码，加入了训练工具。两者互补。

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| **数据格式样例** | **`SoulX-Duplug-training/example_data_fisher.jsonl`** | **官方 jsonl 格式；Phase 1.6 定字段前必读，否则流水线做完才发现格式不符** |
| **训练入口** | **`SoulX-Duplug-training/finetune.py`** | 同范式同任务的训练循环 |
| 启动参数 | `launch.sh` | lr / batch / 阶段划分的实际取值 |
| EMA 实现（三种） | `utils/ema/{ema,ema_lightning,ema_nemo}.py` | 需要时直接借 |
| 变长组批 | `utils/epoch_shuffle.py`、`utils/dynamic_train.py` | 我们的音频也是变长（V5） |
| 学习率调度 | `utils/sparkvox/utils/scheduler.py` | — |
| 权重导出 | `scripts/export_weights.py` | Phase 3 出 checkpoint 时参考 |
| 文本归一化 | `utils/normalizers/`（含 `english.json` 1741 行） | 中英双语转写归一化可复用 |

### ★ Phase 3.5.1 的阅读顺序

```
1. example_data_fisher.jsonl   → 定 §1.6 数据字段（避免格式返工）
2. finetune.py                 → 定训练循环骨架
3. launch.sh                   → 抄超参起点
4. 才动手写 avsvad/train/trainer.py
```

---

## 4. 两个必须记住的数字

| | 值 | 出处（已核实） | 写错的后果 |
|---|---|---|---|
| **帧长** | **80 ms**（12.5 Hz） | `SoulX-Duplug/model/model.py:50`；X2-Turn `README_zh.md:112-116` | 视听对齐全错 |
| **逐帧读出索引** | **`prefix_length + frame_index − 1`** | `X2-Turn/…/inference.py:165` | 全局 80 ms 系统性偏差，**且在指标上表现为「视觉略有帮助」，极难察觉** |

→ 因此 `tests/test_alignment.py`（脉冲响应单测）是**强制**的，见计划单 §2.7。

---

## 5. 核实这些行号的方法

行号会随上游更新失效。**若发现不符，用下面的命令重新定位，然后回来更正本文件**：

```bash
D=X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers
grep -n "TURN_CLASS_IDS"       $D/modeling.py
grep -n "train_vad_head_only"  $D/modeling.py
grep -n "prefix_length"        $D/inference.py

S=X2-Turn/voxtral-realtime/src/voxtral_realtime
grep -n "class AcousticVoiceGate"              $S/audio.py
grep -n "acoustic_gate\|on_frame"              $S/server.py
grep -n "acoustic_vad_max_hold_frames"         $S/turn/controller.py

grep -n "class EncoderProjector\|token_samples" SoulX-Duplug/model/model.py
grep -n "complete_logit\|complete_bias"         SoulX-Duplug/service/model.py
```

**不要凭记忆写行号。** 上表所有行号均在纳入的提交上逐条 `grep` 验证过，
并已于 2026-09-14 在上游 `@8992c7c` 上复验（`modeling.py` / `inference.py` 与上游一致）。
