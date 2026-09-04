# 参考代码位置索引

> **这是第三方代码位置的唯一权威来源。** 其他文档不重复列举，只链接到本文件。
>
> 三个目录**一律只读**（见 [`../../THIRD_PARTY.md`](../../THIRD_PARTY.md)）。
> 行号基于纳入时的提交，若与实际不符请以文件内搜索为准，并回来更正本文件。

---

## 1. X2-Turn（base，`@53d3b9a`）

**角色**：本项目的 base 权重来源，**全部冻结**。

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| `VoxtralMTP` 双头结构 | `X2-Turn/voxtral-realtime/src/voxtral_realtime/transformers/modeling.py:15-69,114-188` | 我们要包装的类；`lm_head` + `vad_lm_head` 双头 |
| 检查点加载 | 同文件 `:228-264` | `load_mtp_checkpoint()` 入口 |
| 冻结开关 `train_vad_head_only` | 同文件 | 现成的「只训话轮头」模式，正是我们要的 |
| `TURN_CLASS_IDS`（id 35–40） | `X2-Turn/.../inference.py:86` | 6 类话轮的 id 组；我们在旁边加 `CI_CLASS_IDS=(41,42)` |
| **hidden 读取偏移** | `X2-Turn/.../inference.py:165` | **`prefix_length + i − 1`，next-token 语义，写错则全局错位 80 ms** |
| 规则控制器注入点 | `X2-Turn/.../server.py:102, 115-120` | `AcousticVoiceGate` 的注入模式，Phase 5 照抄 |
| 安装脚本 | `X2-Turn/install.sh` | 环境配置首选 |
| 流式客户端 | `X2-Turn/turn-demo/stream_client.py` | 真机链路参考 |

## 2. SoulX-Duplug（范式 · 推理侧，`main @45bd237`）

**角色**：「冻结前端 + 可训 Projector + LoRA」范式的成功先例，**且是同任务**
（它有显式 complete/incomplete，X2-Turn 折叠成了 `turn_end`/`uncertain`）。

| 内容 | 位置 | 为什么要读 |
|---|---|---|
| `EncoderProjector` | `SoulX-Duplug/model/model.py:17-35` | Projector 结构参考（**但宽度不要照抄**，见 arch §3.9） |
| `token_samples = int(0.08*16000)` | `model/model.py:50` | 80 ms/token 的实现方式 |
| WhisperVQ 冻结加载 | `model/model.py:53-58` | `requires_grad=False` + `.eval()` 的写法 |
| projector freeze 开关 | `model/model.py:60-71` | 对应我们 S1 阶段 |
| `forward()` audio/text embedding 混合 | `model/model.py:145-166` | 融合位置的参考 |
| **complete/incomplete 判决** | **`service/model.py:708-733`** | **判别式读出的完整实现，最重要的参考** |
| 状态 token id（151675–151682） | `config/config.py:27-35` | 8 个状态 token 的注册方式 |
| per-token loss rate | `config/config.py:118-131` | 损失加权 |
| 远场 RMS 门控 | `service/model.py:252-256` | 兜底机制参考 |
| chunk / back / ahead size | `config/config.yaml:29-33` | 前视/回看配置 |

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

| | 值 | 出处 | 写错的后果 |
|---|---|---|---|
| **帧长** | **80 ms**（12.5 Hz） | X2-Turn `README_zh.md:112-116` | 视听对齐全错 |
| **hidden 偏移** | **`prefix_length + i − 1`** | `inference.py:165` | 全局 80 ms 系统性偏差，**且在指标上表现为「视觉略有帮助」，极难察觉** |

→ 因此 `tests/test_alignment.py`（脉冲响应单测）是**强制**的，见计划单 §2.7。
