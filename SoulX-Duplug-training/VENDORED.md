# ⚠️ 此目录为纳入的第三方代码，请勿修改

| | |
|---|---|
| 上游 | https://github.com/Soul-AILab/SoulX-Duplug |
| 分支 | **`training-code`**（**训练代码**） |
| 提交 | `928b06508ed2de1344208d06fb1f6fb2ebfb1df5` (2026-07-17) |
| 许可 | Apache-2.0（见 `LICENSE`） |
| 角色 | **Phase 3 写 Trainer 的直接参考** |

**这不是 `../SoulX-Duplug/` 的超集，而是另一棵树** ——
它删掉了推理服务，加入了训练工具。两者互补。

Phase 3 之前必读：

| 文件 | 用途 |
|---|---|
| **`finetune.py`** | 同范式同任务的训练入口 |
| **`example_data_fisher.jsonl`** | 官方数据格式样例，Phase 1.6 字段设计应对照它 |
| `launch.sh` | 启动参数参考 |
| `utils/ema/` | EMA 实现（三种） |
| `utils/epoch_shuffle.py` / `dynamic_train.py` | 数据调度 |
| `scripts/export_weights.py` | 权重导出 |

详见 [`../THIRD_PARTY.md`](../THIRD_PARTY.md)。
