# ⚠️ 此目录为纳入的第三方代码，请勿修改

| | |
|---|---|
| 上游 | https://github.com/Soul-AILab/SoulX-Duplug |
| 分支 | `main`（**推理服务**） |
| 提交 | `45bd23792e7c47ce7e07eb048b111fcb57bb67ca` (2026-08-24) |
| 许可 | Apache-2.0（见 `LICENSE`） |
| 角色 | **范式参考**：冻结前端 + 可训 Projector + LoRA |

**训练代码在另一棵树** `../SoulX-Duplug-training/`（`training-code` 分支）。

本目录要读的关键位置：

| 内容 | 位置 |
|---|---|
| `EncoderProjector` | `model/model.py:17-35` |
| WhisperVQ 冻结加载 | `model/model.py:53-58` |
| projector freeze 开关 | `model/model.py:60-71` |
| **complete/incomplete 判决** | `service/model.py:708-733` |

详见 [`../THIRD_PARTY.md`](../THIRD_PARTY.md)。
