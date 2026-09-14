# ⚠️ 此目录为纳入的第三方代码，请勿修改

| | |
|---|---|
| 上游 | https://github.com/X-Square-Robot/X2-Turn |
| 分支 | `main` |
| **上游基点（论文引这个）** | **`8992c7c189039cd88e50564c865219486ebbdd75`** (2026-09-10) |
| 本地树直接来源 | `yutian929/X2-Turn` @ `53d3b9a68f0968a14e3ae70a5a1a678442312b7f` — ⚠️ **fork 独有提交，上游不存在** |
| 许可 | Apache-2.0（见 `LICENSE`） |
| 角色 | **本项目的 base 权重来源**，全部冻结 |

**核心模型代码**（`voxtral-realtime/src/voxtral_realtime/transformers/{modeling,inference}.py`）
✅ 与上游 `8992c7c` **逐字节一致**。

⚠️ **但本目录不是纯净的上游快照**：

- `install.sh`、`turn-demo/stream_client.py`、`turn-demo/requirements-client.txt`、
  `full-duplex-demo/` 下的 CosyVoice 链路 → **fork 自制，上游从未有过**
- `config.py`、`turn/controller.py`、`tests/test_config.py`
  → 已于 2026-09-14 **同步至上游 v4**（理由：C3 基线有效性）
- 上游 v4 的 Qwen3TTS 链路 → **未同步**（Phase 5 再评估）

我们的改造代码在 `../AV-SemanticVAD/avsvad/` 下，通过包装引用本目录，**不 fork 源码树**。

完整增删清单与更新方法详见 [`../THIRD_PARTY.md`](../THIRD_PARTY.md) §1.1–1.3、§4。
