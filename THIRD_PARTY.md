# 第三方代码溯源清单

> 本仓库**直接纳入**（vendor）了三份第三方代码树，不使用 git submodule。
> 目的：服务器上 `git clone` 一次即可拿到全部依赖，无需处理子模块与凭据。

---

## 一、纳入清单

| 目录 | 来源仓库 | 分支 | 提交 | 日期 | 许可 |
|---|---|---|---|---|---|
| `X2-Turn/` | [X-Square-Robot/X2-Turn](https://github.com/X-Square-Robot/X2-Turn)（经 `yutian929/X2-Turn` fork） | `main` | ⚠️ **见 §1.1**（fork SHA `53d3b9a`，**不存在于上游**） | 2026-08-31 | Apache-2.0 |
| `SoulX-Duplug/` | [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug)（经 `yutian929/SoulX-Duplug` fork） | `main` | `45bd23792e7c47ce7e07eb048b111fcb57bb67ca` | 2026-08-24 | Apache-2.0 |
| `SoulX-Duplug-training/` | [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug) | **`training-code`** | `928b06508ed2de1344208d06fb1f6fb2ebfb1df5` | 2026-07-17 | Apache-2.0 |

**三份 `LICENSE` 均已随代码保留**（Apache-2.0 的要求）。

> ⚠️ **论文 Reproducibility 段落不能只引 fork SHA** —— `X2-Turn/` 的 `53d3b9a` 在上游
> `X-Square-Robot/X2-Turn` 中**不存在**，读者按它什么都拉不到。**必须引 §1.1 的双 SHA。**

---

## 1.1 ★ `X2-Turn/` 的真实溯源（2026-09-14 核实后新增）

**核实方法**：`git clone` 上游 main 后 `git cat-file -t 53d3b9a…` → **object 不存在**；
`git log --all` 显示上游 2026-08-31 当天无提交，8 月最后一次提交为 8-28 的 `3609a0f`。

| 项 | 值 |
|---|---|
| **上游基点（可复现，论文引这个）** | `X-Square-Robot/X2-Turn` @ **`8992c7c189039cd88e50564c865219486ebbdd75`**（2026-09-10，"Point README paper links to arXiv v3"） |
| **本地树的直接来源** | `yutian929/X2-Turn` @ `53d3b9a68f0968a14e3ae70a5a1a678442312b7f`（**fork 独有提交，上游不可见**） |
| 核心模型代码状态 | ✅ **与上游 `8992c7c` 逐字节一致**（`transformers/modeling.py`、`transformers/inference.py`） |
| 论文写法 | 基座代码称「X2-Turn @ `8992c7c`」；若需说明本地树，附注 fork SHA 并列出 §1.2 的增删清单 |

**上游 8-31 之后的 11 个提交均未触及模型推理路径**（内容为 Qwen3TTS 对话 demo、README/News、
CI 运行时、arXiv v3 链接），因此 C2（模型改造）与 Phase 0 探针不受上游更新影响。

---

## 1.2 ★ 本地 `X2-Turn/` 相对上游 `8992c7c` 的增删清单

> 由 `diff -r --brief` 全量生成（2026-09-14）。**本地树不是纯净的上游快照**，
> 因此不能笼统声称「目录内容等于上游」；可声称的是**模型代码未修改**（见 §1.1 末行）。

### A. 已同步到上游 v4（2026-09-14 执行，理由见 §1.3）

| 文件 | 变更 |
|---|---|
| `voxtral-realtime/src/voxtral_realtime/config.py` | 规则控制器默认值 → 上游 v4 |
| `voxtral-realtime/src/voxtral_realtime/turn/controller.py` | `FrameTurnConfig` 默认值 → 上游 v4 |
| `voxtral-realtime/tests/test_config.py` | 补 8 条默认值断言 |

### B. fork 独有新增（上游历史中从未存在，`git log --all -- <file>` 为空）

| 路径 | 性质 | 我们的用法 |
|---|---|---|
| `install.sh`（237 行） | **fork 自制**安装脚本 | 备选；**官方路径是 `environments/*.yml`**（见 `server-setup.md` §2） |
| `turn-demo/stream_client.py` | **fork 自制**流式客户端 | 真机链路参考（Phase 5）；**不可称其为上游实现** |
| `turn-demo/requirements-client.txt` | 上述客户端的依赖 | — |
| `full-duplex-demo/cosyvoice_vllm_plugin/` | CosyVoice TTS 插件 | Phase 5 备选 TTS |
| `full-duplex-demo/voxtral_bridge/tts_server.py`<br>`…/tts_server_cosyvoice.py` | CosyVoice 服务端 | 同上 |
| `full-duplex-demo/dialogue_system/clients/tts_client.py` | CosyVoice 客户端 | 同上 |

### C. 上游有、本地缺（**未同步**，Phase 5 再评估）

| 路径 | 内容 |
|---|---|
| `full-duplex-demo/dialogue_system/clients/qwen3_tts_client.py` | 上游 v4 改用本地 Qwen3TTS-Streaming |
| `full-duplex-demo/scripts/start_qwen3tts_engine.sh` | Qwen3TTS 引擎启动 |
| `full-duplex-demo/tests/test_qwen3_tts_client.py` | 对应测试 |

> **暂不同步的理由**：TTS 只影响 demo 的出声环节，**不参与 complete/incomplete 判决**，
> 换链路属纯工程成本。Phase 5.1.1 搭真机时再决定走 CosyVoice 还是 Qwen3TTS。

### D. 其余文本性差异（不影响功能，保留 fork 版）

`README.md`(20 行) · `README_zh.md`(19) · `turn-demo/README.md`(44) ·
`environments/README.md`(16) · `environments/environment-dialogue.yml`(4) ·
`.github/workflows/ci.yml`(8) · `scripts/check_public_release.py`(1) ·
`scripts/check_release_language.py`(3) · `full-duplex-demo/` 下若干 demo 配置
（`.env.example`、`docker-compose.yml`、`pyproject.toml`、`app.py`、`llm_client.py`、
`run_app.sh`、`start_demo.sh`、`THIRD_PARTY_NOTICES.md`）—— 多为 CosyVoice↔Qwen3TTS 之差。

---

## 1.3 ★ 为什么 A 类三个文件必须同步（涉及 C3 的基线有效性）

上游 `01af067`「Make v4 dialogue experience reproducible」（2026-09-05）改了规则控制器默认值：

| 参数 | 旧（fork 原值） | **上游 v4（现已同步）** | 上游注释 |
|---|---|---|---|
| `silence_end_frames` | 3（240 ms） | **10（800 ms）** | *live mic: tolerate natural pauses* |
| `tail_min_frames` | 2 | **1** | *FDB-style fast tail confirmation* |
| `tail_max_frames` | 5（400 ms） | **1（80 ms）** | 同上 |
| `tail_stable_frames` | 2 | **1** | — |
| `short_tail_min_frames` | 4 | **3** | — |
| `short_tail_max_frames` | 7 | **5** | — |
| `acoustic_vad_max_hold_frames` | 8（640 ms veto） | **3（240 ms veto）** | — |

**为什么这不是普通的参数更新**：上游把静音阈值从 240 ms 提到 **800 ms**，注释明说是为了
「容忍自然停顿」—— **这正是本项目声称要解决的问题，上游先用纯规则调参做了一次部分缓解**。

后果两条，已同步到计划单：

1. **C3 的对照基线必须用 v4 默认值**。拿已废弃的 `silence_end_frames=3` 当基线是打稻草人，
   审稿人一句「你跟未调优的基线比」即可废掉主表。
2. **§4.4 Pareto 图须把 v4 默认值标为显式参照点** —— 它恰好把「纯调阈值能走多远」
   这个对照替我们做实了，是回应「为啥不直接调阈值」的最强证据。

---

## 二、为什么 SoulX 有两棵树

`training-code` **不是 `main` 的超集，而是另一棵树** —— 两者互补：

| | `SoulX-Duplug/`（main） | `SoulX-Duplug-training/`（training-code） |
|---|---|---|
| 定位 | **推理服务** | **训练代码** |
| 独有内容 | `server.py`、`service/model.py`（757 行）、`web/`、`web_client.py`、`utils/device_utils.py` | **`finetune.py`**、`launch.sh`、**`example_data_fisher.jsonl`**、`utils/ema/`、`utils/normalizers/`、`utils/epoch_shuffle.py`、`utils/dynamic_train.py`、`scripts/export_weights.py`、`utils/sparkvox/utils/scheduler.py` |
| 我们用它做什么 | 读**模型结构**与 **complete/incomplete 判决实现** | 读**训练循环**与**数据格式**，作为 Phase 3 的直接参考 |

> submodule 一次只能检出一个分支，这是改用 vendor 方式的一个额外收益。

### ★ `SoulX-Duplug-training/` 里最有价值的两个文件

| 文件 | 为什么重要 |
|---|---|
| `finetune.py` | 同范式同任务的训练入口，Phase 3.5.1 自写 Trainer 前必读 |
| `example_data_fisher.jsonl` | **官方数据格式样例**，Phase 1.6 设计字段时应对照它，降低格式返工风险 |

---

## 三、修改政策（重要）

**三个目录一律只读。** 不在其中直接改代码。

理由：
1. 我们的改造走**包装**而非 fork（见 `AV-SemanticVAD/plan/implementation-plan.md` 附录 C），
   所有新增代码在 `AV-SemanticVAD/avsvad/` 下，通过 `avsvad/upstream.py` 引用上游模块
2. 保持只读，才能随时用下节的命令**确认我们没有偏离上游**，
   也才能在论文里诚实声明「未修改基座代码」

**若确有必要修改**：
- 必须在 `AV-SemanticVAD/docs/` 下记录 patch 与理由
- 并在本文件追加「已修改文件」小节

### ★ 已修改 / 已同步文件登记（唯一登记处）

| 日期 | 目录 | 文件 | 操作 | 理由 |
|---|---|---|---|---|
| 2026-09-14 | `X2-Turn/` | `config.py`、`turn/controller.py`、`tests/test_config.py` | **同步至上游 `8992c7c`** | §1.3：C3 基线必须用上游 v4 默认值 |

> 上述三个文件是**向上游对齐**，不是我们的私有改动 ——
> 同步后与上游**逐字节一致**，仍可与上游 diff。
> **除此之外，三个第三方目录内没有我们的任何代码改动。**
>
> ⚠️ 但注意：`X2-Turn/` 里另有 **fork 自带**的独有文件（`install.sh`、
> `stream_client.py`、CosyVoice 链路等，见 §1.2 B），它们不是我们加的，
> 也不是上游的。**引用它们时必须注明「fork 自制」**，否则会误称为上游实现。

---

## 四、如何与上游对比 / 更新

已删除内层 `.git`，因此不能直接 `git pull`。需要对比或更新时：

```bash
# 1. 在仓库外克隆一份上游做对比
git clone https://github.com/X-Square-Robot/X2-Turn.git /tmp/x2turn-upstream
diff -r --brief X2-Turn /tmp/x2turn-upstream \
     -x '.git' -x '__pycache__' -x 'models' -x 'VENDORED.md'

# 2. ★ 先确认核心模型代码是否真的落后（多数情况下并没有）
D=voxtral-realtime/src/voxtral_realtime/transformers
for f in modeling.py inference.py; do
    diff -q X2-Turn/$D/$f /tmp/x2turn-upstream/$D/$f && echo "IDENTICAL $f"
done

# 3. ★ 顺手复验 code-anchors.md 的锚点行号仍成立
grep -n "TURN_CLASS_IDS = \|train_vad_head_only" /tmp/x2turn-upstream/$D/modeling.py
grep -n "prediction_index = prefix_length"       /tmp/x2turn-upstream/$D/inference.py
```

### ⚠️ 更新原则：选择性同步，不要整体覆盖

整体 `cp -r` 覆盖会**静默删掉 fork 独有文件**（§1.2 B 的 `install.sh`、
`stream_client.py`、CosyVoice 链路），而这三个目录已无内层 `.git`，
**删了不会有任何提示**。按下表分档：

| 档 | 内容 | 处理 |
|---|---|---|
| **A** | 影响我们实验有效性的（如规则控制器默认值） | **同步**，并登记到 §3 |
| **B** | 只影响 demo 外围的（如 TTS 链路） | 暂不动，到对应 Phase 再评估 |
| **C** | fork 独有件、README、CI | 保留 fork 版，在 §1.2 注明归属 |

更新后**务必回来改 §1.1 的上游基点 SHA 与 §1.2 的增删清单**，否则溯源失效。

**SoulX 两棵树对应的上游分支**：

```bash
git clone -b main          https://github.com/Soul-AILab/SoulX-Duplug.git /tmp/soulx-main
git clone -b training-code https://github.com/Soul-AILab/SoulX-Duplug.git /tmp/soulx-train
```

更新后**务必回来改本文件第一节的 SHA 与日期**，否则溯源失效。

---

## 五、不在库内的大文件

| 内容 | 位置 | 体积 | 获取方式 |
|---|---|---|---|
| SoulX 权重 | `SoulX-Duplug/pretrained_models/` | **7.2 GB** | 见 `AV-SemanticVAD/docs/server-setup.md` §3 |
| X2-Turn 权重 | `X2-Turn/models/` | ≈8 GB | 同上 |

两者均已在根 `.gitignore` 中排除。**克隆本仓库不会带上权重，需按部署文档单独下载。**
