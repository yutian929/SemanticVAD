# 第三方代码溯源清单

> 本仓库**直接纳入**（vendor）了三份第三方代码树，不使用 git submodule。
> 目的：服务器上 `git clone` 一次即可拿到全部依赖，无需处理子模块与凭据。

---

## 一、纳入清单

| 目录 | 来源仓库 | 分支 | 提交 | 日期 | 许可 |
|---|---|---|---|---|---|
| `X2-Turn/` | [X-Square-Robot/X2-Turn](https://github.com/X-Square-Robot/X2-Turn)（经 `yutian929/X2-Turn` fork） | `main` | `53d3b9a68f0968a14e3ae70a5a1a678442312b7f` | 2026-08-31 | Apache-2.0 |
| `SoulX-Duplug/` | [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug)（经 `yutian929/SoulX-Duplug` fork） | `main` | `45bd23792e7c47ce7e07eb048b111fcb57bb67ca` | 2026-08-24 | Apache-2.0 |
| `SoulX-Duplug-training/` | [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug) | **`training-code`** | `928b06508ed2de1344208d06fb1f6fb2ebfb1df5` | 2026-07-17 | Apache-2.0 |

**三份 `LICENSE` 均已随代码保留**（Apache-2.0 的要求）。
论文的 Reproducibility 段落应引用上表的提交 SHA。

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

---

## 四、如何与上游对比 / 更新

已删除内层 `.git`，因此不能直接 `git pull`。需要对比或更新时：

```bash
# 1. 在仓库外克隆一份上游做对比
git clone https://github.com/X-Square-Robot/X2-Turn.git /tmp/x2turn-upstream
diff -r --brief X2-Turn /tmp/x2turn-upstream \
     -x '.git' -x '__pycache__' -x 'models' -x 'VENDORED.md'

# 2. 确认差异后，用上游内容替换（保留本地权重目录）
#    ⚠️ 替换前先看清 diff，不要盲目覆盖
```

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
