# SoulX-Duplug macOS 本地部署与耗时测试

本文档面向在一台**全新的 Apple Silicon MacBook** 上从零部署模型并测试推理耗时。

> 前置要求：Apple Silicon（arm64，可用 `uname -m` 确认）、内存建议 ≥ 16GB。

## 完整操作流程

### 1. 安装 Homebrew（一次性）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Apple Silicon 上 Homebrew 装在 `/opt/homebrew`，手动兜底加入 PATH：

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 2. 安装 uv（一次性，可选）

`install_mac.sh` 已内置自动检测安装 uv 的逻辑，此步可跳过；提前装好更稳：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

### 3. 拉取项目

```bash
git clone https://github.com/yutian929/SoulX-Duplug.git
cd SoulX-Duplug
```

> 若你 fork 了仓库，请替换为你自己的地址。国内网络卡顿时可加镜像前缀：`https://ghproxy.com/https://github.com/...`。

### 4. 一键安装环境 + 依赖 + 模型

```bash
bash install_mac.sh --all
```

该脚本依次完成：

- 安装 `ffmpeg` / `sox`（通过 Homebrew）
- 创建 `.venv-mac` 虚拟环境并安装全部依赖
- 下载模型权重（约 7.3GB）
- 预下载 ASR 模型（约 936MB）

**主要耗时在模型下载，取决于网速。**

### 5. 测试耗时

```bash
bash benchmark_mac.sh
```

依次运行三套基准：

1. `benchmark_local.py` —— 单 chunk 延迟 + RTF（是否实时）
2. `profile_breakdown.py` —— 逐段耗时拆解，定位瓶颈（LLM / ASR / GLM VQ）
3. `benchmark_asr_impact.py` —— ASR 延迟对 RTF 的影响

结果同时打印到屏幕，并保存到 `benchmark_results/benchmark_时间戳.log`。

## 常用命令速查

| 命令 | 作用 |
|------|------|
| `bash benchmark_mac.sh --quick` | 快速模式（20 个 chunk，先验证能跑通） |
| `bash benchmark_mac.sh --device cpu` | 强制 CPU 跑（对比 MPS） |
| `bash benchmark_mac.sh --num-chunks 200` | 自定义统计 chunk 数 |
| `bash benchmark_mac.sh --skip-profile` | 只测延迟，跳过耗时拆解 |
| `bash benchmark_mac.sh --skip-asr` | 跳过 ASR 影响测试 |
| `bash benchmark_mac.sh -n myenv` | 自定义虚拟环境目录名（默认 `.venv-mac`） |

## 关键指标与预期

- **RTF（Real-Time Factor）** = 单 chunk 耗时 / chunk 时长，**RTF < 1 表示能实时**。
- 模型首次加载约需 2 分钟（MPS 需编译 Metal kernel）。
- M3/16GB 实测：MPS 约 1.5–1.8，CPU 约 2.9–3.0，均未达实时；瓶颈在 LLM 前向（约 50%），且为 memory-bound，**内存带宽比芯片算力更关键**。
- 将 `config/config.yaml` 的 `chunk_size` 由 `2560` 调大到 `5120`（320ms）可显著降低 RTF，但判定粒度变粗。

## 注意事项

- **不要**设置 `HF_ENDPOINT=hf-mirror.com`：模型仓库在镜像上不存在，`install_mac.sh` 已自动规避该问题。
- 首次运行务必保留 warmup（默认丢弃前 5 个 chunk），否则 MPS 首次编译会导致前几个 chunk 慢数十倍。
- 若要启动服务自测功能，可运行 `bash run.sh`，并用 `test.py` 自检。
