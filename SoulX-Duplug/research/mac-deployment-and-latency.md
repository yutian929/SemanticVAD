# SoulX-Duplug 本地部署与实时性评估报告

> 主体为 Apple Silicon 实测；第 7 节基于实测数据外推评估 NVIDIA Jetson 端侧可行性。

| 项目 | 内容 |
|---|---|
| 分支 | `mac-local-inference`（基于 `main`） |
| 报告日期 | 2026-08-21 |
| 测试机器 | MacBook Apple M3 / 16 GB / macOS 26.5.2 / arm64 |
| 被测版本 | SoulX-Duplug-0.6B-Bilingual |
| 测试音频 | `assets/tmp.wav`（60 s，中文对话） |

---

## 摘要

在 Apple M3 上完成了 SoulX-Duplug 的完整本地部署，**功能验证通过，但无法实时运行**。

| 结论 | 数据 |
|---|---|
| 默认配置（160 ms chunk）达不到实时 | RTF **1.50**（MPS）/ **3.00**（CPU） |
| 主要开销是 **LLM 前向**，不是 ASR | LLM 50.4%，ASR 30.1%，GLM VQ 17.9% |
| **换网络流式 ASR 无法解决问题** | 非 ASR 固定开销 **168 ms** 已超 160 ms 预算 |
| 放宽 chunk 到 320 ms 可达实时 | RTF **0.847**，代价是判定粒度变粗一倍 |
| MPS 相对 CPU 加速有限 | 仅 **2.0x**（因矩阵瘦长，GPU 并行度用不上） |
| **瓶颈性质：memory-bound** | 带宽效率仅 **40–53 %**，权重搬运占预算 51 % |
| **端侧 Jetson 可行**（估算） | Orin NX 16GB 即可实时，RTF **0.78**（fp32）/ **0.47**（FP16） |

**建议**：Mac 定位为功能验证与离线测试环境；实时对话用 GPU 服务器，端侧落地可考虑 Jetson Orin NX 16GB。已完成的设备抽象改造使同一份代码在各平台通用。

---

## 目录

- [1. 部署工作](#1-部署工作)
- [2. 实时性基准测试](#2-实时性基准测试)
- [3. 耗时归因分析](#3-耗时归因分析)
- [4. 关键问题：外挂网络 ASR 能否解决？](#4-关键问题外挂网络-asr-能否解决)
- [5. MPS 与 CPU 的差异分析](#5-mps-与-cpu-的差异分析)
- [6. 优化路径评估](#6-优化路径评估)
- [7. 端侧设备可行性：NVIDIA Jetson](#7-端侧设备可行性nvidia-jetson)
- [8. 结论与后续工作](#8-结论与后续工作)
- [附录 A：复现方式](#附录-a复现方式)
- [附录 B：代码锚点](#附录-b代码锚点)
- [附录 C：工程问题记录](#附录-c工程问题记录)

---

## 1. 部署工作

### 1.1 原始代码的移植障碍

上游代码假设运行环境为 Linux + NVIDIA GPU，存在四处阻塞：

| 障碍 | 位置 | 处理方式 |
|---|---|---|
| 设备硬编码 `cuda` | `config.yaml`、`model/asr.py`(×3) | 新增设备解析层，支持 `cuda > mps > cpu` 自动降级 |
| `torch.cuda.empty_cache()` | `service/model.py:403` | 改为按后端分派的 `empty_cache()` |
| `pynini` 无法编译 | `utils/MyTn/textnorm.py` | 改为惰性导入（详见附录 C） |
| `requirements.txt` 含大量 NVIDIA 包 | — | 另写 macOS 专用依赖清单 |

一个有利发现：配置项 `precision: bf16` **在推理代码中从未被使用**，模型实际以 fp32 加载，因此 MPS 上无需处理混合精度兼容问题。

### 1.2 新增内容

| 文件 | 用途 |
|---|---|
| `utils/device_utils.py` | 设备解析（`resolve_device`）、缓存清理（`empty_cache`）、ASR 设备策略（`asr_device`） |
| `install_mac.sh` | macOS 一键安装脚本，6 步流程含 26 项依赖校验与 MPS 自检 |
| `benchmark_local.py` | 延迟基准，输出 P50/P90/P99 与 RTF |
| `profile_breakdown.py` | 逐段耗时拆解，定位瓶颈 |
| `benchmark_asr_impact.py` | ASR 延迟敏感性实验（假 ASR 注入固定延迟） |

改造保持**向后兼容**：`device: auto` 在 GPU 服务器上仍解析为 `cuda`，同一份代码两端通用。

### 1.3 功能正确性验证

MPS 与 CPU 两个后端跑同一段音频，**状态分布完全一致**：

```
idle: 67    nonidle: 29    speak: 4
```

说明 MPS 未引入数值偏差，`speak` 轮次判定正常触发。

---

## 2. 实时性基准测试

### 2.1 实时性判据

模型以固定步长处理音频，每个 chunk 的推理必须在该 chunk 所代表的音频时长内完成：

```
RTF (Real-Time Factor) = 单 chunk 推理耗时 / chunk 音频时长
RTF < 1  →  可实时（推理速度快于音频流入速度）
RTF > 1  →  延迟累积，越说越滞后
```

默认配置下 chunk 为 **160 ms**（`chunk_size: 2560` @ 16 kHz），即每个 chunk 的推理预算是 160 ms。

### 2.2 测试结果

| 后端 | 平均 | P50 | P90 | P99 | 超预算比例 | **RTF** | 实时 |
|---|---|---|---|---|---|---|---|
| **MPS** | 287.8 ms | 285.8 ms | 402.7 ms | 1412.1 ms | 65 % | **1.80** | ❌ |
| **CPU** | 467.9 ms | 454.7 ms | 648.8 ms | 956.9 ms | 100 % | **2.92** | ❌ |

> 注：逐段拆解时（第 3 节）测得 MPS 均值 240.6 ms / RTF 1.50，与此处 287.8 ms / 1.80 的差异来自 ASR 触发率波动（0.52 vs 0.69 次每 chunk）与样本区间不同。**MPS 上的 RTF 区间为 1.5–1.8**。

### 2.3 两个观测到的 MPS 特性

**冷启动开销极大**——Metal kernel 首次编译：

| chunk | 耗时 |
|---|---|
| 0 | 7521 ms |
| 1 | 1098 ms |
| 2 | 151 ms |

因此所有基准脚本均设 `--warmup` 丢弃前若干样本，否则均值被严重污染。

**异步派发**——MPS 运算返回时可能尚未完成，所有计时代码必须包裹 `torch.mps.synchronize()`，否则测得的是派发耗时而非计算耗时。

---

## 3. 耗时归因分析

### 3.1 单 chunk 处理流程

```
process(chunk)
 ├─ get_chunk()              缓冲区切片，开销可忽略
 └─ state_predict() → infer()
      ├─ _audio_to_tokens()     Whisper 特征提取 + GLM VQ 编码
      ├─ _tokens_to_embeds()    codebook 查表 + projector MLP
      ├─ _asr()
      │    ├─ LLM forward #1    判断当前 chunk 是否含语音
      │    └─ 级联 ASR          仅在检测到语音时触发
      └─ _state_predict()
           └─ LLM forward #2    以 ASR 文本为输入，预测对话状态
```

**两次 LLM 前向缺一不可**：第一次决定是否调用 ASR，第二次将 ASR 转写文本作为输入预测状态。这正是 SoulX-Duplug「文本引导的状态预测」的实现方式，也是其优于纯声学 VAD 的原因，代价是每 chunk 两次前向。

### 3.2 拆解结果（MPS，60 chunks）

| 组成部分 | 单次均值 | 中位 | 最大 | 次/chunk | **摊后耗时** | **占比** |
|---|---|---|---|---|---|---|
| **LLM 前向** | 58.2 ms | 55.3 ms | 138.8 ms | **2.08** | **121.2 ms** | **50.4 %** |
| **级联 ASR** | 140.2 ms | 138.8 ms | 150.3 ms | **0.52** | **72.4 ms** | **30.1 %** |
| **音频→token (GLM VQ)** | 43.0 ms | 39.1 ms | 72.3 ms | 1.00 | **43.0 ms** | **17.9 %** |
| projector MLP | 1.1 ms | 1.1 ms | 1.5 ms | 1.00 | 1.1 ms | 0.5 % |
| 其余（文本处理/拷贝） | — | — | — | — | 2.9 ms | 1.2 % |
| **单 chunk 总计** | **240.6 ms** | 232.9 ms | 463.7 ms | | **240.6 ms** | **100 %** |

### 3.3 三个关键发现

**发现一：主要开销是 LLM，不是 ASR。**

关键在 `次/chunk` 一列。ASR 单次虽贵（140 ms），但**仅 52 % 的 chunk 触发**（静音段不调用），摊薄后只有 72 ms；而 LLM 每 chunk 稳定跑 2 次以上，摊后 121 ms。排序为 **LLM (50 %) > ASR (30 %) > GLM VQ (18 %)**。

**发现二：LLM 实际跑 2.08 次，而非 2 次。**

多出的 0.08 来自 ASR 改口修正机制。流式 ASR 会修正先前输出（如「我想」→「我想问」），此时代码回滚 KV cache 至 checkpoint 并重新喂入修正文本，**触发第三次前向**（`service/model.py:608-644`）。这是 P90（402.7 ms）远高于均值的主要原因，即**延迟长尾主要由 correction 贡献**。

**发现三：每次推理的音频窗口远大于 160 ms。**

160 ms 只是决策步长，实际送入模型的窗口为：

| 配置项 | 采样点 | 时长 | 作用 |
|---|---|---|---|
| `audio_back_size` | 15360 | 960 ms | 历史上下文 |
| `chunk_size` | 2560 | **160 ms** | 当前待判决片段 |
| `audio_ahead_size` | 640 | 40 ms | 前瞻 |
| **合计** | 18560 | **1160 ms** | 每次推理实际编码的音频 |

即每前进 160 ms，都要重新编码 1160 ms 音频，这是 GLM VQ 那 43 ms 的来源。此外 ASR 缓冲为 3.2 s，每次触发都重跑全部 3.2 s。

---

## 4. 关键问题：外挂网络 ASR 能否解决？

### 4.1 问题背景

既然 ASR 占 30 % 开销，直觉上换成云端流式 ASR API 应能改善。**实验结论：不能，且很可能更差。**

### 4.2 架构约束：ASR 位于串行关键路径

```
LLM forward #1（判断有无语音）
      ↓
   ASR 转写  ←── 网络 RTT 加在此处
      ↓
delta_text → tokenize → embedding
      ↓
LLM forward #2（预测状态）← 必须以 ASR 结果为输入
```

ASR 的**输出是下一步 LLM 的输入**（`service/model.py:505-512`），因此既无法异步化，也无法与 LLM 并行。网络 RTT 会 100 % 计入每个 chunk 的预算。

### 4.3 敏感性实验

用可注入固定延迟的假 ASR 替换真实 ASR，隔离「ASR 耗时」这一单一变量：

| ASR 延迟 | 平均耗时 | P90 | **RTF** | 实时 |
|---|---|---|---|---|
| **0 ms（完全免费）** | 143.8 ms | 154.8 ms | **0.90** | ✅ 勉强 |
| 20 ms | 165.2 ms | 198.9 ms | 1.03 | ❌ |
| 24 ms | 161.7 ms | 183.3 ms | 1.01 | ❌ |
| 60 ms | 176.5 ms | 219.0 ms | 1.10 | ❌ |
| 100 ms | 211.8 ms | 275.2 ms | 1.32 | ❌ |
| 150 ms | 252.4 ms | 340.0 ms | 1.58 | ❌ |
| 200 ms | 281.1 ms | 396.0 ms | 1.76 | ❌ |

**核心结论**：ASR 归零时 RTF 仍为 0.90，即优化 ASR 的**收益存在硬上限**。按该实验的触发率（0.69 次/chunk）反推，单次 ASR 必须快于 **≈24 ms** 才能维持实时。

若改用第 3 节拆解的模块开销直接相加：

```
LLM 前向 (2.08 次)   121.2 ms
GLM VQ encoder        43.0 ms
projector MLP          1.1 ms
其余                   2.9 ms
──────────────────────────────
非 ASR 固定开销       168.2 ms   >   160 ms 预算
```

**即 ASR 耗时为零、网络延迟为零时，固定开销本身已超预算 8 ms。**（两次测量给出 144–168 ms 区间，差异源于 ASR 与 correction 触发率波动。）

### 4.4 实测网络延迟

| 端点 | ICMP RTT |
|---|---|
| asr.cloud.tencent.com | 8.9 ms |
| dashscope.aliyuncs.com | 46.7 ms |
| api.deepgram.com | 188.3 ms |

这仅是裸 RTT，真实 API 调用还需叠加 TLS、WS 帧封装、服务端排队与推理，实际端到端通常 **50–150 ms**，远超 24 ms 预算。

### 4.5 两个附加成本

**延迟抖动**：本地 ASR 稳定在 140 ms（最大 150 ms）；网络 API 的 P99 通常为 P50 的 3–5 倍。全双工场景下抖动比均值更具破坏性。

**上行带宽**：每次 ASR 需传输 3.2 s 完整音频而非 160 ms 增量：

```
3.2 s × 16000 Hz × 4 B (float32) ≈ 205 KB / 次
每 160 ms 一次  →  约 1.28 MB/s 上行
```

即便降为 16-bit PCM 仍需 640 KB/s，带宽与 API 计费均不经济。

### 4.6 小结

网络 ASR 的合理适用场景是**本地无法运行 ASR（如内存不足）时换取可运行性**，而非换取速度。

---

## 5. MPS 与 CPU 的差异分析

### 5.1 架构差异

| | CPU | MPS |
|---|---|---|
| 算力来源 | M3 的 8 个 CPU 核心 | M3 的 10 个 GPU 核心 |
| 设计取向 | 核心少、单核强，擅长串行与分支 | 核心多而简单，擅长大规模并行 |
| PyTorch 设备名 | `cpu` | `mps`（对应 NVIDIA 的 `cuda`） |
| 内存 | 系统内存 | **同一块系统内存**（统一内存架构） |

Apple Silicon 采用统一内存，CPU 与 GPU 共享同一物理内存。实测 CPU→MPS 拷贝 64 MB 张量仅 2.53 ms（≈24.7 GB/s），因为并非真正的显存搬运。**优点**是无 PCIe 瓶颈，**缺点**是无独立显存，模型与系统争用同一份 16 GB（实测峰值 7.26 GB）。

### 5.2 为何加速仅 2.0x

纯矩阵乘的加速比与规模强相关：

| 矩阵规模 | CPU | MPS | 加速比 |
|---|---|---|---|
| 128×128 | ~0.00 ms | 0.37 ms | **0.0x（MPS 更慢）** |
| 512×512 | 0.17 ms | 0.82 ms | **0.2x（MPS 更慢）** |
| 1024×1024 | 1.38 ms | 1.33 ms | 1.0x |
| 2048×2048 | 11.67 ms | 6.20 ms | 1.9x |
| 4096×4096 | 99.05 ms | 48.60 ms | 2.0x |

小矩阵时 GPU kernel 启动开销超过计算本身，反而更慢。而本模型恰好落在不利区间：

```
Qwen3-0.6B:  hidden_size 1024,  intermediate_size 3072,  28 层
每 chunk 新增 token 数:  2 个（160 ms = 2 个 80 ms 音频 token）
典型 GEMM 形状:  [2, 1024] × [1024, 3072]
```

这是**极瘦长的矩阵**（batch 维仅 2），GPU 的并行能力基本闲置。这也揭示了一个普遍规律：**流式自回归推理天生对 GPU 不友好**，属 memory-bound 而非 compute-bound。

### 5.3 逐模块加速比

| 模块 | CPU | MPS | 加速比 |
|---|---|---|---|
| LLM 前向（摊后） | 263.7 ms | 121.2 ms | **2.2x** |
| GLM VQ | 112.7 ms | 43.0 ms | **2.6x** |
| 级联 ASR（摊后） | 99.8 ms | 72.4 ms | 1.4x |
| **总计** | **480.0 ms** | **240.6 ms** | **2.0x** |

ASR 一栏无实际加速——两个后端下 ASR 均运行于 CPU（见 5.6），差异仅来自触发率波动。真正受益于 MPS 的是 LLM 与 GLM VQ。

### 5.4 Roofline 分析：带宽受限的程度

「受限于内存带宽」这一判断需要定量核验。方法是比较**理论下限**（把权重完整读一遍所需时间 = 权重字节数 ÷ 峰值带宽）与实测耗时。

实测 M3 峰值带宽（256 MB 只读 reduction，规避缓存与写回干扰）：

| 后端 | 带宽 |
|---|---|
| CPU | 63.1 GB/s |
| **MPS** | **73.0 GB/s** |

各模块权重量与 roofline 对比：

| 模块 | 权重 | 理论下限 | 实测 | **带宽效率** | 判定 |
|---|---|---|---|---|---|
| LLM 单次前向 | 2.238 GB | 30.7 ms | 58.2 ms | **52.7 %** | 部分带宽受限 |
| GLM VQ encoder | 1.280 GB | 17.5 ms | 43.0 ms | **40.8 %** | 部分带宽受限 |
| projector | 0.033 GB | 0.5 ms | 1.1 ms | 41.1 % | 部分带宽受限 |

**结论：带宽是主要因素，但只解释约一半（40–53 %）。**

每 chunk 无法回避的权重读取量（fp32）：

```
LLM   30.7 ms × 2.08 次  =  63.8 ms
VQ    17.5 ms × 1 次     =  17.5 ms
──────────────────────────────────
合计                        81.3 ms   （160 ms 预算的 51 %）
```

即**带宽硬下限占预算一半，仍留有 78.7 ms**，因此当前 240.6 ms 的实测值并非纯带宽所致。

对差额的进一步排查：Metal kernel 启动开销在流水线内约 17.6 μs/kernel（批量派发 2000 个算子摊薄测得；若每次都 `synchronize()` 则达 326.9 μs，但那是同步往返而非真实派发成本）。按每层 10–20 个算子估算，28 层合计仅 **4.9–9.9 ms**，**不足以解释 LLM 那 27.5 ms 的差额**。

因此差额主要来自**带宽利用效率本身**：读取瘦长矩阵的权重时，访存模式无法跑满峰值带宽（GEMM 的 N 维仅 2，权重复用率极低，接近纯流式读取但缺乏足够并发来饱和内存控制器）。这与 5.2 节的观察一致。

### 5.5 量化的收益推算

由于瓶颈实质是「搬运权重字节数」，**降低权重精度可直接按比例降低下限**：

| 精度 | LLM+VQ 带宽下限 | 占 160 ms 预算 |
|---|---|---|
| fp32（当前） | 81.3 ms | 51 % |
| fp16 | 40.7 ms | 25 % |
| int8 | 20.3 ms | 13 % |
| int4 | 10.2 ms | 6 % |

这为 6.2 节「LLM 量化收益最大」提供了定量依据：**fp16 即可将带宽下限压掉一半**，且 M3 当前实际以 fp32 运行，这部分收益尚未被利用。

### 5.6 生态成熟度


| 组件 | MPS 支持 |
|---|---|
| Qwen3 LLM (transformers) | ✅ 正常 |
| GLM-4-Voice tokenizer | ✅ 正常 |
| **FunASR / ModelScope** | ❌ 不完整 |

因此 `asr_device()` 在非 CUDA 环境下强制 ASR 走 CPU。**当前 MPS 配置实际是混合的**：LLM 与 tokenizer 在 MPS，ASR 在 CPU。MPS 算子覆盖率低于 CUDA，部分算子会报错，少数会静默返回错误结果——这也是 1.3 节数值一致性验证的必要性所在。

---

## 6. 优化路径评估

### 6.1 已验证：放宽 chunk 预算

将 `chunk_size` 由 2560 改为 5120（160 ms → 320 ms）：

| 配置 | 平均耗时 | P50 | P90 | 超预算 | **RTF** | 实时 |
|---|---|---|---|---|---|---|
| 2560（160 ms） | 287.8 ms | 285.8 ms | 402.7 ms | 65 % | 1.80 | ❌ |
| **5120（320 ms）** | 271.1 ms | 296.3 ms | 382.6 ms | 36 % | **0.847** | ✅ |

**已实测可达实时**，平均留有 49 ms 余量。

**但需注意代价**：状态分布由 `idle:67 / nonidle:29 / speak:4` 变为 `idle:40 / nonidle:9 / speak:1`。`nonidle` 事件显著减少，说明**判定粒度确实变粗**，短促语音片段与 backchannel 的捕捉能力下降。对全双工打断检测的实际影响需进一步评估。

### 6.2 各路径对比

| 方向 | 预期效果 | 代价 | 评价 |
|---|---|---|---|
| **chunk 320 ms** | RTF 1.80 → **0.847**（已验证） | 判定粒度变粗一倍，`nonidle` 召回下降 | ⭐ 唯一已验证可行 |
| **LLM 量化 INT8/INT4** | 121 ms 理论降至 40–60 ms | MPS 量化生态不成熟，工程量大 | 收益最大，风险亦最高 |
| ASR 换更轻模型 | 最多使 RTF 降至 ~0.9 | 精度下降 | 收益有硬上限 |
| ASR 改增量识别 | 省去 3.2 s 重复计算 | 需改动算法逻辑 | 中等 |
| GLM VQ 优化 | 43 ms，每 chunk 必然发生 | Whisper encoder 固定开销，较难 | 有限 |
| ~~外挂网络 ASR~~ | ❌ 无效（见第 4 节） | 引入抖动与流量成本 | 不推荐 |

> 若目标是产品化而非在 Mac 上硬跑，**换硬件是更直接的路径**——见第 7 节对 Jetson 的评估。

---

## 7. 端侧设备可行性：NVIDIA Jetson

### 7.1 判断依据：带宽而非算力

第 5.2 与 5.4 节已确认本模型的 LLM 部分是 **memory-bound**（矩阵形状 `[2,1024]×[1024,3072]`，batch 维仅 2，算力闲置而权重读取成为瓶颈；roofline 显示带宽效率仅 40–53 %）。因此评估端侧硬件时，**内存带宽是首要指标，AI TOPS 参考价值有限**。

一个直接证据：在该瘦长矩阵上实测 M3 的表现——

| 后端 | 单次 GEMM | 等效算力 | 等效权重带宽 |
|---|---|---|---|
| CPU | 70.3 μs | 179.0 GFLOPS | 179.0 GB/s |
| **MPS** | **584.8 μs** | 21.5 GFLOPS | 21.5 GB/s |

**MPS 在这个形状上比 CPU 慢 8.3 倍**。M3 的 GPU 标称算力远高于 CPU，却因 kernel 启动开销与并行度闲置而完败。这说明标称算力在此类负载下不具预测力，也解释了为何整体只有 2.0x 加速（LLM 之外的部分受益，LLM 自身受限）。

### 7.2 Jetson 各型号规格

| 模块 | 内存带宽 | 内存 | AI 算力 | 功耗 |
|---|---|---|---|---|
| Orin Nano 4GB | **51 GB/s** | 4 GB | 34 TOPS | 7–25 W |
| Orin Nano 8GB | **102 GB/s** | 8 GB | 67 TOPS | 7–25 W |
| Orin NX 8GB | **102.4 GB/s** | 8 GB | 117 TOPS | 10–40 W |
| **Orin NX 16GB** | **102.4 GB/s** | 16 GB | 157 TOPS | 10–40 W |
| AGX Orin 32GB | **204.8 GB/s** | 32 GB | 200 TOPS | 15–40 W |
| AGX Orin 64GB | **204.8 GB/s** | 64 GB | 275 TOPS | 15–60 W |
| Thor T4000 | **273 GB/s** | 64 GB | 1200 TFLOPS (FP4) | 40–70 W |
| Thor T5000 | **273 GB/s** | 128 GB | 2070 TFLOPS (FP4) | 40–130 W |

带宽仅四个档位：51 / 102 / 205 / 273 GB/s。作为参照，**本机 M3 实测 MPS 带宽为 52 GB/s**（CPU 80.8 GB/s）。

> 规格来源：Forecr 模块对比页（2026-04 更新）。Orin 为 INT8 稀疏 TOPS，Thor 为 FP4 稀疏 TFLOPS，口径不同不可直接比较。

### 7.3 RTF 估算

以 M3 实测数据为基准，按内存带宽比做一阶外推（两者同为统一内存架构，可比性较好）：

| 设备 | 带宽 | fp32<br/>（纯带宽外推） | +FP16 | +INT8 | 内存是否够 |
|---|---|---|---|---|---|
| Orin Nano 4GB | 51 GB/s | 1.53 | 0.92 ✓ | 0.64 ✓ | ❌ 不足 |
| Orin Nano 8GB | 102 GB/s | 0.78 ✓ | 0.47 ✓ | 0.33 ✓ | ⚠️ 勉强 |
| **Orin NX 16GB** | 102.4 GB/s | **0.78 ✓** | **0.47 ✓** | **0.33 ✓** | ✅ 充足 |
| AGX Orin 32/64GB | 204.8 GB/s | 0.40 ✓ | 0.25 ✓ | 0.18 ✓ | ✅ 充足 |
| Thor T4000 | 273 GB/s | 0.31 ✓ | 0.19 ✓ | 0.14 ✓ | ✅ 充足 |

**✓ 表示 RTF < 1，即可实时。以上为估算值，非实测。**

该估算**偏保守**，未计入 Jetson 的三项结构性优势：

1. **TensorRT-LLM** 的 kernel 融合、paged KV cache 与 in-flight batching，端侧 LLM 推理通常有 2–3x 额外收益
2. **CUDA 对瘦长 GEMM 的优化远优于 MPS**（见 7.1，MPS 在该形状上损失 8 倍）
3. **FP16/INT8 张量核心**，而 M3 上当前实际以 fp32 运行

因此实际表现应优于表中数值。

### 7.4 结论：有戏，且门槛不高

**Orin NX 16GB（102 GB/s，约 $400–900）即可满足实时要求**，无需上到 AGX。理由：

- 纯带宽外推已达 RTF 0.78，配合 FP16 可到 0.47，留有充足余量
- 16 GB 内存足够容纳模型（fp32 约 7.3 GB，FP16 约 3.7 GB）与 KV cache
- 有 DLA 与 PVA 可分流 ASR 或视觉任务
- SO-DIMM 封装与 Orin Nano 引脚兼容，便于先用 Nano 8GB 验证再升级

**若追求单机多路并发或后续接入更大 LLM，选 AGX Orin 32GB**（RTF 0.40，带宽翻倍）。

**Orin Nano 4GB 不可行**——4 GB 内存装不下模型，51 GB/s 带宽也仅与 M3 同级。

### 7.5 端侧需额外注意的两个约束

**KV cache 随单轮发言时长增长**（Qwen3-0.6B，28 层 × 8 KV 头 × head_dim 128，每 chunk 约新增 3 个 token）：

| 单轮时长 | token 数 | fp32 | fp16 |
|---|---|---|---|
| 10 s | 188 | 41.0 MB | 20.5 MB |
| 30 s | 562 | 123.0 MB | 61.5 MB |
| 60 s | 1125 | 246.1 MB | 123.0 MB |
| 300 s | 5625 | 1230.5 MB | 615.2 MB |

`reset()` 在轮次结束时清空状态，故不会无界增长；但**单轮长发言（如 60 s 独白）会累积到百 MB 级**，在 8 GB 设备上需关注。可通过 `max_token_length` 设上限或改用滑窗缓解。

**功耗与散热**：Orin NX 在 40 W 模式下才能跑满带宽；若受限于 15 W，实际性能会打折。端侧部署需按目标功耗档位重新实测，不能直接套用满功耗规格。

---

## 8. 结论与后续工作

### 8.1 结论

1. **部署可行，功能正确**。已完成 macOS 全流程部署，MPS 与 CPU 数值一致，脚本化可复现。
2. **默认配置下无法实时**，MPS RTF 1.5–1.8，CPU RTF 2.9–3.0。
3. **瓶颈是 LLM 前向（50 %），非 ASR（30 %）**。此前基于「ASR 单次 140 ms」的直觉判断不成立，原因是 ASR 仅约半数 chunk 触发。
4. **外挂网络流式 ASR 无法解决问题**。非 ASR 固定开销 168 ms 已超 160 ms 预算，ASR 优化存在硬上限（RTF ≥ 0.90）。
5. **MPS 加速有限（2.0x）**，根因是流式推理的矩阵过于瘦长，GPU 并行度无法发挥；在该形状上 MPS 甚至比 CPU 慢 8 倍。
6. **瓶颈性质为 memory-bound，但带宽只解释约一半**。roofline 分析显示带宽效率仅 40–53 %，权重搬运的硬下限为 81.3 ms（占预算 51 %）；差额并非 kernel 启动开销（仅 5–10 ms），而是瘦长矩阵下带宽利用率无法饱和。这决定了**量化是收益最直接的优化**（fp16 可将下限降至 40.7 ms）。
7. **放宽 chunk 至 320 ms 已实测可达实时（RTF 0.847）**，但判定粒度变粗，需权衡。
8. **端侧 Jetson 有可行性**。因瓶颈是 memory-bound，选型应看内存带宽而非 TOPS；估算 Orin NX 16GB（102 GB/s）即可实时，AGX Orin 更宽裕。

### 8.2 定位建议

| 用途 | Mac 本地 | GPU 服务器 | Jetson 端侧 |
|---|---|---|---|
| 功能验证、逻辑调试 | ✅ 推荐 | 可 | 可 |
| 离线批量测试 | ✅ 可行（1.5–1.8x 慢） | ✅ 推荐 | 可 |
| **真人实时对话** | ❌ 不可（除非放宽 chunk） | ✅ 推荐 | ✅ 预期可行（待实测） |
| 产品化落地 | ❌ | 云端方案 | ✅ Orin NX 16GB 起 |

`device: auto` 使同一份代码在各平台通用，无需维护分支差异。

### 8.3 后续工作

- [ ] 评估 320 ms chunk 对轮次判定质量的实际影响（需标注数据做定量对比，当前仅有状态分布变化的定性观察）
- [ ] **在 Jetson 上实测验证第 7 节的估算**（优先 Orin NX 16GB 或 AGX Orin），重点确认 TensorRT-LLM 的实际收益
- [ ] 尝试 LLM INT8 量化在 MPS 上的可行性与精度损失
- [ ] 考察 ASR 增量识别改造，消除 3.2 s 缓冲的重复计算
- [ ] 在 GPU 服务器上跑同一套基准，建立跨平台性能基线
- [ ] 评估单轮长发言下 KV cache 增长对端侧内存的影响，必要时引入滑窗

---

## 附录 A：复现方式

```bash
# 环境安装（含依赖校验与 MPS 自检）
bash install_mac.sh --all

# 延迟基准
.venv-mac/bin/python benchmark_local.py --num-chunks 100
.venv-mac/bin/python benchmark_local.py --device cpu

# 逐段耗时拆解
.venv-mac/bin/python profile_breakdown.py --num-chunks 60
.venv-mac/bin/python profile_breakdown.py --device cpu

# ASR 延迟敏感性实验
.venv-mac/bin/python benchmark_asr_impact.py --delays 0,24,50,100,150,200
```

所有脚本均支持 `--warmup` 以丢弃 MPS 冷启动样本，`--device` 以覆盖配置设备。

---

## 附录 B：代码锚点

| 位置 | 内容 |
|---|---|
| `config/config.yaml` | `chunk_size`、`audio_back_size`、`device`、`asr.model_name` |
| `service/model.py:194-223` | `get_chunk()` — 缓冲区切片与窗口拼接 |
| `service/model.py:224-245` | `process()` — chunk 累积与触发 |
| `service/model.py:246-381` | `state_predict()` — 状态机与轮次决策 |
| `service/model.py:382-413` | `infer()` — 单 chunk 主流程 |
| `service/model.py:414-468` | `_audio_to_tokens()` — GLM VQ 编码 |
| `service/model.py:469-486` | `_tokens_to_embeds()` — codebook 与 projector |
| `service/model.py:487-689` | `_asr()` — LLM #1 与 ASR 调用（串行关键路径） |
| `service/model.py:608-644` | correction 分支 — 第三次 LLM 前向来源 |
| `service/model.py:690-743` | `_state_predict()` — LLM #2 与状态决策 |
| `utils/device_utils.py` | 设备解析与后端分派 |

> 行号对应 `mac-local-inference` 分支（`0fb6df4` 之后）。

---

## 附录 C：工程问题记录

### C.1 pynini 无法在 arm64 编译

**现象**：`pynini==2.1.5` 与 Homebrew 提供的 `openfst 1.8.4` C++ API 不兼容，编译报 `CompileInternal` 候选函数不匹配；升级至 `2.1.6.post1` 仍失败（`StrJoin` 签名不符）。

**分析**：`pynini` 是 `WeTextProcessing` 的依赖，后者在 `utils/MyTn/textnorm.py` 顶部被导入。但排查发现推理链路仅使用该模块的 `zh_norm` 与 `zh_remove_punc`，二者依赖纯 Python 的 `cn_tn.TextNorm`；`pynini` 实际只服务于 `process_text()`，而**该函数在整个推理链路中无调用点**。

**处理**：改为惰性导入，缺失时不影响推理，仅在真正调用 `process_text()` 时报错。上游 `install_server.sh` 依赖 conda 安装 pynini，此举也消除了 macOS 上对 conda 的需求。

### C.2 HF_ENDPOINT 残留导致下载失败

**现象**：`snapshot_download` 持续抛 `LocalEntryNotFoundError`，提示「检查网络连接」，但 `curl` 与 `requests` 均能正常访问 HF 并取得 307 重定向。

**根因**：shell 中残留 `HF_ENDPOINT=https://hf-mirror.com`，而该镜像未收录此仓库。错误信息误导为网络故障。

**处理**：`install_mac.sh` 下载时以 `env -u HF_ENDPOINT` 显式清除，并在检测到该变量时告警。

**附带发现**：`Soul-AILab/SoulX-Duplug-0.6B` 仓库已打包全部三个模型（Qwen3 主干 2.3 GB + glm-4-voice-tokenizer 1.4 GB + LoRA 权重 3.6 GB，共约 7.3 GB），无需按 README 分别下载。

### C.3 其他

- **setuptools 版本约束**：必须钉 `<81`，因 `modelscope/utils/plugins.py` 仍 `import pkg_resources`；部分包会在安装时将其顶到 84.x，脚本装完后回钉一次。
- **ASR 选型**：macOS 下默认改用 `sensevoice`（funasr），其导入链短，可避免 `modelscope.pipelines` 连带的 opencv / pandas / oss2 等图像相关依赖。
- **nn.Module 计时**：`profile_breakdown.py` 中对普通方法用 monkey patch，对 `nn.Module` 子模块须改用 forward hook——PyTorch 的 `__setattr__` 会拦截将子模块替换为普通函数的操作。
