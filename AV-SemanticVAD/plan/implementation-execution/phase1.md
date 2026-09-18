# Phase 1 执行日志（数据集：AVSC-Corpus）

对应 [`../implementation-plan.md`](../implementation-plan.md) §1（数据来源/许可/标注流水线）。
本文件记录 Phase 1 实际做了什么、结论、与计划的偏差。上手导航见 [`../README.md`](../README.md)。

> ## 📌 Phase 1 当前结论摘要（TL;DR）
>
> 1. **主训练源锁定 MM-F2F（英文）**，走**英文一条线先跑通**；中文暂无现成大规模 AV 源。
> 2. **核实原始来源，订正计划三处错误**：MM-F2F 是**英文非中文**；**不发布媒体**（只发标注+YouTube链接+脚本）；原「去标识化可绕开人脸合规」不成立。
> 3. **本机环境拦路石**：不通 YouTube / Google Drive（curl=000）；hf-mirror 只代理文件、`/api` 与页面 308 跳回被封的 huggingface.co；Seamless Interaction 为 gated。→ 媒体须在外部机器下载后传回。
> 4. **取数方案（两步交接）已就位**：先传小标注 → 我算出所需视频子集 → 只下该子集。清单见 `data/MM-F2F/download_list.txt`。
> 5. **待办**：① 用户上传标注 ② 探针 B 下游脚本（切分/抽脸/MediaPipe/GRU/噪声切片）。

状态图例：✅ 完成 ｜ 🔄 进行中 ｜ ⛔ 受阻 ｜ ⏳ 待做

| 任务 | 状态 |
|---|---|
| 1.1 数据源核实与许可（MM-F2F） | 🔄 已核实来源/许可/分发方式，再分发条款仍待核（§1.1.1） |
| 数据获取通道可行性排查（本机网络） | ✅ 已查清：YouTube/GDrive 不通，仅国内通道可达 |
| MM-F2F 取数交接（下载→落盘） | 🔄 等用户上传标注 |
| 探针 B 下游脚本 | ⏳ 待做 |

---

## 1.1 — MM-F2F 数据源核实（2026-09-16）★ 对计划的订正

**触发**：Phase 0 结论要"取带视频 AV 数据验证 sufficiency"，计划把 **MM-F2F** 列为主训练集。
上手前核实其原始来源（论文 arXiv:2505.12654 + 仓库 github.com/Linyx1125/MM-F2F）。

### 核实到的事实（原始来源）

| 项 | 事实 | 出处 |
|---|---|---|
| 论文 | 厦大，ACL 2025；turn-taking/backchannel 预测 | arXiv:2505.12654 / aclanthology 2025.acl-long.743 |
| **语言** | **英文**（"in-the-wild online **English** conversation videos"，验证者确认 entirely in English） | 论文 HTML / repo README |
| 规模 | 210h、773 视频、~20M 帧、1.5M 词、51K turn | README abstract |
| 标注 | 词级 CSV：`video_id, sentence_id, text, start, end, label, speaker`；`label` 0=keep/1=turn/2=backchannel（**听者行为**=轴2，非语义完整性轴1） | dataset/README.md |
| **媒体分发** | **不发布处理后媒体**："we have decided **not to release the processed data directly** … we provide the **original video links + processing scripts**"；视频须按 `video_id` 自 YouTube 下 | dataset/README.md |
| 去标识化 | 论文描述 >10,000 合成人脸替换 + 声纹扰动（消融 0.836→0.823），**但该去标识版不发布**；实际拿到的是生 YouTube 视频 | 论文 + dataset/README |
| 代码许可 | **MIT** | LICENSE |

### 对计划的订正（已改权威文档）

三处错误已在 `implementation-plan.md`（顶部 changelog + §0.1.1 表 + §1.1 表/段 + §4.5.8）、
`research/evidence-table.md`、`benchmark-proposal.md`、`phase0.md`、`plan/README.md` 订正：

1. **MM-F2F 是英文，非中文** —— 影响"中文主训练"假设与 §4.5.8 语言拆分。
2. **不发布媒体** —— 只发标注+链接+脚本；原 §1.1「用它可绕开人脸合规风险」**不成立**（拿到的是生 YouTube、非去标识；我们同样不能再分发媒体，只能仿其发标注+脚本）。
3. **中文侧无现成大规模 AV 源** —— 只能自采（§1.1.3，需伦理）或 Full-Duplex-Bench-zh 补测试集。

### 决策（2026-09-16，用户拍板）

- **英文一条线先跑通**：主训练/开发/探针 B 全用 MM-F2F（英文）。
- 中文是否投入自采，待英文线 sufficiency 有结论后再定，**不阻塞当前进度**。

---

## 数据获取通道可行性排查（本机网络，2026-09-16）✅

**做法**：`curl -m` 探连通性 + `huggingface_hub` 探 mirror。

| 目标 | 结果 |
|---|---|
| YouTube / Google Drive / Google | ❌ 全 `000`（不通）；yt-dlp 未安装 |
| 百度网盘 / GitHub / raw.githubusercontent / hf-mirror / ModelScope | ✅ 200/301/302 可达 |
| hf-mirror 下载 HF 数据集 | ❌ 仅代理文件内容；`/api/*` 与 `/datasets/<id>` 页面 **308 跳回 huggingface.co（被封）**；`huggingface_hub` 报 `Cannot assign requested address` |
| Seamless Interaction / MM-VAP 经 mirror | ❌ 同上；Seamless 还是 gated（需登录同意条款） |
| 本机已有数据 | 仅纯音频（Easy-Turn-en、Full-Duplex-Bench-zh 均 wav+json，**无视频**） |

**结论**：MM-F2F 媒体（YouTube）与标注（Google Drive）在本机都拿不到；mirror 路线不可行。
→ 媒体须在**能访问 YouTube/Google 的外部机器**下载后传回本机。

---

## 取数交接方案（两步，压缩下载量）🔄 等用户上传标注

产物：
- `data/MM-F2F/HANDOFF.md`（详细交接说明）
- `data/MM-F2F/download_list.txt`（纯清单，给外部机器操作）
- 目录：`data/MM-F2F/{annotations,videos}/`（已建）

流程：
1. **用户**：下标注（GDrive/百度，几 MB）→ 解压到 `data/MM-F2F/annotations/`（`preprocess/*.csv` + `word_level_split/`）。
2. **我**：解析标注 → 选探针 B 所需视频子集（几十个）→ 给出 `video_id` 列表 + yt-dlp 命令。
3. **用户**：只下该子集视频 → `data/MM-F2F/videos/<video_id>.<ext>`（文件名=video_id）。
4. **我**：切分音频 → 抽脸 → MediaPipe 24 维 → GRU 探针 B + 噪声切片评测（收尾 Phase 0 的 P0.4 / 验证 sufficiency）。

---

## 探针 B 首跑（AMI ES2002 系列，2026-09-17）⚠️ null 结果

**流水线**（脚本 `scripts/probe_b_*.py`）：词级标注切句内停顿（同人保floor、无他人插话）
→ 探针级弱标注（填充词/悬空词→incomplete；终结标点→complete；剔除 >3s"边画边停"与 ambiguous）
→ speaker↔camera 映射（meetings.xml）→ FaceLandmarker 24 维/帧（21 blendshape 口/眼/注视/眉 + 3 头姿）
→ 窗口 [停顿前1.6s, 停顿结束] 池化 73 维 → logistic 5 折 CV。

**数据**：ES2002{a,b,c,d}，**N=154（complete 96 / incomplete 58）**，人脸有效帧比例中位 0.92。

**结果**：
| 特征 | 5 折 AUC |
|---|---|
| **visual_all (73 维)** | **0.421 ± 0.109** |
| pause_dur（混淆基线） | 0.433 |
| face_valid_ratio | 0.374 |
| （sanity）视觉→说话人身份 | acc **0.649** vs 随机 0.25 |

**判读**：视觉特征**能分辨说话人（特征有效、抽取无 bug），但预测 complete/incomplete ≈ 随机**
（AUC 0.42，落在计划 §P0.4 的「≈0.50 → 干净下视觉无用」档）。**与全部外部证据一致**
（MM-F2F/AV-Dialog clean 视觉仅 +1~1.3%，VideoFDB 加视频变差）。

**诚实 caveats（未定论，抑制信号的可能因素）**：
1. 弱标注是**文本启发式**（填充词/标点），非人工/LLM 语义判断，有噪声。
2. **AMI 分辨率低**（352×288，脸仅占 ~5%），注视/口型微动可能丢失。
3. **会议域**：4 人会议里注视/头姿由「看谁/看幻灯」驱动，与话轮完整性可能无关甚至混淆。
4. N 小、单组 4 人；5 折 CV 因同人跨折而偏乐观。
5. 干净条件；计划的充分性主张本在**噪声**下，但视觉在干净上≈随机 → 「噪声下补回」存疑（无信号难补）。

**对路线的影响**：支持计划风险单「clean 增益≈0（高概率）」与下限定位（benchmark/数据集 + 诚实 negative/鲁棒性）。
**不足以单独判定项目成败**——需先消除 caveat 1–3（更好标签 / 更高清人脸 / 更对靶域）再复核。

### 追加：GRU 复核（回应"该用神经网络学"，2026-09-17）
把逐帧 24 维序列喂因果 GRU（hidden32+dropout+早停+类权重，计划 §2.2 设计），排除"线性/池化太糙"：
| 方法 | 分层5折 AUC | 留一会议 AUC |
|---|---|---|
| 线性探针(池化) | 0.421 | — |
| **因果 GRU(时序)** | **0.424±0.049** | **0.343±0.099** |
→ **时序+非线性同样≈随机，跨会议更掉到 0.34（无可迁移信号）**。逐特征单变量 AUC 中位 |dev|=0.03，
最强线索仅 browDown(皱眉)≈0.40（"思考脸"方向，噪声范围内）。**null 稳健，非方法问题。**

**关键区分（避免误推广）**：本探针测的是**语义完整性**(complete/incomplete)。文献里视觉"噪声下 +13%"
（AV-Dialog）/"52%→72%"（MM-VAP）多是 **VAD/话轮边界**（唇动≈语音活动）这条轴，**不是语义完整性**。
故"视觉对语义完整性帮助有限"与"视觉在噪声下对 VAD 有用"可并存 —— **尚未测的是噪声下语义完整性充分性**。

## 待办

- ⏳ 用户上传 MM-F2F 标注（第 1 步）。
- ⏳ 探针 B 下游脚本：标注解析+子集筛选 / 音频切分 / 抽脸 / MediaPipe 特征 / GRU 探针 B / 噪声切片评测。
- ⏳ §1.1.1 MM-F2F 再分发条款正式核验（衍生标注能否随论文发布）。
