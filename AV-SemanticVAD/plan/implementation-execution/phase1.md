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

## 待办

- ⏳ 用户上传 MM-F2F 标注（第 1 步）。
- ⏳ 探针 B 下游脚本：标注解析+子集筛选 / 音频切分 / 抽脸 / MediaPipe 特征 / GRU 探针 B / 噪声切片评测。
- ⏳ §1.1.1 MM-F2F 再分发条款正式核验（衍生标注能否随论文发布）。
