# 执行记录（implementation-execution）

> **这是探针与调研的事实记录。** 计划「应该做什么」现归 [`../cvpr-framing.md`](../cvpr-framing.md) 与 [`../architecture.md`](../architecture.md)；
> 本目录只记录「实际做了什么、得到什么结果、与计划的偏差」，**按项目 phase 分文件**。


## 按 phase 分的执行日志

| 文件 | 项目 phase | 状态 | 一句话结论 |
|---|---|---|---|
| [`phase0.md`](phase0.md) | Phase 0 · 前置探针与环境 | ✅ 完成 | 音频干净近天花板；噪声下崩塌→视觉有潜在主场；sufficiency 待验 |
| [`phase1.md`](phase1.md) | Phase 1 · 数据集 AVSC-Corpus | 🔄 进行中 | MM-F2F=英文/不发媒体（订正计划）；英文优先；本机不通 YouTube/GDrive→外部下载交接 |
| [`phase2.md`](phase2.md) | Phase 2 · 视觉模态（多人 per-face） | 🔄 进行中 | 转向多人 per-face 语义完整性；深度调研裁决新颖（钉"语义完整性+per-face"）；MuVAP/AV-Dialog 是最近竞品 |
| （future）`phase3..5.md` | Phase 3–5 | ⏳ 未开始 | — |

> 今天（2026-09-13-14）的所有工作——环境、探针 A、Gate、噪声 gate——**在权威计划里都属 Phase 0**，
> 故全部记在 `phase0.md`。

## 约定

- 每个任务记：**做法 / 命令 / 结果 / 结论 / 与计划的偏差**。
- 产物落位：脚本 → `AV-SemanticVAD/scripts/`；结果 → `AV-SemanticVAD/results/`；
  测试 → `AV-SemanticVAD/tests/`。**不改三个只读第三方目录**（`X2-Turn/`、`SoulX-Duplug/`、`SoulX-Duplug-training/`）。
- 旧计划附录 D 的待核实项（V1–V13）随该文件删除；仍有效的未决项见 `../cvpr-framing.md` §5。
