# 执行记录（implementation-execution）

> **这是计划的执行日志。** 计划「应该做什么」在 [`../implementation-plan.md`](../implementation-plan.md)（唯一权威）；
> 本目录只记录「实际做了什么、得到什么结果、与计划的偏差」，**按项目 phase 分文件**。
> 上手导航见 [`../README.md`](../README.md)。

## 按 phase 分的执行日志

| 文件 | 项目 phase | 状态 | 一句话结论 |
|---|---|---|---|
| [`phase0.md`](phase0.md) | Phase 0 · 前置探针与环境 | ✅ 完成 | 音频干净近天花板；噪声下崩塌→视觉有潜在主场；sufficiency 待验 |
| [`phase1.md`](phase1.md) | Phase 1 · 数据集 AVSC-Corpus | 🔄 进行中 | MM-F2F=英文/不发媒体（订正计划）；英文优先；本机不通 YouTube/GDrive→外部下载交接 |
| （future）`phase2..5.md` | Phase 2–5 | ⏳ 未开始 | — |

> 今天（2026-09-13-14）的所有工作——环境、探针 A、Gate、噪声 gate——**在权威计划里都属 Phase 0**，
> 故全部记在 `phase0.md`。它的结论（新 framing）也已同步进 `../implementation-plan.md` 顶部 changelog。

## 约定

- 每个任务记：**做法 / 命令 / 结果 / 结论 / 与计划的偏差**。
- 产物落位：脚本 → `AV-SemanticVAD/scripts/`；结果 → `AV-SemanticVAD/results/`；
  测试 → `AV-SemanticVAD/tests/`。**不改三个只读第三方目录**（`X2-Turn/`、`SoulX-Duplug/`、`SoulX-Duplug-training/`）。
- 待核实项（V1–V13）在计划附录 D，本记录负责把它们逐个标为「已解锁 + 答案」。
