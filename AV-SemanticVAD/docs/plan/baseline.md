# Baseline：免训练的现成模块级联

> 全部用现成的预训练模型 + 规则拼装，不做任何训练；输入输出与 [`architecture.md`](architecture.md) §0 一致。
> 设当前是第 `t` 个 80 ms 片段，画面里 `K` 个人，共 `S = K + 1` 条流（K 张脸 ＋ 1 条 `others`）。
>
> **最后更新**：2026-09-28（ASR 与话轮状态改为直接用 X2-Turn；各模型选型后续逐个讨论）

```mermaid
flowchart TB
    classDef inp fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef mod fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef rule fill:#fffde7,stroke:#f9a825,color:#0d1b2a;
    classDef var fill:#ffffff,stroke:#90a4ae,color:#0d1b2a;
    classDef outp fill:#fff3e0,stroke:#e65100,color:#0d1b2a;

    subgraph R1["第一段 · 感知（现成预训练模型，只推理）"]
        direction LR
        X(["混合音频 audio chunk<br/>x_t · [1280] · float32"]):::inp
        F(["人脸 crop face clips<br/>F_t · [K, n_v, 3, H, W] · uint8"]):::inp
        X2["① X2-Turn<br/>一份 · 只听混合音频"]:::mod
        Y(["本帧的字 token<br/>y_t · [1] · int64<br/>(无新字 = PAD 32)"]):::var
        Q(["说话状态概率 state prob<br/>q_t · [6] · float32"]):::var
        ASD["② Light-ASD<br/>每张脸一份"]:::mod
        PA(["每张脸的说话概率 speaking prob<br/>p_asd_t · [K, n_v] · float32"]):::var

        X --> X2
        X2 --> Y
        X2 --> Q
        X --> ASD
        F --> ASD --> PA
    end

    subgraph R2["第二段 · 拼装（规则 + 现成控制器）"]
        direction LR
        Y2(["本帧的字<br/>y_t · [1] · int64"]):::var
        Q2(["说话状态概率<br/>q_t · [6] · float32"]):::var
        PA2(["每张脸的说话概率<br/>p_asd_t · [K, n_v] · float32"]):::var
        ASG["③ ASD-argmax 分配<br/>阈值 τ_a"]:::rule
        YS(["每条流的字 per-stream token<br/>ŷ_t · [S] · int64"]):::var
        SS(["每条流的状态 per-stream state<br/>ŝ_t · [S] · int64 ∈ 6 类"]):::var
        P(["头姿视线 + 位置框<br/>P_t · [K, n_v, d_p] · B_t · [K, n_v, 4] · float32"]):::inp
        GZ["④ 视线夹角阈值 θ"]:::rule
        GA(["是否看着机器人 looking at robot<br/>g_t · [K] · bool"]):::var
        OUT(["逐帧输出 · 每条流一行<br/>person_id: str · state: int64 ∈ 6 类<br/>token: int64 · addressee: float"]):::outp
        CT["⑤ FrameTurnController<br/>每条流一份 · 阈值 ρ"]:::mod
        EO(["说完一句的事件 accept event<br/>person_id: str · text: str<br/>to_robot: bool · conf: float"]):::outp

        Y2 --> ASG
        Q2 --> ASG
        PA2 --> ASG
        ASG --> YS
        ASG --> SS
        P --> GZ --> GA
        YS --> OUT
        SS --> OUT
        GA --> OUT
        OUT --> CT --> EO
    end

    R1 ==>|"y_t、q_t、p_asd_t 进入第二段"| R2
```

## 各模型介绍

| 步 | 模型 / 方法 | 介绍 |
|---|---|---|
| ① | **X2-Turn**（X2-Turn-4B-0812） | 流式语音识别 + 话轮状态：只听混合音频，每 80 ms 输出一个字 token 和 6 类状态（`idle` / `noidle` / `speaking` / `turn_end` / `backchannel` / `uncertain`）；中英双语，输出对应 480 ms 前那一帧 |
| ② | **Light-ASD** | 主动说话人检测：结合声音和每张脸的唇动，判断这张脸是否在说话（AVA 预训练） |
| ③ | **ASD-argmax 分配** | 规则：每一帧 X2-Turn 的字和状态，分给这一帧 ASD 概率最高的那张脸；都低于阈值 `τ_a` 则给 `others`；其余流记为 `idle` + PAD |
| ④ | **视线夹角阈值 θ** | 规则：视线与镜头方向的夹角小于 `θ`，即判为"看着机器人" |
| ⑤ | **FrameTurnController**（X2-Turn 自带） | 每条流一份：把字拼成句子、判断说完没，说完时发出 accept 事件；`to_robot` = 这句话期间看着机器人的帧比例大于 `ρ` |
