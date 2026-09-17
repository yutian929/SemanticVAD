#!/usr/bin/env python
"""P0.1 — 核实 V1 / V2（纯读配置，无需环境/权重加载）。

V1: 读 config.json，锁定 hidden_size / vocab_size / 层数 → 定 Projector 与 LoRA 参数量。
V2: 查 tekken.json，判定 token id 41–50 是否空闲 → 决定 ci 判决头能否零参数复用
    vad_lm_head（方案 H-A）；若被占用则退方案 H-B（+6K 参数）。

对应 implementation-plan.md §P0.1。只读，不修改任何第三方目录。
"""
import json
import sys
from pathlib import Path

MODEL_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "/home/thundrzhang/SemanticVAD/X2-Turn/models/X2-Turn-4B-0812"
)

# code-anchors.md §1: TURN_CLASS_IDS = (35..40)；我们拟用 CI_CLASS_IDS = (41,42)
TURN_CLASS_IDS = tuple(range(35, 41))          # 35,36,37,38,39,40
CI_CLASS_IDS = (41, 42)                          # complete / incomplete 候选
PROBE_RANGE = range(41, 51)                       # 计划要求核对 41–50


def load_json(name):
    with open(MODEL_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def verify_v1(cfg):
    print("=" * 70)
    print("V1 — config.json 关键字段")
    print("=" * 70)
    text = cfg.get("text_config", {})
    audio = cfg.get("audio_config", {})
    fields = {
        "hidden_size (top)": cfg.get("hidden_size"),
        "text_config.hidden_size": text.get("hidden_size"),
        "text_config.vocab_size": text.get("vocab_size"),
        "text_config.num_hidden_layers": text.get("num_hidden_layers"),
        "text_config.num_attention_heads": text.get("num_attention_heads"),
        "text_config.head_dim": text.get("head_dim"),
        "audio_config.hidden_size": audio.get("hidden_size"),
        "audio_length_per_tok": cfg.get("audio_length_per_tok"),
        "downsample_factor": cfg.get("downsample_factor"),
        "default_num_delay_tokens": cfg.get("default_num_delay_tokens"),
        "projector_hidden_act": cfg.get("projector_hidden_act"),
        "dtype": cfg.get("dtype"),
        "architectures": cfg.get("architectures"),
    }
    for k, v in fields.items():
        print(f"  {k:38s} = {v}")
    H = text.get("hidden_size") or cfg.get("hidden_size")
    V = text.get("vocab_size")
    print(f"\n  → 主解码器 hidden_size H = {H}")
    print(f"  → vocab_size V = {V}")
    return {"hidden_size": H, "vocab_size": V,
            "num_hidden_layers": text.get("num_hidden_layers")}


def verify_v2(tekken):
    print("\n" + "=" * 70)
    print("V2 — tekken.json：token id 41–50 是否空闲（决定 H-A 零参数方案）")
    print("=" * 70)

    # tekken.json 结构探测：不同版本可能是 {"vocab": [...]}, 或含 special_tokens
    top_keys = list(tekken.keys()) if isinstance(tekken, dict) else ["<list>"]
    print(f"  tekken.json 顶层键: {top_keys}")

    special = None
    vocab = None
    if isinstance(tekken, dict):
        vocab = tekken.get("vocab")
        # Mistral tekken 常见: config + vocab(list of {rank,token_bytes,...}) + special_tokens
        special = tekken.get("special_tokens") or tekken.get("special_token_map")
        cfg = tekken.get("config")
        if cfg:
            print(f"  tekken config: {cfg}")

    # 判定原则（关键）：tekken 的全部 1000 个 special_tokens 都是"槽位"，
    # 一个 id 是否"空闲"不看它有没有条目，而看它是不是未命名的占位符
    # `<SPECIAL_N>`。X2-Turn 的 TURN_CLASS_IDS(35-40) 本身也是 `<SPECIAL_35..40>`，
    # 语义是在代码(modeling.py:15)里赋予的，词表并不区分。因此凡是 `<SPECIAL_N>`
    # 形态即可安全征用。
    import re
    placeholder = re.compile(r"^<SPECIAL_\d+>$")
    if special is not None:
        print(f"\n  special_tokens 共 {len(special)} 项，列出 rank<=60 的：")
        norm = []
        for i, s in enumerate(special):
            if isinstance(s, dict):
                rank = s.get("rank", s.get("id", i))
                tok = s.get("token_str") or s.get("token") or s.get("content")
            else:
                rank, tok = i, s
            norm.append((rank, tok))
        for rank, tok in sorted(norm):
            if rank is not None and rank <= 60:
                flag = ""
                if rank in TURN_CLASS_IDS:
                    flag = "  <- TURN_CLASS_IDS (35-40)"
                elif rank in PROBE_RANGE:
                    flag = "  <- 探测区 41-50"
                print(f"    rank {rank:>3}: {tok!r}{flag}")

        tok_by_rank = {rank: tok for rank, tok in norm}
        # "占用" = 该 id 有实义命名（非 <SPECIAL_N> 占位符）
        occupied = {i for i in PROBE_RANGE
                    if i in tok_by_rank and not placeholder.match(str(tok_by_rank[i]))}
        free = [i for i in PROBE_RANGE
                if i not in tok_by_rank or placeholder.match(str(tok_by_rank[i]))]
        # 对照检查：确认 X2-Turn 复用的 35-40 也全是占位符（证明"征用占位符"就是上游做法）
        turn_all_placeholder = all(
            i in tok_by_rank and placeholder.match(str(tok_by_rank[i]))
            for i in TURN_CLASS_IDS)

        print(f"\n  35–40 (X2-Turn 已征用) 是否全为 <SPECIAL_N> 占位符: "
              f"{turn_all_placeholder}  ← 证明征用占位符即上游做法")
        print(f"  探测区 41–50 中被实义占用: {sorted(occupied) if occupied else '无'}")
        print(f"  探测区 41–50 中空闲(占位符): {free}")
        ci_free = all(i in free for i in CI_CLASS_IDS)
        print(f"\n  → CI_CLASS_IDS {CI_CLASS_IDS} 是否空闲: {ci_free}")
        print(f"  → 方案: {'H-A（零参数复用 vad_lm_head 空间）可行' if ci_free else 'H-A 不成立 → 退 H-B（+6K 参数）'}")
        return {"ci_free": ci_free,
                "occupied_41_50": sorted(occupied),
                "free_41_50": free,
                "turn_ids_35_40_all_placeholder": turn_all_placeholder,
                "ci_class_ids": list(CI_CLASS_IDS)}

    print("  ⚠️ 未找到 special_tokens 字段；需人工检视结构。"
          f" vocab 是否存在: {vocab is not None}")
    return {"ci_free": None, "note": "special_tokens field not found"}


def main():
    print(f"MODEL_DIR = {MODEL_DIR}\n")
    cfg = load_json("config.json")
    tekken = load_json("tekken.json")
    v1 = verify_v1(cfg)
    v2 = verify_v2(tekken)
    print("\n" + "=" * 70)
    print("SUMMARY (JSON)")
    print("=" * 70)
    print(json.dumps({"V1": v1, "V2": v2}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
