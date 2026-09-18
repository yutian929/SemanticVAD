#!/usr/bin/env python
"""探针级弱标注：给句内停顿事件打 complete/incomplete（启发式，透明可复现）。

⚠️ 探针级代理，非最终标签。真实流水线用 LLM(§1.2.3)。这里用停顿点前文的
句法/词法线索：终结标点→complete；填充词/悬空虚词/逗号→incomplete；其余→ambiguous(丢弃)。
另按 §观察 过滤：仅保留 min<=dur<=max（默认剔除 >3s 的"边画边停")。

用法：python probe_b_label.py ES2002a [--min 0.25 --max 3.0]
输入 results/probe_b/<m>_pauses.json → 输出 results/probe_b/<m>_labeled.json
"""
import sys, os, re, json, argparse, collections

FILLERS = {"um", "uh", "uhm", "hmm", "mm", "er", "erm", "eh", "mm-hmm", "mmhmm", "hm"}
DANGLING = {  # 结尾若是这些功能词，句子多半没说完
    "and","but","or","so","because","cause","'cause","that","which","who","to","of",
    "for","the","a","an","is","are","was","were","be","been","being","am","will","would",
    "can","could","should","might","must","i","we","they","he","she","it","you","my","your",
    "his","her","our","their","its","like","with","at","in","on","as","if","when","then",
    "just","kind","sort","gonna","wanna","about","into","from","by","this","these","those",
}

def last_content_token(pre):
    toks = pre.split()
    return toks[-1] if toks else ""

def label(pre):
    p = pre.rstrip()
    # 终结标点（句号/问号/叹号）紧邻停顿 → complete
    if re.search(r"[.?!]\s*$", p):
        return "complete", 0.8, "terminal_punc"
    ends_comma = bool(re.search(r",\s*$", p))
    # 取最后一个字母数字 token
    w = last_content_token(re.sub(r"[^\w'\-\s]", " ", p)).lower()
    if w in FILLERS:
        return "incomplete", 0.9, "filler"
    if w in DANGLING:
        return "incomplete", 0.75, "dangling_word"
    if ends_comma:
        return "incomplete", 0.6, "comma"
    return "ambiguous", 0.0, "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting")
    ap.add_argument("--min", type=float, default=0.25)
    ap.add_argument("--max", type=float, default=3.0)
    args = ap.parse_args()

    inf = f"AV-SemanticVAD/results/probe_b/{args.meeting}_pauses.json"
    events = json.load(open(inf))
    out = []
    dropped_dur = dropped_amb = 0
    for e in events:
        if not (args.min <= e["dur"] <= args.max):
            dropped_dur += 1; continue
        lab, conf, rule = label(e["pre_text"])
        if lab == "ambiguous":
            dropped_amb += 1; continue
        e = dict(e); e.update(label=lab, ci=1 if lab == "complete" else 0,
                              conf=conf, rule=rule)
        out.append(e)

    cnt = collections.Counter(e["label"] for e in out)
    rules = collections.Counter(e["rule"] for e in out)
    print(f"# {args.meeting}: kept {len(out)} (drop dur={dropped_dur}, ambiguous={dropped_amb})")
    print(f"#   labels: {dict(cnt)}  | rules: {dict(rules)}")
    outf = f"AV-SemanticVAD/results/probe_b/{args.meeting}_labeled.json"
    json.dump(out, open(outf, "w"), ensure_ascii=False, indent=1)
    print(f"#   saved -> {outf}")
    return out

if __name__ == "__main__":
    main()
