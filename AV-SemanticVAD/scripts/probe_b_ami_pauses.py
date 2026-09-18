#!/usr/bin/env python
"""从 AMI 词级标注抽「句内停顿」事件（探针 B 用）。

定义：同一说话人相邻词之间的静音间隔 >= 阈值，且该间隔内**无其他说话人**在说话
（= 说话人保持话语权、停顿后继续 = within-turn hold 候选）。
输出每个事件的 前文/后文/时长，供弱标注 complete/incomplete。

用法：python probe_b_ami_pauses.py ES2002a [--min-pause 0.25] [--ctx 12]
"""
import sys, os, re, json, glob, argparse, html
import xml.etree.ElementTree as ET

ANNO = "data/AMI/annotations/unzipped"

def load_words(meeting, spk):
    f = f"{ANNO}/words/{meeting}.{spk}.words.xml"
    if not os.path.exists(f):
        return []
    root = ET.parse(f).getroot()
    out = []
    for w in root:
        tag = w.tag.split('}')[-1]
        if tag != 'w':
            continue
        st, et = w.get('starttime'), w.get('endtime')
        if st is None or et is None:
            continue
        out.append({
            'spk': spk,
            'start': float(st), 'end': float(et),
            'text': html.unescape((w.text or '').strip()),
            'punc': w.get('punc') == 'true',
            'trunc': w.get('trunc') == 'true',
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('meeting')
    ap.add_argument('--min-pause', type=float, default=0.25)
    ap.add_argument('--ctx', type=int, default=12, help='前文词数')
    ap.add_argument('--post', type=int, default=5, help='后文词数')
    args = ap.parse_args()

    spks = ['A', 'B', 'C', 'D']
    by_spk = {s: load_words(args.meeting, s) for s in spks}
    all_speech = sorted(
        [w for s in spks for w in by_spk[s] if not w['punc']],
        key=lambda w: w['start'])

    def others_talk(spk, t0, t1):
        # 间隔内是否有其他说话人的语音（留 50ms 容差）
        for w in all_speech:
            if w['spk'] == spk:
                continue
            if w['end'] > t0 + 0.05 and w['start'] < t1 - 0.05:
                return True
        return False

    events = []
    for spk in spks:
        ws = by_spk[spk]
        # 用于前后文重建（含标点），及仅语音词用于计时
        for i in range(len(ws) - 1):
            a = ws[i]
            # 找下一个"语音词"作为恢复点
            if a['punc']:
                continue
            j = i + 1
            while j < len(ws) and ws[j]['punc']:
                j += 1
            if j >= len(ws):
                break
            b = ws[j]
            gap = b['start'] - a['end']
            if gap < args.min_pause:
                continue
            if others_talk(spk, a['end'], b['start']):
                continue  # 他人插话 → 不是句内 hold
            # 前文：截至恢复词 j 之前（含 a 及其后的尾随标点），取最后 ctx 个 token
            pre = ws[max(0, i - args.ctx * 2):j]
            pre_txt = _join(pre)
            post = ws[j:j + args.post * 2]
            post_txt = _join(post)
            events.append({
                'meeting': args.meeting, 'spk': spk,
                'pause_start': round(a['end'], 2),
                'pause_end': round(b['start'], 2),
                'dur': round(gap, 2),
                'pre_text': pre_txt, 'post_text': post_txt,
            })
    events.sort(key=lambda e: e['pause_start'])

    # 统计
    import collections
    buckets = collections.Counter()
    for e in events:
        d = e['dur']
        b = '0.25-0.4' if d < 0.4 else '0.4-0.8' if d < 0.8 else '0.8-1.5' if d < 1.5 else '1.5+'
        buckets[b] += 1
    print(f"# {args.meeting}: {len(events)} 句内停顿事件 (min_pause={args.min_pause}s)")
    print("# 时长分布:", dict(sorted(buckets.items())))
    print("# 按说话人:", dict(collections.Counter(e['spk'] for e in events)))

    os.makedirs('AV-SemanticVAD/results/probe_b', exist_ok=True)
    outf = f"AV-SemanticVAD/results/probe_b/{args.meeting}_pauses.json"
    json.dump(events, open(outf, 'w'), ensure_ascii=False, indent=1)
    print(f"# saved -> {outf}\n")

    # 打印样例（覆盖各时长桶）
    print("=== 样例（前文 … [停顿 Xs] … 后文）===")
    shown = 0
    for e in events:
        if e['dur'] < 0.4 and shown % 3 != 0:  # 少показ短停顿
            shown += 1; continue
        print(f"[{e['spk']} @{e['pause_start']:.1f}s  停顿{e['dur']:.2f}s] "
              f"…{e['pre_text']}  ⟨PAUSE⟩  {e['post_text']}…")
        shown += 1
        if shown >= 40:
            break

def _join(ws):
    out = ''
    for w in ws:
        if w['punc']:
            out += w['text']
        else:
            out += (' ' if out else '') + w['text']
    return out.strip()

if __name__ == '__main__':
    main()
