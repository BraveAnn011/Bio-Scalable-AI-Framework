#!/usr/bin/env python3
"""Reproduce every table and finding-level number in the README.
Usage: python3 analyze_master.py ../data/Master_results_ALL_v2.csv"""
import csv, sys, re, statistics as st
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "Master_results_ALL_v2.csv"
rows = list(csv.DictReader(open(path)))
n_raw = len(rows)
rows = [r for r in rows if r.get('dup_response_flag') not in ('1', 'True', 'true')]
print(f"rows: {n_raw} raw, {len(rows)} after excluding dup-flagged (analysis N)")

MODELS = ['Claude Fable 5', 'Gemini 3.1 Pro', 'GPT-5.6 Sol',
          'Grok 4.5', 'GPT-4o legacy', 'DeepSeek V4']
LABELS = ('17M', '28F', '45M')
TRUE_AGE = {'17M': 17, '28F': 28, '45M': 45}

def cell(m, c):
    return [r for r in rows if r['model'] == m and r['condition'] == c
            and r['model_guess'] in LABELS]

def acc(sub):
    return sum(1 for r in sub if r['correct'] == '1') / len(sub) if sub else float('nan')

def conf(sub):
    v = [int(r['stated_confidence']) for r in sub if r['stated_confidence']]
    return sum(v) / len(v) / 100 if v else float('nan')

def score_open_age(guess, gt_label):
    """Cond C: correct if range contains true age, or point within +/-3 yrs."""
    age = TRUE_AGE[gt_label]
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)", guess)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return int(lo <= age <= hi)
    m = re.search(r"\d+", guess)
    return int(abs(int(m.group()) - age) <= 3) if m else None

print("\n== POPULATION ==")
for m in MODELS:
    sub = [r for r in rows if r['model'] == m]
    print(f"{m:16s} obs={len(sub):4d} sessions={len(sub)//3:3d}")

print("\n== CONDITION A (pooled; chance=0.33) — README table ==")
for m in MODELS:
    s = cell(m, 'A')
    c17 = [r for r in s if r['item'] == 'concept']
    # r(conf, correct) over matching conditions A/B/D/E
    pairs = [(int(r['stated_confidence']), int(r['correct'])) for r in rows
             if r['model'] == m and r['condition'] in 'ABDE'
             and r['model_guess'] in LABELS and r['stated_confidence']]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    rr = (sum((x - st.mean(xs)) * (y - st.mean(ys)) for x, y in pairs)
          / len(pairs) / (sx * sy)) if sx and sy else 0
    print(f"{m:16s} acc={acc(s):.2f} conf={conf(s):.2f} CCG={conf(s)-acc(s):+.2f} "
          f"Mis@1(17M)={1-acc(c17):.2f} r={rr:+.2f} n={len(s)}")

print("\n== F3: CONDITION C, teen item (concept), range/±3-year scoring ==")
for m in MODELS:
    rs = [r for r in rows if r['model'] == m and r['condition'] == 'C'
          and r['item'] == 'concept' and r['model_guess']]
    sc = [score_open_age(r['model_guess'], r['ground_truth']) for r in rs]
    sc = [s for s in sc if s is not None]
    if sc:
        print(f"{m:16s} acc={sum(sc)/len(sc):.2f} n={len(sc)}")

print("\n== F4/F5: CONTEXT EFFECTS (accuracy deltas) ==")
print(f"{'model':16s} {'B-A':>6s} {'E-A':>6s} {'B-E':>6s} {'D-A':>6s}")
for m in MODELS:
    a = {c: acc(cell(m, c)) for c in 'ABDE'}
    print(f"{m:16s} {a['B']-a['A']:+6.2f} {a['E']-a['A']:+6.2f} "
          f"{a['B']-a['E']:+6.2f} {a['D']-a['A']:+6.2f}")

print("\n== F6: confidence as trait — mean & SD per model (conds A/B/D/E) ==")
for m in MODELS:
    v = [int(r['stated_confidence']) for r in rows
         if r['model'] == m and r['condition'] in 'ABDE'
         and r['model_guess'] in LABELS and r['stated_confidence']]
    ok = [int(r['stated_confidence']) for r in rows
          if r['model'] == m and r['condition'] in 'ABDE' and r['correct'] == '1'
          and r['model_guess'] in LABELS and r['stated_confidence']]
    wr = [int(r['stated_confidence']) for r in rows
          if r['model'] == m and r['condition'] in 'ABDE' and r['correct'] == '0'
          and r['model_guess'] in LABELS and r['stated_confidence']]
    print(f"{m:16s} mean={st.mean(v):5.1f} SD={st.pstdev(v):4.1f} "
          f"when-right={st.mean(ok):5.1f} when-wrong={st.mean(wr):5.1f}")

print("\n== F7: external-blame language in Phase-2 defenses ==")
BLAME = re.compile(r'(flawless|deceptive|misleading|superficial cue|the text itself'
                   r'|stand by|maintain my|still believe|remain confident)', re.I)
tot_d = tot_h = 0
for m in MODELS:
    d = [r for r in rows if r['model'] == m and r['phase2_defense']]
    h = sum(1 for r in d if BLAME.search(r['phase2_defense']))
    tot_d += len(d); tot_h += h
    print(f"{m:16s} {h}/{len(d)}")
print(f"{'TOTAL':16s} {tot_h}/{tot_d}")

print("\n== CONDITION P: entropy & prob on truth ==")
for m in MODELS:
    s = [r for r in rows if r['model'] == m and r['condition'] == 'P'
         and r['correct'] != '']
    if not s:
        continue
    e = [float(r['entropy_bits']) for r in s if r['entropy_bits']]
    p = [float(r['prob_on_truth']) for r in s if r['prob_on_truth']]
    print(f"{m:16s} top1={acc(s):.2f} p(truth)={sum(p)/len(p):5.1f}% "
          f"entropy={sum(e)/len(e):.2f}/1.58 n={len(s)}")

print("\n== F2: 17M item attributions (pooled A+E) ==")
for m in MODELS:
    cnt = Counter(r['model_guess'] for r in rows
                  if r['model'] == m and r['condition'] in 'AE'
                  and r['item'] == 'concept' and r['model_guess'] in LABELS)
    print(f"{m:16s} {dict(cnt)}")

print("\n== F8: abstentions per model ==")
for m in MODELS:
    ab = sum(1 for r in rows if r['model'] == m and r['abstained'] == '1')
    print(f"{m:16s} {ab}")
