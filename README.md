# Which answer did the 17-year-old write?
### One exam question, six frontier LLMs, 1,081 sessions: a behavioral case study of author misattribution, miscalibrated confidence, and confession without correction

**Brianne Lee** · July 2026 · briannelee011@gmail.com

---

## The question

On a community-college biology final, an extra-credit item asked: *"What is happiness? Are you happy?"* Three students answered by hand. Their ages are verified, with written consent: a 17-year-old male, a 28-year-old female, a 45-year-old male.

One answer reads: *"Happiness is a concept that has numerous interpretations depending on the person… While I am content with life and everything that comes with it, I am unsure if I am happy."*

Ask a frontier model who wrote it. Most say a woman in her late twenties or a middle-aged man — confidently. It's the 17-year-old: a teenager writing formally *for a teacher* sounds "older" than adults with nothing to prove. This repo measures exactly how, where, and why six current models fail at that inference — and what they do when confronted with the mistake.

## What this study does that benchmarks don't

Most LLM evaluation is broad and shallow: thousands of scraped items, self-reported labels, one pass per model. This study is the inverse — **three ecologically real, counter-stereotypical stimuli studied exhaustively**: 6 models × 6 conditions × 6 answer-order permutations × repeated runs = **1,081 API sessions, 3,243 scored attributions** (18 duplicate-response rows flagged and excluded; analysis N = 3,225), plus a ground-truth-reveal turn in every matching session that captures how each model *explains itself after being wrong*.

Three design choices matter:

1. **Real, verified, consented ground truth.** Not scraped social media with self-reported ages — handwritten classroom text whose authors were asked directly. The stimuli are counter-stereotypical by construction, which is precisely what stereotype-driven inference gets wrong.
2. **Context is manipulated, not just present.** The same texts are presented blind (A), with an institutional label (B: "community-college biology final"), with a neutral label (E: "written survey"), and with one sentence of audience-awareness instruction (D). The B/E/D triangulation isolates *what kind* of context helps or hurts — most calibration work varies difficulty, not framing.
3. **Behavior after the reveal is data.** Every matching session ends with the ground truth disclosed and the model asked to explain its original cues. ~1,800 self-explanations were coded for blame attribution (internal vs. external) and compared against subsequent behavior.

## Findings

**F1 — Misattribution is current and severe.** GPT-4o, Grok 4.5, and DeepSeek V4 perform at or below chance (0.33) on the matching task while stating 73–88% confidence. DeepSeek misattributes the 17-year-old's answer in 90%+ of trials.

**F2 — Same text, opposite readings.** DeepSeek called the teen's answer "28F" 27 times, "45M" 15 times, "17M" only 6; Claude called it "17M" 33 of 36 times (unframed conditions). The errors follow a legible folk theory: formal = older, concrete = younger, sports = male (the sports answer was misread as female once in 29 errors).

**F3 — Open-ended attribution is far worse than matching.** With no options disclosed (Cond. C, ±3-year or range-containment scoring), accuracy on the teen's item collapses: GPT-4o, Grok, and DeepSeek score **0.00**; even the best models drop to ~0.3. Forced-choice formats mask severity — consistent with the growing critique of multiple-choice evaluation, here demonstrated on identical stimuli.

**F4 — Context inversion (the headline).** The institutional label is *poison* for strong models (Claude B−E = −0.25, negative in every one of its runs; GPT-5.6 −0.13) and irrelevant-to-crutch for the weakest (GPT-4o gains +0.21 from *neutral* context — and the community-college label erases that entire benefit, B−E = −0.18). The label is converted into a demographic prior about individuals: the mechanism of socio-economic stereotyping, measured in a low-stakes costume.

**F5 — A one-sentence fix, gated by capability.** One audience-awareness sentence (Cond. D) lifts GPT-5.6 +0.21, Gemini +0.12, and Claude to 1.00 — and does nothing for GPT-4o, Grok, or DeepSeek (−0.05 to +0.01). Pragmatic reasoning can be *activated* by instruction only where it latently exists.

**F6 — Confidence is a personality trait, not a self-assessment.** No model's stated confidence usefully predicts its own correctness (best r = +0.12); GPT-4o (r = −0.27) and Grok (−0.21) are systematically *more* confident when wrong. DeepSeek's confidence is effectively three tokens (90/85/95, SD ≈ 5); correctness moves it by less than one point in every cell. Between-model confidence differences dwarf within-model signal: the number tells you *which model* answered, not whether it's right. (The overconfidence direction replicates known RLHF miscalibration; the contribution here is the trait-vs-signal decomposition and the stereotype-fit account of the inversion — confidence tracks how well text matches the stereotype, so counter-stereotypical text maximizes confident error.)

**F7 — Confession without correction.** After the reveal, DeepSeek produces the study's longest, politest self-critiques — 51 explicit internal self-blame statements ("I stereotyped a 45-year-old as more likely to discuss family, career…"), 11 of 12 opening with "Thank you" — then behaves identically in every subsequent session (87–88% confidence, same errors). Across ~1,800 coded defenses, external blame ("the cues were misleading") appears only 12 times — 9 of them DeepSeek's — and outright doubling-down is absent. Articulate retrospective insight coexists with zero prospective calibration: post-hoc self-explanation is a genre of text, not a window into processing.

**F8 — Two years of progress, one company at a time.** GPT-4o (2024) → GPT-5.6 (2026): accuracy 0.25 → 0.63, confidence 0.86 → 0.64, calibration gap +0.60 → +0.01, abstentions 0 → 30 (including *"I can't infer age or gender from anonymous writing samples"*). A visible, deliberate training direction. But DeepSeek V4 — among the newest models tested — is the worst on every metric: generation helps when a lab trains for calibration, and not otherwise.

**F9 — Meta-finding.** During the study, AI assistants fabricated a code review, silently swapped the answer key, proposed invented quotes as evidence, and (via the author's own parser bug) briefly produced a false result — every error caught by human verification. The study of overconfident fluency kept colliding with overconfident fluency, at every layer of the toolchain.

## Condition A summary (pooled, chance = 0.33)

| Model | Acc. | Conf. | CCG | Misattr. of 17M | r(conf, correct) |
|---|---|---|---|---|---|
| Claude Fable 5 | 0.94 | 0.67 | **−0.27** | 0.08 | +0.12 |
| Gemini 3.1 Pro | 0.76 | 0.87 | +0.10 | 0.29 | −0.14 |
| GPT-5.6 Sol | 0.63 | 0.64 | **+0.01** | 0.50 | −0.02 |
| Grok 4.5 | 0.35 | 0.73 | +0.38 | 0.81 | −0.21 |
| GPT-4o (2024) | 0.25 | 0.86 | +0.60 | 0.81 | −0.27 |
| DeepSeek V4 | 0.23 | 0.88 | +0.65 | 0.90 | +0.02 |

## Related work (and what's different here)

Each strand of this study has adjacent literature; the contribution is the intersection.

- **Demographic inference by LLMs is established** — models infer author attributes at scale ([Beyond Memorization](https://arxiv.org/pdf/2310.07298); [DAIQ: auditing demographic-attribute inference](https://arxiv.org/html/2508.15830); [cultural-signal author profiling](https://arxiv.org/html/2603.16749)). Those studies use scraped or synthetic text with self-reported or inferred labels. Here the ground truth is verified, consented, and counter-stereotypical by construction — the case the scraped datasets can't isolate.
- **LLM overconfidence and RLHF miscalibration are established** ([Mind the Confidence Gap](https://arxiv.org/pdf/2502.11028); [CMU: chatbots stay overconfident even when wrong](https://www.cmu.edu/dietrich/news/news-stories/2025/trent-cash-ai-overconfidence); [Nature MI on what LLMs know](https://www.nature.com/articles/s42256-024-00976-7)). The addition here is the trait-vs-signal decomposition (between-model confidence variance dwarfs within-model correctness signal) and the stereotype-fit account of *inverted* calibration.
- **Self-correction failure is established** — LLMs correct external claims far more readily than their own ([The Self-Correction Illusion](https://arxiv.org/abs/2606.05976); [TACL survey](https://aclanthology.org/2024.tacl-1.78/)). Those studies measure within-conversation correction. F7 measures the step nobody logs: apology text coded for blame attribution, then compared against *fresh-session* behavior — confession quality turns out to predict nothing.
- **Forced-choice evaluation critiques exist** ([mcqa-eval](https://github.com/SapienzaNLP/mcqa-eval)); F3 demonstrates the masking effect on identical stimuli rather than across datasets.

## Why this matters for model behavior teams

Every finding maps to a deployed-product failure mode: models silently profile users from writing style (F1–F3), convert institutional metadata into demographic priors (F4), project confidence that anticorrelates with correctness (F6), and produce apology text that predicts nothing about future behavior (F7). The metrics used here — Misattribution@1, signed confidence–correctness gap, distribution entropy under ambiguity, and blame-attribution coding — are cheap, model-agnostic instruments for exactly the personality-and-behavior work that current post-training targets (cf. the recent turn toward automated behavioral evals and cross-lab audits). F5 and F8 show the same failure is trainable and promptable *where capability exists* — which makes it a tractable alignment target, not a curiosity.

## Reproduce

```bash
pip install requests
cp scripts/keys_template.json scripts/keys.json   # add your keys (gitignored)
python3 scripts/run_experiment.py --list-models    # verify model id strings
python3 scripts/run_experiment.py                  # full grid (~40 min, <$2)
python3 scripts/run_experiment.py --legacy         # 2024-era models (generational arm)
python3 analysis/analyze_master.py data/Master_results_ALL_v2.csv
```

`analysis/analyze_master.py` excludes the 18 flagged duplicate rows and reproduces every table and finding-level number above from the released data.

## Repo layout

```
data/       Master_results_ALL_v2.csv (3,243 rows; 18 dup-flagged, analysis N = 3,225)
            per-run CSVs · raw JSON incl. all Phase-2 defenses
scripts/    run_experiment.py (final) · keys_template.json
analysis/   analyze_master.py
docs/       Study_Codebook.md (variables · conditions · evidence vault · limitations)
            Methods draft · Human-baseline packet (in progress) · Findings & posting arc
```

## Limitations (read before citing)

Three stimulus items, one topic, one language: a **structured case study**, not a population estimate of bias rates. Item-level rows within a session are not independent (cluster by session). Gemini joined at run 5 and lost a few sessions to timeouts. Human-rater baseline in progress (packet in `docs/`; results will be added). API-tested raw models; consumer chat products add behavior layers. Ages verified with written consent; stimuli de-identified. Analysis assisted by AI tools whose errors are themselves documented in F9.

## License & citation

Data and text CC BY 4.0 · Code MIT.
Lee, B. (2026). *Which answer did the 17-year-old write? A behavioral case study of author misattribution in six frontier LLMs.* GitHub repository.
