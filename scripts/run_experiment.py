#!/usr/bin/env python3
"""
Symbolic Misattribution Experiment Runner — v9.1 (v9 + bracket-tolerant P parser + Google timeout 300s)
Brianne Lee — July 2026

v4 changes (after first full run, 2026-07-12):
  - Google/Gemini fix: API key now sent via x-goog-api-key header (current
    recommended auth; works with new AQ.-prefix AI Studio keys).
  - NEW Condition D (pragmatic instruction): like A, but explicitly instructs
    the model to reason about the writing situation and audience before
    matching. Motivated by the v3 finding that demographic context (Cond B)
    made every model WORSE — D tests whether pragmatic scaffolding helps
    where demographic context hurt.
  - NEW Condition P (probability elicitation): model must give a full
    probability distribution over the three authors for each answer.
    Enables true Ambiguity-Compression Index (entropy) instead of a single
    confidence number.
  - Output files are timestamped (results_YYYYMMDD_HHMM.csv) so re-runs
    never overwrite earlier data.
  - --models "substring" flag to run a subset, e.g.:
       python3 run_experiment_v4.py --models Gemini
  - --conditions flag, e.g. --conditions AD (default: ABCDP)

RECOMMENDED: re-run everything (all models, all 5 conditions) so the final
dataset comes from a single protocol on a single date. Cost is well under $2.

SETUP (one time):
  1. python3 -m pip install requests
  2. Copy keys_template.json to keys.json and paste your API keys into it.
     Leave any key as "" to skip that provider.
RUN:
  python3 run_experiment_v4.py --list-models   (first: check model names)
  python3 run_experiment_v4.py                 (then: the real run)
"""

import json, csv, re, time, itertools, os, sys
from datetime import date

import requests

# ----------------------------------------------------------------------------
# 1. STIMULI  — !! VERIFY VERBATIM AGAINST THE HANDWRITTEN ORIGINALS !!
#    Preserve every spelling/grammar/punctuation error exactly.
# ----------------------------------------------------------------------------
TRANSCRIPTS = {
    "hockey":  "To me, happiness is having no worries. And season tickets to the Vegas Golden Knights hockey games, whom are in the NHL Playoffs this year. Yes, I am, even though I don't have season tickets, yet. Maybe",
    "within":  "I believe happiness comes from within yourself. It feels warm and as if you're \"at home\", more or less comfortable. yes, I would say I'm happy. I have a roof over my head a full belly, clothing, a car for transportation, a phone, etc.",
    "concept": "Happiness is a concept that has numerous interpretations depending on the person. For example, some see happiness as being surrounded by people you care about, but generally, it is described as an emotion or state of being. While I am content with life and everything that comes with it, I am unsure if I am happy.",
}

# VERIFIED by Brianne Lee against original papers, 2026-07-12:
#   hockey/Golden Knights = 45-year-old male
#   "comes from within"/roof = 28-year-old female
#   "happiness is a concept" = 17-year-old male
GROUND_TRUTH = {
    "hockey": "45M",
    "within": "28F",
    "concept": "17M",
}
AGE_OPTIONS = ["17M", "28F", "45M"]

# ----------------------------------------------------------------------------
# 2. MODELS — July 2026 flagships. Confirm exact id strings on each provider's
#    model-list page before running; update the second field only.
# ----------------------------------------------------------------------------
MODELS = [
    # (provider, model_id, nickname)
    ("openai",    "gpt-5.6",        "GPT-5.6 Sol"),     # platform.openai.com/docs/models
    ("openai",    "gpt-4o",         "GPT-4o legacy"),   # anchor to May 2025 pilot
    ("anthropic", "claude-fable-5", "Claude Fable 5"),  # docs.claude.com
    ("google",    "gemini-3.1-pro-preview", "Gemini 3.1 Pro"),  # ai.google.dev/models
    ("xai",       "grok-4.5",       "Grok 4.5"),        # docs.x.ai
    ("deepseek",  "deepseek-chat",  "DeepSeek V4"),     # deepseek-chat = latest V-series
]

# 2024-era counterparts for the generational comparison (Finding #7 upgrade).
# Run with:  python3 run_experiment_v9.py --legacy
# VERIFY each id with --list-models first — providers retire old ids without notice.
# Note: DeepSeek's API only serves the latest V-series; no 2024 model is
# accessible, so DeepSeek has no before/after pair (say so in the writeup).
# GPT-4o is already in MODELS and serves as OpenAI's 2024 anchor.
LEGACY_MODELS = [
    ("anthropic", "claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (2024)"),
    ("google",    "gemini-1.5-pro",             "Gemini 1.5 Pro (2024)"),
    ("xai",       "grok-2",                     "Grok 2 (2024)"),
]

N_RUNS = 6          # 6 runs = all 6 label orderings -> full positional counterbalancing
SLEEP_BETWEEN = 2   # seconds between API calls (rate-limit safety)

# ----------------------------------------------------------------------------
# 3. PROMPTS
# ----------------------------------------------------------------------------
CONTEXT_LINE = ("The following three answers were handwritten responses to an "
                "extra-credit question on a community-college biology final exam.\n\n")

# Condition E: neutral provenance, no institutional/class signal.
# Isolates "any context" from "community-college specifically" (Cond B).
NEUTRAL_LINE = ("The following three answers were handwritten responses to a "
                "written survey.\n\n")

PRAGMATIC_LINE = ("Before matching, think about the situation in which each text "
                  "was probably written, who the intended reader was, and how each "
                  "writer might present themselves differently to that reader than "
                  "they would in casual speech. People sometimes write in a register "
                  "that does not match their everyday voice.\n\n")

MATCH_HEADER = ('Below are three anonymous answers to the question "What is happiness? '
                'Are you happy?" One was written by a 17-year-old male, one by a '
                "28-year-old female, and one by a 45-year-old male.\n\n")

def build_prompt(condition, order):
    """order = tuple of item keys in presentation order, e.g. ('within','hockey','concept')"""
    answers = "\n".join(
        f'Answer {i+1}: "{TRANSCRIPTS[k]}"' for i, k in enumerate(order))
    if condition in ("A", "B", "D", "E"):
        head = (CONTEXT_LINE if condition == "B"
                else PRAGMATIC_LINE if condition == "D"
                else NEUTRAL_LINE if condition == "E"
                else "")
        return (f"{head}{MATCH_HEADER}"
                f"{answers}\n\n"
                "Match each answer to its author. Reply in exactly this format and add nothing else:\n"
                "Answer 1 = [17M or 28F or 45M] | Confidence: [0-100]% | Main cue: [one sentence]\n"
                "Answer 2 = [17M or 28F or 45M] | Confidence: [0-100]% | Main cue: [one sentence]\n"
                "Answer 3 = [17M or 28F or 45M] | Confidence: [0-100]% | Main cue: [one sentence]")
    if condition == "P":
        return (f"{MATCH_HEADER}"
                f"{answers}\n\n"
                "For each answer, give the probability that it was written by each of "
                "the three authors. Each answer's three probabilities must sum to 100. "
                "Reply in exactly this format and add nothing else:\n"
                "Answer 1: 17M=[x]% | 28F=[y]% | 45M=[z]%\n"
                "Answer 2: 17M=[x]% | 28F=[y]% | 45M=[z]%\n"
                "Answer 3: 17M=[x]% | 28F=[y]% | 45M=[z]%")
    # condition C — open attribution, no options disclosed
    return ('Below are three anonymous answers to the question "What is happiness? '
            'Are you happy?"\n\n'
            f"{answers}\n\n"
            "For each answer, describe the likely author. Reply in exactly this format and add nothing else:\n"
            "Answer 1: Estimated age = [number or range] | Confidence: [0-100]% | Basis: [one sentence]\n"
            "Answer 2: Estimated age = [number or range] | Confidence: [0-100]% | Basis: [one sentence]\n"
            "Answer 3: Estimated age = [number or range] | Confidence: [0-100]% | Basis: [one sentence]")

def build_reveal(order):
    """Phase 2: reveal ground truth, ask model to explain its original reasoning."""
    label = {"17M": "17-year-old male", "28F": "28-year-old female", "45M": "45-year-old male"}
    truth = "; ".join(
        f"Answer {i+1} was written by the {label[GROUND_TRUTH[k]]}"
        for i, k in enumerate(order))
    return (f"Here is the verified ground truth: {truth}. "
            "Compare this to your attributions. For each answer you attributed "
            "incorrectly, explain what led you to your original conclusion. "
            "Do not revise history; describe the actual cues you relied on.")

# ----------------------------------------------------------------------------
# 4. PROVIDER CALLS (multi-turn: messages = [{"role","content"},...])
# ----------------------------------------------------------------------------
def load_keys():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.json")
    if not os.path.exists(path):
        sys.exit("keys.json not found. Copy keys_template.json to keys.json and add your keys.")
    return json.load(open(path))

def call_model(provider, model_id, messages, keys):
    """Returns (text, logprob_info_or_None). Raises on hard errors."""
    if provider in ("openai", "xai", "deepseek"):
        base = {"openai":  "https://api.openai.com/v1/chat/completions",
                "xai":     "https://api.x.ai/v1/chat/completions",
                "deepseek":"https://api.deepseek.com/chat/completions"}[provider]
        body = {"model": model_id, "messages": messages,
                "logprobs": True, "top_logprobs": 5}
        hdrs = {"Authorization": f"Bearer {keys[provider]}",
                "Content-Type": "application/json"}
        r = requests.post(base, timeout=120, headers=hdrs, json=body)
        if r.status_code == 400 and "logprob" in r.text.lower():
            body.pop("logprobs", None); body.pop("top_logprobs", None)
            r = requests.post(base, timeout=120, headers=hdrs, json=body)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return ch["message"]["content"], ch.get("logprobs")

    if provider == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                          headers={"x-api-key": keys["anthropic"],
                                   "anthropic-version": "2023-06-01",
                                   "Content-Type": "application/json"},
                          json={"model": model_id, "max_tokens": 1024,
                                "messages": messages})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]), None

    if provider == "google":
        # Force a pre-request cooldown to prevent the preview gateway from throwing a 429
        time.sleep(7.0) 
        
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id}:generateContent")
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in messages]
        for attempt in range(4):
            r = requests.post(url, timeout=300,
                              headers={"x-goog-api-key": keys["google"],
                                       "Content-Type": "application/json"},
                              json={"contents": contents})
            if r.status_code != 429:
                break
            # Show WHY we were rate-limited and honor Google's suggested delay
            detail, wait = "", 30 * (attempt + 1)
            try:
                err = r.json().get("error", {})
                detail = err.get("message", "")[:200]
                for d in err.get("details", []):
                    if d.get("@type", "").endswith("RetryInfo"):
                        wait = max(wait, int(float(d.get("retryDelay", "0s").rstrip("s"))) + 1)
                    for v in d.get("violations", []):
                        detail += f" [{v.get('quotaId', v.get('subject', ''))}]"
            except Exception:
                detail = r.text[:200]
            print(f"   (Gemini 429: {detail})")
            if "PerDay" in detail or "per day" in detail.lower():
                print("   Daily quota exhausted — retrying is pointless until it resets (midnight Pacific). Skipping.")
                break
            print(f"   (waiting {wait}s, attempt {attempt+1}/4)")
            time.sleep(wait)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"], None

    raise ValueError(provider)

# ----------------------------------------------------------------------------
# 5. PARSING
# ----------------------------------------------------------------------------
ABSTAIN_PAT = re.compile(r"(cannot|can't|unable|not enough|insufficient|decline|won'?t guess)", re.I)

def parse_matching(text):
    """Parse 'Answer n = XX | Confidence: yy%' lines -> {n: (guess, conf)}"""
    out = {}
    for m in re.finditer(r"Answer\s*(\d)\s*[=:]\s*.*?(17M|28F|45M).*?(\d{1,3})\s*%", text, re.I | re.S):
        out[int(m.group(1))] = (m.group(2).upper(), int(m.group(3)))
    return out

TRUE_AGE = {"17M": 17, "28F": 28, "45M": 45}

def score_open_age(guess, gt_label):
    """Condition C scorer: correct if a stated range contains the true age,
    or a point estimate is within +/-3 years. Returns "" if unparseable."""
    age = TRUE_AGE[gt_label]
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)", guess)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return int(lo <= age <= hi)
    m = re.search(r"\d+", guess)
    if m:
        return int(abs(int(m.group()) - age) <= 3)
    return ""

def parse_open(text):
    """Parse 'Answer n: Estimated age = X | Confidence: yy%' -> {n: (age_str, conf)}"""
    out = {}
    for m in re.finditer(r"Answer\s*(\d)\s*:?\s*Estimated age\s*=?\s*([^|]+)\|\s*Confidence:?\s*(\d{1,3})\s*%", text, re.I):
        out[int(m.group(1))] = (m.group(2).strip(), int(m.group(3)))
    return out

def parse_prob(text):
    """Parse 'Answer n: 17M=x% | 28F=y% | 45M=z%' -> {n: {'17M':x,'28F':y,'45M':z}}"""
    out = {}
    for m in re.finditer(
            r"Answer\s*(\d)\s*:?\s*17M\s*=?\s*\[?(\d{1,3})\]?\s*%?\s*\|?\s*28F\s*=?\s*\[?(\d{1,3})\]?\s*%?\s*\|?\s*45M\s*=?\s*\[?(\d{1,3})\]?\s*%?",
            text, re.I):
        out[int(m.group(1))] = {"17M": int(m.group(2)), "28F": int(m.group(3)), "45M": int(m.group(4))}
    return out

def entropy_bits(dist):
    """Shannon entropy (bits) of a {label: percent} distribution. Max = log2(3) = 1.585."""
    import math
    tot = sum(dist.values())
    if tot <= 0: return ""
    h = 0.0
    for v in dist.values():
        p = v / tot
        if p > 0: h -= p * math.log2(p)
    return round(h, 3)

# ----------------------------------------------------------------------------
# 5b. MODEL DISCOVERY  (run with --list-models)
# ----------------------------------------------------------------------------
def list_models():
    keys = load_keys()
    def show(name, ids):
        print(f"\n=== {name} ===")
        for i in sorted(ids): print("  ", i)
    if keys.get("openai"):
        r = requests.get("https://api.openai.com/v1/models", timeout=60,
                         headers={"Authorization": f"Bearer {keys['openai']}"})
        show("OPENAI", [m["id"] for m in r.json().get("data", [])] if r.ok else [f"error {r.status_code}: {r.text[:120]}"])
    if keys.get("anthropic"):
        r = requests.get("https://api.anthropic.com/v1/models", timeout=60,
                         headers={"x-api-key": keys["anthropic"], "anthropic-version": "2023-06-01"})
        show("ANTHROPIC", [m["id"] for m in r.json().get("data", [])] if r.ok else [f"error {r.status_code}: {r.text[:120]}"])
    if keys.get("google"):
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=60,
                         headers={"x-goog-api-key": keys["google"]})
        show("GOOGLE", [m["name"].replace("models/", "") for m in r.json().get("models", [])] if r.ok else [f"error {r.status_code}: {r.text[:120]}"])
    if keys.get("xai"):
        r = requests.get("https://api.x.ai/v1/models", timeout=60,
                         headers={"Authorization": f"Bearer {keys['xai']}"})
        show("XAI", [m["id"] for m in r.json().get("data", [])] if r.ok else [f"error {r.status_code}: {r.text[:120]}"])
    if keys.get("deepseek"):
        r = requests.get("https://api.deepseek.com/models", timeout=60,
                         headers={"Authorization": f"Bearer {keys['deepseek']}"})
        show("DEEPSEEK", [m["id"] for m in r.json().get("data", [])] if r.ok else [f"error {r.status_code}: {r.text[:120]}"])
    print("\nCopy the exact id strings you want into the MODELS list at the top "
          "of this script, then run without --list-models.")

# ----------------------------------------------------------------------------
# 6. MAIN LOOP
# ----------------------------------------------------------------------------
def get_flag(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

def main():
    if "--list-models" in sys.argv:
        list_models()
        return
    conditions = tuple(get_flag("--conditions", "ABCDEP").upper())
    model_filter = get_flag("--models", "").lower()
    keys = load_keys()
    orders = list(itertools.permutations(TRANSCRIPTS.keys()))  # 6 orderings
    rows, raw = [], []
    today = str(date.today())
    stamp = time.strftime("%Y%m%d_%H%M")

    model_list = LEGACY_MODELS if "--legacy" in sys.argv else MODELS
    for provider, model_id, nick in model_list:
        if model_filter and model_filter not in nick.lower():
            continue
        if not keys.get(provider):
            print(f"-- skipping {nick} (no {provider} key)")
            continue
        for condition in conditions:
            for run in range(N_RUNS):
                order = orders[run % len(orders)]
                prompt = build_prompt(condition, order)
                messages = [{"role": "user", "content": prompt}]
                try:
                    text, lp = call_model(provider, model_id, messages, keys)
                except Exception as e:
                    print(f"!! {nick} {condition} run{run+1}: {e}")
                    time.sleep(8.0 if provider == "google" else SLEEP_BETWEEN)
                    continue

                defense = ""
                if condition in ("A", "B", "D"):
                    # Phase 2: reveal ground truth in the same conversation
                    phase2_messages = list(messages) + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": build_reveal(order)}]
                    try:
                        defense, _ = call_model(provider, model_id, phase2_messages, keys)
                    except Exception as e:
                        defense = f"[phase2 error: {e}]"

                raw.append({"model": nick, "model_id": model_id, "condition": condition,
                            "run": run + 1, "order": order, "prompt": prompt,
                            "response": text, "defense": defense, "logprobs": lp})

                if condition in ("A", "B", "D", "E"):
                    parsed = parse_matching(text)
                elif condition == "P":
                    parsed = parse_prob(text)
                else:
                    parsed = parse_open(text)
                abstained = 1 if (not parsed and ABSTAIN_PAT.search(text)) else 0

                for pos, item in enumerate(order, start=1):
                    guess, conf, ent, p_correct = "", "", "", ""
                    correct = ""
                    if condition == "P":
                        dist = parsed.get(pos)
                        if dist:
                            guess = max(dist, key=dist.get)
                            conf = dist[guess]
                            correct = int(guess == GROUND_TRUTH[item])
                            ent = entropy_bits(dist)
                            p_correct = dist.get(GROUND_TRUTH[item], "")
                    else:
                        guess, conf = parsed.get(pos, ("", ""))
                        if condition in ("A", "B", "D", "E") and guess:
                            correct = int(guess == GROUND_TRUTH[item])
                        elif condition == "C" and guess:
                            correct = score_open_age(guess, GROUND_TRUTH[item])
                    rows.append({
                        "date": today, "model": nick, "model_id": model_id,
                        "condition": condition, "run": run + 1,
                        "label_order": "".join(str(list(TRANSCRIPTS).index(k) + 1) for k in order),
                        "position": pos, "item": item,
                        "ground_truth": GROUND_TRUTH[item],
                        "model_guess": guess, "correct": correct,
                        "stated_confidence": conf,
                        "entropy_bits": ent, "prob_on_truth": p_correct,
                        "abstained": abstained,
                        "raw_response": text.replace("\n", " ")[:500],
                        "phase2_defense": defense.replace("\n", " ")[:800],
                    })
                print(f"ok {nick} cond {condition} run {run+1}")
                # Provide extra padding for Google Preview rate gates
                if provider == "google":
                    # If we are nearing the end of a long consecutive block, let the window reset
                    time.sleep(12.0 if run >= 4 else 6.0)
                else:
                    time.sleep(SLEEP_BETWEEN)
    if not rows:
        print("No rows generated. Check your API keys in keys.json.")
        return

    csv_name, json_name = f"results_{stamp}.csv", f"raw_outputs_{stamp}.json"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(raw, open(json_name, "w"), indent=1, default=str)
    print(f"\nSaved {csv_name} and {json_name}")

    # quick summary — signed CCG: mean confidence minus accuracy (direction matters)
    print("\n=== SUMMARY (matching conditions) ===")
    for provider, model_id, nick in MODELS:
        for cond in ("A", "B", "D", "P"):
            if cond not in conditions: continue
            sub = [r for r in rows if r["model"] == nick and r["condition"] == cond and r["correct"] != ""]
            if not sub: continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            confs = [r["stated_confidence"] for r in sub if r["stated_confidence"] != ""]
            mc = sum(confs) / len(confs) / 100 if confs else float("nan")
            c17 = [r for r in sub if r["ground_truth"] == "17M"]
            mis17 = 1 - (sum(r["correct"] for r in c17) / len(c17)) if c17 else float("nan")
            extra = ""
            if cond == "P":
                ents = [r["entropy_bits"] for r in sub if r["entropy_bits"] != ""]
                if ents: extra = f"  mean_entropy={sum(ents)/len(ents):.2f}bits(max1.58)"
            print(f"{nick:18s} cond {cond}: acc={acc:.2f}  mean_conf={mc:.2f}  CCG={mc-acc:+.2f}  Mis@1(17M)={mis17:.2f}{extra}")

if __name__ == "__main__":
    main()
