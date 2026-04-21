"""
Generate synthetic labeled data so you can dry-run the whole pipeline without
spending API budget or waiting for trace generation.

Writes files to the same paths the real pipeline uses:
  data_out/raw_traces.jsonl
  data_out/spans.jsonl
  data_out/spans_seed_labeled.jsonl
  data_out/spans_judge_labeled.jsonl

After running this you can immediately run:
  python -m model.train_segmenter
  python -m model.evaluate_segmenter

The synthetic spans use clear vocabulary signals, so the trained model should
hit 90%+ F1. That tells you your training pipeline works; it says nothing about
how well the real segmenter will do on real reasoning traces.

Usage:
    python -m scripts.make_synthetic_data --n-spans 600
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_TRACES_PATH, SPANS_PATH, SEED_LABELS_PATH, JUDGE_LABELS_PATH, SEED,
)


STRATEGY_TEMPLATES = [
    "Let me try {method} for this problem.",
    "I'll use {method} to approach this.",
    "First, I need to figure out {goal}.",
    "Instead, let me try a different approach with {method}.",
    "Actually, {method} might work better here.",
    "I think {method} is the right way to solve this.",
    "Let me reconsider. Maybe {method} would be cleaner.",
    "Another approach: use {method}.",
    "Next, I should {goal}.",
    "Therefore, I will apply {method}.",
]

EXECUTION_TEMPLATES = [
    "Substituting gives {eq1}. Simplifying: {eq2}.",
    "Computing {eq1} yields {eq2}.",
    "{eq1} equals {eq2} equals {eq3}.",
    "Multiplying both sides: {eq1}. So {eq2}.",
    "Adding the terms: {eq1} + {eq2} = {eq3}.",
    "Expanding the expression: {eq1}. This gives {eq2}.",
    "Plugging in the values: {eq1} = {eq2}.",
    "The derivative is {eq1}. Setting it to zero: {eq2}.",
    "Squaring both sides: {eq1}. Therefore {eq2}.",
    "Dividing by {eq1}, we get {eq2}.",
]

REFLECTION_TEMPLATES = [
    "Wait, let me double-check this. Did I compute {eq1} correctly?",
    "Hmm, let me verify the previous step. {eq1} should equal {eq2}.",
    "Actually, I think I made a sign error. Let me re-examine {eq1}.",
    "Let me check: does {eq1} really give {eq2}?",
    "But wait, I need to verify this makes sense. Is {eq1} consistent?",
    "Let me pause and check my work on {eq1}.",
    "Did I account for all cases? What about {goal}?",
    "On second thought, let me confirm that {eq1} is correct.",
]

METHODS = [
    "substitution", "elimination", "factoring", "the quadratic formula",
    "completing the square", "a graphical approach", "induction",
    "proof by contradiction", "modular arithmetic", "the Pythagorean theorem",
    "a change of variables", "integration by parts", "the chain rule",
]
GOALS = [
    "the total cost", "the remaining pieces", "the maximum value",
    "the equation of the line", "the intersection point", "the common factor",
    "the critical points", "the leading coefficient",
]
EQS = [
    "3x + 5 = 14", "x^2 - 4 = 0", "y = 2x + 7", "2(y+2) + y = 14",
    "x = 3", "4y = 8", "f'(x) = 0", "6*7 - 1 = 41", "3(4) + 2 = 14",
    "a^2 + b^2 = c^2", "log(x) + log(y)", "(x+1)(x-2)",
]


def make_span(label: str, rng: random.Random) -> str:
    if label == "strategy":
        tpl = rng.choice(STRATEGY_TEMPLATES)
        return tpl.format(method=rng.choice(METHODS), goal=rng.choice(GOALS))
    if label == "reflection":
        tpl = rng.choice(REFLECTION_TEMPLATES)
        return tpl.format(eq1=rng.choice(EQS), eq2=rng.choice(EQS), goal=rng.choice(GOALS))
    # execution
    tpl = rng.choice(EXECUTION_TEMPLATES)
    return tpl.format(
        eq1=rng.choice(EQS), eq2=rng.choice(EQS), eq3=rng.choice(EQS),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spans", type=int, default=600)
    ap.add_argument("--seed-frac", type=float, default=0.1,
                    help="Fraction routed to the hand-labeled seed file")
    args = ap.parse_args()

    rng = random.Random(SEED)

    # Build synthetic traces. Each trace is a mix of spans.
    rows = []
    trace_id = 0
    while len(rows) < args.n_spans:
        trace_id += 1
        # One trace: 1 strategy, 2-3 execution, maybe 1 reflection
        pattern = ["strategy"]
        pattern += ["execution"] * rng.randint(2, 3)
        if rng.random() < 0.4:
            pattern.append("reflection")
            pattern += ["execution"] * rng.randint(0, 2)
        if rng.random() < 0.3:
            pattern.append("strategy")
            pattern += ["execution"] * rng.randint(1, 2)

        char_offset = 0
        trace_text_parts = []
        for span_idx, lab in enumerate(pattern):
            text = make_span(lab, rng)
            start = char_offset
            end = start + len(text)
            rows.append({
                "trace_id": f"syn{trace_id}",
                "span_idx": span_idx,
                "start_char": start,
                "end_char": end,
                "text": text,
                "trigger": None,
                "correct": rng.random() < 0.6,
                "problem": f"synthetic problem {trace_id}",
                "label": lab,
            })
            trace_text_parts.append(text)
            char_offset = end + 1

    rng.shuffle(rows)
    rows = rows[:args.n_spans]

    # --- raw_traces.jsonl (just so downstream scripts don't crash) ---
    RAW_TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_TRACES_PATH.open("w") as f:
        by_trace = {}
        for r in rows:
            by_trace.setdefault(r["trace_id"], []).append(r)
        for tid, items in by_trace.items():
            items.sort(key=lambda x: x["span_idx"])
            thinking = " ".join(x["text"] for x in items)
            f.write(json.dumps({
                "trace_id": tid,
                "problem": items[0]["problem"],
                "gold": "",
                "raw_output": thinking,
                "thinking": thinking,
                "correct": items[0]["correct"],
            }) + "\n")

    # --- spans.jsonl ---
    with SPANS_PATH.open("w") as f:
        for r in rows:
            r2 = {k: v for k, v in r.items() if k != "label"}
            f.write(json.dumps(r2) + "\n")

    # --- split into seed (hand-labeled) and judge-labeled ---
    n_seed = max(30, int(len(rows) * args.seed_frac))
    seed_rows = rows[:n_seed]
    judge_rows = rows[n_seed:]

    with SEED_LABELS_PATH.open("w") as f:
        for r in seed_rows:
            f.write(json.dumps(r) + "\n")

    with JUDGE_LABELS_PATH.open("w") as f:
        for r in judge_rows:
            # Simulate judge noise: 8% wrong labels.
            if rng.random() < 0.08:
                r = dict(r)
                other = [l for l in ["strategy", "execution", "reflection"] if l != r["label"]]
                r["label"] = rng.choice(other)
            r["judge_raw"] = r["label"]
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} synthetic spans across {trace_id} traces.")
    print(f"  Seed (hand-labeled): {len(seed_rows)}")
    print(f"  Judge-labeled:       {len(judge_rows)} (8% synthetic noise)")
    print(f"\nNow run:")
    print(f"  python -m model.train_segmenter")
    print(f"  python -m model.evaluate_segmenter")


if __name__ == "__main__":
    main()
