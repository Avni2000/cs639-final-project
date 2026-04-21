"""
Use an LLM judge to label spans at scale.

Flow:
  1. Load the small hand-labeled seed set. Use it as few-shot examples.
  2. Iterate over the full span pool (spans.jsonl). For each one, ask the judge
     to classify it as strategy / execution / (reflection).
  3. Write out a JSONL of judge-labeled spans that we can train on.

Runs offline, once. At inference time the distilled segmenter replaces the judge.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m data.llm_judge_labeler --limit 3000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LABELS, USE_META, SPANS_PATH, SEED_LABELS_PATH, JUDGE_LABELS_PATH,
    JUDGE_PROVIDER, JUDGE_MODEL_ANTHROPIC, JUDGE_MODEL_OPENAI,
)


# ----------------- Prompt construction -----------------

def _label_defs() -> str:
    base = (
        "- strategy: the model is choosing, comparing, or revisiting an approach. "
        "Phrases like 'let me try', 'I will use', 'instead', 'another approach'.\n"
        "- execution: the model is carrying out a plan. Algebra, arithmetic, following "
        "a previously stated approach with no change of direction.\n"
    )
    if USE_META:
        base += (
            "- reflection: the model is checking, verifying, or doubting its own reasoning. "
            "Phrases like 'wait let me verify', 'did I mess up', 'does this make sense'.\n"
        )
    return base


def _build_fewshot(seed_path: Path, k: int = 8) -> str:
    """Pick k diverse seed examples and format them as few-shot cases."""
    if not seed_path.exists():
        return ""
    rows = [json.loads(l) for l in seed_path.open()]
    # Balance classes as best we can
    by_label = {lab: [r for r in rows if r.get("label") == lab] for lab in LABELS}
    per = max(1, k // len(LABELS))
    picked = []
    for lab in LABELS:
        picked.extend(by_label[lab][:per])
    # Format
    parts = []
    for r in picked:
        snippet = r["text"][:400].replace("\n", " ")
        parts.append(f"SPAN: {snippet}\nLABEL: {r['label']}")
    return "\n\n".join(parts)


def build_prompt(span_text: str, fewshot: str) -> str:
    return (
        "You are labeling a span of reasoning from a math-solving model. "
        "Classify each span as one of the labels below.\n\n"
        f"Labels:\n{_label_defs()}\n"
        "Rules:\n"
        "- Look at the content of the span, not just the first word.\n"
        "- 'Wait' can open either strategy or reflection. Decide by what follows.\n"
        "- Respond with exactly one word: the label. No explanation.\n\n"
        f"{'Examples:' if fewshot else ''}\n{fewshot}\n\n"
        f"SPAN: {span_text[:800]}\n"
        "LABEL:"
    )


# ----------------- Provider wrappers -----------------

class AnthropicJudge:
    def __init__(self, model: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def label(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip().lower()


class OpenAIJudge:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def label(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip().lower()


def get_judge():
    if JUDGE_PROVIDER == "anthropic":
        return AnthropicJudge(JUDGE_MODEL_ANTHROPIC)
    if JUDGE_PROVIDER == "openai":
        return OpenAIJudge(JUDGE_MODEL_OPENAI)
    raise ValueError(f"Unknown provider: {JUDGE_PROVIDER}")


# ----------------- Main loop -----------------

def normalize_label(raw: str) -> str | None:
    raw = raw.lower().strip().strip(".").strip(",")
    for lab in LABELS:
        if raw.startswith(lab):
            return lab
    # Common aliases
    aliases = {
        "planning": "strategy",
        "plan": "strategy",
        "strategy-selection": "strategy",
        "executing": "execution",
        "execute": "execution",
        "meta": "reflection",
        "meta-reflection": "reflection",
        "self-reflection": "reflection",
    }
    return aliases.get(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="Cap total judge calls")
    ap.add_argument("--max-seed", type=int, default=12)
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between calls")
    args = ap.parse_args()

    if not SPANS_PATH.exists():
        raise FileNotFoundError(f"Missing {SPANS_PATH}. Run hand_annotate.py --extract first.")

    # Skip spans we already have labels for (either seed or previous judge run).
    already_done = set()
    if SEED_LABELS_PATH.exists():
        for l in SEED_LABELS_PATH.open():
            r = json.loads(l)
            already_done.add((r["trace_id"], r["span_idx"]))
    if JUDGE_LABELS_PATH.exists():
        for l in JUDGE_LABELS_PATH.open():
            r = json.loads(l)
            already_done.add((r["trace_id"], r["span_idx"]))

    fewshot = _build_fewshot(SEED_LABELS_PATH, k=args.max_seed)
    judge = get_judge()

    n_done = 0
    n_bad = 0
    with SPANS_PATH.open() as fin, JUDGE_LABELS_PATH.open("a") as fout:
        pbar = tqdm(total=args.limit, desc="judging")
        for line in fin:
            if n_done >= args.limit:
                break
            row = json.loads(line)
            key = (row["trace_id"], row["span_idx"])
            if key in already_done:
                continue
            prompt = build_prompt(row["text"], fewshot)
            try:
                raw = judge.label(prompt)
            except Exception as e:
                print(f"judge error: {e}", file=sys.stderr)
                time.sleep(2.0)
                continue
            lab = normalize_label(raw)
            if lab is None:
                n_bad += 1
                continue
            row["label"] = lab
            row["judge_raw"] = raw
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done += 1
            pbar.update(1)
            if args.sleep:
                time.sleep(args.sleep)
        pbar.close()

    print(f"Labeled {n_done} spans. Unparseable: {n_bad}.")
    print(f"Output: {JUDGE_LABELS_PATH}")


if __name__ == "__main__":
    main()
