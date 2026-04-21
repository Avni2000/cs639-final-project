"""
Diagnostics for the segmenter data pipeline.

Run this after each stage to sanity-check what you have. It surfaces the most
common failure modes: class imbalance, duplicate spans, trigger leakage, and
judge-vs-seed disagreements.

Usage:
    python -m scripts.inspect_data
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LABELS, RAW_TRACES_PATH, SPANS_PATH, SEED_LABELS_PATH, JUDGE_LABELS_PATH,
)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f]


def section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def inspect_traces():
    rows = load_jsonl(RAW_TRACES_PATH)
    if not rows:
        print(f"(no {RAW_TRACES_PATH.name})")
        return
    section(f"raw_traces.jsonl  ({len(rows)} traces)")
    lengths = [len(r.get("thinking", "")) for r in rows]
    correct = sum(1 for r in rows if r.get("correct"))
    print(f"  correct: {correct}/{len(rows)} = {correct/len(rows):.1%}")
    print(f"  thinking length (chars): mean={mean(lengths):.0f}  median={median(lengths):.0f}  max={max(lengths)}")


def inspect_spans():
    rows = load_jsonl(SPANS_PATH)
    if not rows:
        print(f"(no {SPANS_PATH.name})")
        return
    section(f"spans.jsonl  ({len(rows)} spans)")
    by_trace = Counter(r["trace_id"] for r in rows)
    lengths = [len(r["text"]) for r in rows]
    triggers = Counter(r.get("trigger") for r in rows)
    print(f"  spans per trace: mean={mean(by_trace.values()):.1f}  median={median(by_trace.values())}  max={max(by_trace.values())}")
    print(f"  span length (chars): mean={mean(lengths):.0f}  median={median(lengths):.0f}")
    print(f"  top 10 triggers:")
    for trig, n in triggers.most_common(10):
        print(f"    {str(trig):>20}  {n}")


def inspect_labels(path: Path, name: str):
    rows = load_jsonl(path)
    if not rows:
        print(f"(no {path.name})")
        return
    section(f"{name}  ({len(rows)} spans)")
    dist = Counter(r.get("label") for r in rows)
    for lab in LABELS:
        n = dist.get(lab, 0)
        pct = n / len(rows) if rows else 0
        bar = "#" * int(pct * 40)
        print(f"  {lab:>12}  {n:>5}  {pct:>5.1%}  {bar}")
    unknown = {l: n for l, n in dist.items() if l not in LABELS}
    if unknown:
        print(f"  unknown labels (not in config.LABELS): {unknown}")


def inspect_agreement():
    """If the same span appears in both seed and judge files, compare labels."""
    seed = {(r["trace_id"], r["span_idx"]): r.get("label") for r in load_jsonl(SEED_LABELS_PATH)}
    judge = {(r["trace_id"], r["span_idx"]): r.get("label") for r in load_jsonl(JUDGE_LABELS_PATH)}
    overlap = set(seed) & set(judge)
    if not overlap:
        return
    section(f"Seed vs Judge agreement on {len(overlap)} overlapping spans")
    agree = sum(1 for k in overlap if seed[k] == judge[k])
    print(f"  agreement: {agree}/{len(overlap)} = {agree/len(overlap):.1%}")
    # Per-label breakdown
    conf = Counter()
    for k in overlap:
        conf[(seed[k], judge[k])] += 1
    print(f"  (seed_label -> judge_label): count")
    for (s, j), n in sorted(conf.items()):
        flag = "" if s == j else "  <-- disagreement"
        print(f"    {s:>10} -> {j:<10}  {n}{flag}")


def inspect_duplicates():
    """Same span text appearing multiple times in labeled data is usually a bug."""
    for path, name in [(SEED_LABELS_PATH, "seed"), (JUDGE_LABELS_PATH, "judge")]:
        rows = load_jsonl(path)
        if not rows:
            continue
        text_counts = Counter(r["text"][:100] for r in rows)
        dups = [(t, n) for t, n in text_counts.items() if n > 1]
        if dups:
            section(f"Duplicate span texts in {name} ({len(dups)} unique duplicated texts)")
            for t, n in sorted(dups, key=lambda x: -x[1])[:5]:
                print(f"  x{n}: {t[:80]}")


def main():
    inspect_traces()
    inspect_spans()
    inspect_labels(SEED_LABELS_PATH, "spans_seed_labeled.jsonl")
    inspect_labels(JUDGE_LABELS_PATH, "spans_judge_labeled.jsonl")
    inspect_agreement()
    inspect_duplicates()
    print()


if __name__ == "__main__":
    main()
