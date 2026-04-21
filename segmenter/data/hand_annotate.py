"""
CLI hand-annotation tool for the seed dataset.

Usage:
    # First: expand traces into spans using boundary detection
    python -m data.hand_annotate --extract

    # Then: label them one at a time
    python -m data.hand_annotate --label --n 100

The seed set does not need to be big. 50 to 100 labeled spans is enough to (a) write a
good few-shot prompt for the LLM judge and (b) spot-check the judge's labels later.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import LABELS, RAW_TRACES_PATH, SPANS_PATH, SEED_LABELS_PATH, SEED
from data.boundary_detection import detect_spans


def cmd_extract(args):
    """Turn each raw trace into N candidate spans."""
    if not RAW_TRACES_PATH.exists():
        raise FileNotFoundError(f"Run generate_traces.py first; missing {RAW_TRACES_PATH}")

    n_traces = 0
    n_spans = 0
    with RAW_TRACES_PATH.open() as fin, SPANS_PATH.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            spans = detect_spans(row["thinking"], trace_id=row["trace_id"])
            for sp in spans:
                out_row = sp.to_dict()
                out_row["correct"] = row["correct"]
                out_row["problem"] = row["problem"]
                fout.write(json.dumps(out_row) + "\n")
                n_spans += 1
            n_traces += 1
    print(f"Extracted {n_spans} spans from {n_traces} traces -> {SPANS_PATH}")


def cmd_label(args):
    """Walk through spans and ask the human for a label."""
    if not SPANS_PATH.exists():
        raise FileNotFoundError(f"Run --extract first; missing {SPANS_PATH}")

    with SPANS_PATH.open() as fin:
        spans = [json.loads(l) for l in fin]

    # Deterministic shuffle so two annotators labeling different halves still mix well.
    rng = random.Random(SEED)
    rng.shuffle(spans)

    # Resume from existing labels if the file is already partially done.
    labeled = []
    labeled_keys = set()
    if SEED_LABELS_PATH.exists():
        with SEED_LABELS_PATH.open() as fin:
            for line in fin:
                row = json.loads(line)
                labeled.append(row)
                labeled_keys.add((row["trace_id"], row["span_idx"]))
        print(f"Resuming. Already labeled: {len(labeled)}.")

    print("\nLabel keys:")
    for i, lab in enumerate(LABELS):
        print(f"  {i} = {lab}")
    print("  s = skip, q = save and quit\n")

    target = args.n
    done_this_session = 0

    with SEED_LABELS_PATH.open("a") as fout:
        for sp in spans:
            if done_this_session >= target:
                break
            key = (sp["trace_id"], sp["span_idx"])
            if key in labeled_keys:
                continue

            print("=" * 80)
            print(f"Trace {sp['trace_id']}  span {sp['span_idx']}  trigger={sp.get('trigger')!r}")
            print("-" * 80)
            print(sp["text"][:800] + ("..." if len(sp["text"]) > 800 else ""))
            print("-" * 80)
            raw = input(f"Label [{'/'.join(str(i) for i in range(len(LABELS)))}/s/q]: ").strip().lower()

            if raw == "q":
                break
            if raw == "s":
                continue
            if not raw.isdigit() or int(raw) not in range(len(LABELS)):
                print("Invalid, skipping.")
                continue

            sp["label"] = LABELS[int(raw)]
            fout.write(json.dumps(sp) + "\n")
            fout.flush()
            done_this_session += 1

    print(f"\nLabeled {done_this_session} this session.")
    print(f"Total labeled: {len(labeled) + done_this_session}")
    print(f"Output: {SEED_LABELS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="Turn raw traces into spans")
    ap.add_argument("--label", action="store_true", help="Launch the labeling CLI")
    ap.add_argument("--n", type=int, default=50, help="How many to label this session")
    args = ap.parse_args()

    if args.extract:
        cmd_extract(args)
    elif args.label:
        cmd_label(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
