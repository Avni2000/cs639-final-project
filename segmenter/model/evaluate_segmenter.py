"""
Standalone evaluation of the trained segmenter.

Reports:
  1. Per-class precision/recall/F1 on the held-out hand-labeled set.
  2. Agreement with the LLM judge on a random sample (does the student copy the teacher?).
  3. Confusion matrix to see what's being mixed up.

Usage:
    python -m model.evaluate_segmenter
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    SEED_LABELS_PATH, JUDGE_LABELS_PATH, MODEL_DIR, MAX_LEN, SEED,
)


class SegmenterClassifier:
    """Wraps the fine-tuned encoder for easy prediction."""
    def __init__(self, model_dir: Path):
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.cpu().numpy()
            ids = np.argmax(logits, axis=-1)
            out.extend(ID2LABEL[i] for i in ids)
        return out


def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open()]


def main():
    clf = SegmenterClassifier(MODEL_DIR)

    # --- 1. Against hand labels ---
    seed_rows = [r for r in load(SEED_LABELS_PATH) if r.get("label") in LABEL2ID]
    if seed_rows:
        texts = [r["text"] for r in seed_rows]
        y_true = [r["label"] for r in seed_rows]
        y_pred = clf.predict(texts)
        print(f"\n===== vs hand labels (n={len(seed_rows)}) =====")
        print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        print("Confusion matrix (rows=true, cols=pred):")
        print("            " + "  ".join(f"{l:>9}" for l in LABELS))
        for lab, row in zip(LABELS, cm):
            print(f"  {lab:>9}  " + "  ".join(f"{v:>9d}" for v in row))
    else:
        print("No hand labels found, skipping.")

    # --- 2. Against judge labels (sanity check distillation) ---
    judge_rows = [r for r in load(JUDGE_LABELS_PATH) if r.get("label") in LABEL2ID]
    if judge_rows:
        random.Random(SEED).shuffle(judge_rows)
        sample = judge_rows[:500]
        texts = [r["text"] for r in sample]
        y_true = [r["label"] for r in sample]
        y_pred = clf.predict(texts)
        agree = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(sample)
        print(f"\n===== Agreement with LLM judge on {len(sample)} random spans =====")
        print(f"Agreement: {agree:.3f}")
        print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
