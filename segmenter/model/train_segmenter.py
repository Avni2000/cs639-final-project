"""
Train the distilled segmenter (student model) on LLM-judge-labeled spans.

Input: spans_judge_labeled.jsonl (and optionally spans_seed_labeled.jsonl, merged).
Output: a fine-tuned encoder that maps span text to a label, saved to checkpoints/.

The seed (hand-labeled) rows are held out as the TEST set, so we measure the student
against human labels, not just against the teacher. This matters: if the student scores
well against the teacher but badly against humans, you've distilled the judge's bias,
not the task.

Usage:
    python -m model.train_segmenter
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BASE_MODEL, LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    SEED_LABELS_PATH, JUDGE_LABELS_PATH, MODEL_DIR,
    MAX_LEN, BATCH_SIZE, LR, EPOCHS, SEED,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labeled():
    """Merge seed + judge labels. Seed wins on conflicts."""
    rows = []
    seed_keys = set()
    if SEED_LABELS_PATH.exists():
        for l in SEED_LABELS_PATH.open():
            r = json.loads(l)
            r["source"] = "seed"
            rows.append(r)
            seed_keys.add((r["trace_id"], r["span_idx"]))
    if JUDGE_LABELS_PATH.exists():
        for l in JUDGE_LABELS_PATH.open():
            r = json.loads(l)
            if (r["trace_id"], r["span_idx"]) in seed_keys:
                continue
            r["source"] = "judge"
            rows.append(r)
    return rows


class SpanDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        enc["labels"] = LABEL2ID[r["label"]]
        return enc


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    set_seed(SEED)

    rows = load_labeled()
    if not rows:
        raise RuntimeError(
            "No labeled data. Run data/hand_annotate.py and data/llm_judge_labeler.py first."
        )
    # Filter any rows with labels not in our current label set (e.g., if USE_META toggled)
    rows = [r for r in rows if r.get("label") in LABEL2ID]
    print(f"Loaded {len(rows)} labeled spans. Label dist:")
    for lab in LABELS:
        n = sum(1 for r in rows if r["label"] == lab)
        print(f"  {lab}: {n}")

    # Hold out seed-labeled rows as the test set. They are the closest we have to human ground truth.
    test_rows = [r for r in rows if r.get("source") == "seed"]
    train_pool = [r for r in rows if r.get("source") != "seed"]

    # If the seed set is tiny (< 20), top it up with a random slice of the judge set.
    if len(test_rows) < 20:
        random.shuffle(train_pool)
        take = min(50, len(train_pool) // 5)
        test_rows += train_pool[:take]
        train_pool = train_pool[take:]

    # Split train_pool into train/val
    random.shuffle(train_pool)
    n_val = max(50, len(train_pool) // 10)
    val_rows = train_pool[:n_val]
    train_rows = train_pool[n_val:]

    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = SpanDataset(train_rows, tokenizer)
    val_ds = SpanDataset(val_rows, tokenizer)
    test_ds = SpanDataset(test_rows, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final test-set report. Seed rows are human-labeled, so this is the number that matters.
    print("\n===== Test set (hand-labeled) =====")
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    print(classification_report(
        y_true, y_pred,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        digits=3,
    ))

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nSaved segmenter to {MODEL_DIR}")


if __name__ == "__main__":
    main()
