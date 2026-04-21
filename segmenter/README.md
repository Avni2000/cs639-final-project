# Segmenter training pipeline

This is the training code for Model 2 (the segmenter) in the strategy-selection
steering project. It implements the three-stage plan from your notes:
hand-annotate a seed set, scale it up with an LLM judge, distill into a fast student.

## What the segmenter does

Takes a reasoning trace (the content between `<think>...</think>` from a model like
DeepSeek-R1) and returns a list of spans, each labeled `strategy`, `execution`, or
optionally `reflection`. Downstream, the Probe (Model 1) reads hidden states at the
end of each `strategy` span.

Two stages inside the segmenter:
1. **Boundary detection** (rule-based, fast). Splits the trace into candidate spans
   using sentence boundaries and reflection tokens from arXiv:2506.12963.
2. **Classification** (learned, also fast). A small DistilBERT fine-tuned to label
   each span by content.

## Repo layout

```
segmenter/
  config.py                        shared settings
  data/
    boundary_detection.py          stage 1: rule-based splitter
    generate_traces.py             run reasoning model on MATH/AIME
    hand_annotate.py               CLI for seed labeling
    llm_judge_labeler.py           scale up labels with Claude or GPT-4o
  model/
    train_segmenter.py             fine-tune distilbert on labeled spans
    evaluate_segmenter.py          metrics + confusion matrix
  inference/
    segmenter_pipeline.py          end-to-end: text -> labeled spans
  scripts/
    make_synthetic_data.py         dry-run data for testing the pipeline
    inspect_data.py                diagnostic stats on labeled data
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key-here"  # or OPENAI_API_KEY
```

If you do not have a GPU for generating traces, skip step 1 and use any reasoning
traces you already have. The segmenter itself trains in a few minutes on CPU.

## Dry-run first (no API key, no GPU)

Before spending API budget or compute, verify the pipeline is wired up correctly
on synthetic data:

```bash
python -m scripts.make_synthetic_data --n-spans 600
python -m scripts.inspect_data
python -m model.train_segmenter
python -m model.evaluate_segmenter
```

On synthetic spans the trained segmenter should hit 90%+ F1. That tells you the
plumbing is correct; it says nothing about real-trace performance.

## Step-by-step (real data)

### Step 1. Generate some reasoning traces (only if you don't have any)

```bash
python -m data.generate_traces --n-problems 200 --traces-per 3
```

This writes `data_out/raw_traces.jsonl`. Each row has the problem, the thinking
portion, and a loose correctness flag. For segmenter training you do NOT need the
full 2000 x 20 run; a few hundred traces is enough. The heavy run happens later
when you also need to save hidden states for the probe.

You can also skip this script entirely and drop your own traces into that path,
as long as each row has `trace_id` and `thinking` fields.

### Step 2. Turn traces into candidate spans

```bash
python -m data.hand_annotate --extract
```

This runs boundary detection on every trace and writes `data_out/spans.jsonl`.
Expect ~10 to 30 spans per trace.

### Step 3. Hand-label a seed set

```bash
python -m data.hand_annotate --label --n 60
```

This pops up a CLI that shows one span at a time. You type `0` for strategy,
`1` for execution (and `2` for reflection if you enabled it in `config.py`).
Aim for 50 to 100 labeled spans. It resumes where you left off.

Why you actually need this:
- It gives the LLM judge real few-shot examples in your actual style.
- It gives the final evaluation a trusted ground-truth test set. Without this,
  you only know whether the student copies the teacher, not whether either of
  them is correct.

### Step 4. Scale labels up with an LLM judge

```bash
python -m data.llm_judge_labeler --limit 3000
```

The judge sees your seed examples as few-shot, then labels the rest of the span
pool. Cost estimate with Claude Sonnet 4.5 at ~$3/M input tokens: roughly $2 for
3000 spans. With `gpt-4o-mini` (`JUDGE_PROVIDER = "openai"` in `config.py`) it's
under $0.50.

Output: `data_out/spans_judge_labeled.jsonl`.

### Step 5. Train the distilled segmenter

```bash
python -m model.train_segmenter
```

This fine-tunes `distilbert-base-uncased` on the judge labels. Your hand-labeled
seed set is held out as the test set. Trains in about 5 minutes on a single GPU
or 20 minutes on CPU.

The final output:
```
===== Test set (hand-labeled) =====
              precision    recall  f1-score   support
    strategy      0.87      0.82      0.85        40
   execution      0.91      0.94      0.92        60
    accuracy                          0.89       100
```

Target: macro-F1 above 0.80 on your hand-labeled set. Below that, label more
seeds (your judge prompt is probably drifting) or try a larger base model like
`microsoft/deberta-v3-small`.

### Step 6. Evaluate

```bash
python -m model.evaluate_segmenter
```

Gives you the per-class report plus a confusion matrix, and also an
"agreement with judge" number on a random 500-span sample. The gap between
those two numbers tells you how much of the error is distillation loss vs
genuine task difficulty.

### Step 7. Use it

```python
from inference.segmenter_pipeline import Segmenter

seg = Segmenter()
spans = seg(trace_text, trace_id="p123")
for sp in spans:
    if sp.label == "strategy":
        # This is where the Probe wants to capture the hidden state.
        # sp.end_char is the char offset in the original trace.
        ...
```

Feed `sp.end_char` to whatever hidden-state hook you set up in the reasoning
model. That's the moment to call the Probe.

## Design choices worth knowing

**Why sentence-level cuts, not token-level.** Token-level segmentation collapses
to "classify every token" which is far more annotation per trace and does not
help the Probe, since hidden states are useful at the *end* of a strategy thought,
not in the middle of one.

**Why start with two labels, not three.** `reflection` is noisier to label
consistently. Get the strategy vs execution split working first, then flip
`USE_META = True` in `config.py` and re-label.

**Why the seed set is held out.** Otherwise you measure how well the student
copies the teacher, which is the wrong question. Classification accuracy against
the judge can be 95% while accuracy against humans is 70%, because the judge
has its own systematic errors.

**Why the judge is offline-only.** The whole point of distilling is that
calling a frontier LLM at every segmentation boundary during inference would
blow the token budget we are trying to save.

## Common problems

- **Judge returns weird strings.** The normalizer in `llm_judge_labeler.py`
  handles `strategy-selection`, `planning`, `execute`, etc. If you see a new
  alias, add it to `normalize_label`.
- **Student is 50/50 on test set.** Probably class imbalance. Check the label
  distribution that `train_segmenter.py` prints at start. If strategy is under
  20%, oversample it or weight the loss.
- **Boundary detection misses spans.** Add more triggers to
  `REFLECTION_TOKENS` in `boundary_detection.py`. The list there is a union
  of the arXiv:2506.12963 reflection tokens and planning markers; your model
  may have its own quirks.
