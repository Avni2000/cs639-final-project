# Integration: Wiring Probe Baseline, Segmenter, and Steering

This directory now has three interconnected modules for the steering pipeline:

## 📁 Module Files

### `integration.py`
**Main wiring layer** that orchestrates the full pipeline.

```python
from integration import integrate

output = integrate(
    problem_text="<AIME problem>",
    model=deepseek_model,
    tokenizer=tokenizer,
    probe_model=linear_probe,
    device="cuda",
    segmenter_model=None,  # Optional; uses heuristic if None
)
```

**Data Flow:**
1. Generate CoT trace (uses probe-baseline logic)
2. Segment trace into reasoning spans (uses `segmenter.data.boundary_detection.split_into_spans()`)
3. Assign labels {planning, execution} (placeholder; will use segmenter classifier when trained)
4. **Filter for planning spans only** ← Key: ignores execution details
5. Extract hidden states at paragraph breaks within planning spans
6. Run probe baseline on those hidden states
7. Mark positions with `should_steer=True` if probe confidence < 0.6

**Output Structure:**
```python
{
    "problem_text": "...",
    "trace_text": "...",
    "planning_spans": [
        {
            "text": "First, I need to...",
            "label": "planning",
            "is_trigger": True,
            "start_char": 123,
            "end_char": 456,
            "paragraph_positions": [
                {
                    "token_idx": 45,
                    "hidden_state": tensor([...]),  # [hidden_dim]
                    "probe_logit": 0.32,
                    "probe_confidence": 0.58,       # sigmoid(logit)
                    "probe_prediction": 0,          # 1=correct, 0=incorrect
                    "should_steer": True,           # confidence < 0.6
                },
                # ... more paragraph positions
            ]
        },
        # ... more planning spans
    ]
}
```

### `steering.py`
**Steering application stub** (to be implemented by steering team).

Key functions:
- `compute_steering_vector(correct_hs, incorrect_hs)` → v = mean(h+) - mean(h-)
- `apply_steering_to_hidden_state(h, v, alpha)` → h_steered = h + α*v
- `get_steer_positions(integration_output)` → list of low-confidence positions
- `steer_and_regenerate(...)` → (stub) apply steering during generation

**Expected Usage:**
```python
from steering import steer_and_regenerate

# After running integration
steering_output = steer_and_regenerate(
    integration_output=probe_results,
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    steering_vector=v,  # Pre-computed from training set
    steer_layer=14,     # Middle-late layer (14-20 for DeepSeek-7B)
    alpha=0.5,          # Scaling factor
)
```

### `example_usage.py`
Complete example showing how to:
1. Load models
2. Run integration pipeline
3. Prepare for steering
4. (Sketch of) steering application

## 🔄 Data Flow Diagram

```
AIME Problem
    ↓
[integration.generate_trace]
    ↓
CoT Trace + Hidden States (layer 28)
    ↓
[integration.segment_trace + assign_labels]
    ↓
Spans: {planning, execution, ...}
    ↓
[integration.filter_planning_spans] ← KEY: Only planning spans
    ↓
Planning Spans Only
    ↓
[integration.extract_hidden_states_for_planning_spans]
    ↓
Planning Spans + Hidden States at Paragraph Breaks
    ↓
[integration.run_probe_baseline]
    ↓
Planning Spans + Probe Confidence Scores
    ↓
[steering.get_steer_positions] ← Identify low-conf positions
    ↓
[steering.steer_and_regenerate] ← Apply steering (TO BE IMPLEMENTED)
    ↓
Steered CoT Trace
    ↓
Re-evaluate / Extract Answer
```

## 🏗️ Integration Design Decisions

### Why Filter for Planning Spans Only?

The probe measures reasoning correctness. Execution details (arithmetic, substitution) aren't reasoning strategy — they're implementation. By probing only on **planning spans** (marked by reflection tokens like "wait", "but", "first"), we:
- Focus steering on high-level reasoning strategy
- Avoid noise from low-level computation correctness
- Reduce the number of steer points (fewer False Positives)

### Probe Threshold (`should_steer < 0.6`)

A confidence below 60% indicates the model's reasoning strategy is likely incorrect. This is the trigger for steering. You can tune this value.

### Steering Layer

- **Probe trains on**: Layer 28 (last layer, full reasoning representation)
- **Steering applies to**: Layer 14-20 (middle-to-late layers)
  - Why? Early layers = token embeddings, mid layers = strategic reasoning, late layers = answer formatting
  - Steering mid layers allows the model to "course correct" before committing to output

## 📦 Dependencies

From `probe-baseline/train.ipynb`:
- `transformers` (AutoTokenizer, AutoModelForCausalLM)
- `torch`
- `math-verify` (for answer grading)

From `segmenter/`:
- `segmenter.data.boundary_detection`
- `segmenter.inference.probe_selector` (unused here, but available)

## 🚀 How Your Team Uses This

### Probe Baseline Team (Avni, Dharini, Rinkle, Gayathri)
1. Train `linear_probe.pt` in your notebook
2. Save it: `torch.save(probe.state_dict(), "probe-baseline/linear_probe.pt")`
3. Pass it to `integration.integrate(..., probe_model=loaded_probe)`
4. Iterate on probe architecture / training as needed

### Segmenter Team (Jesse, Bin, Srinivas)
1. Train your DistilBERT classifier
2. Save it: `torch.save(model.state_dict(), "segmenter/4_checkpoints/segmenter.pt")`
3. Implement `model.predict(span_text) → label` method
4. Pass it to `integration.integrate(..., segmenter_model=trained_segmenter)`

### Steering Team (TBD)
1. Compute steering vector v from training set correct/incorrect traces
2. Implement model hooks in `steering.steer_and_regenerate()` to:
   - Intercept forward pass at `steer_layer`
   - Modify hidden state: `h += alpha * v` when reaching low-conf positions
   - Continue generation from steered state
3. Evaluate final answers on steered traces

## 🔧 Customization Points

All tunable in function calls:

```python
integrate(
    problem_text="...",
    model=model,
    tokenizer=tokenizer,
    probe_model=probe,
    device="cuda",
    segmenter_model=None,           # Can supply trained classifier
    max_new_tokens=16384,           # Trace length limit
    probe_layer=-1,                 # Which layer to extract hidden states from
)

steer_and_regenerate(
    integration_output=probe_results,
    ...
    steer_layer=14,                 # Which layer to apply steering to
    alpha=0.5,                      # Steering magnitude
)
```

And inside `integration.py`:
```python
# In run_probe_baseline():
should_steer = confidence < 0.6  # Change this threshold
```

## 📝 Next Steps

1. **Probe team**: Extract probe model from notebook, integrate into `integration.py`
2. **Segmenter team**: Train classifier, hook into `assign_labels_to_spans()`
3. **Steering team**: Implement `steering.steer_and_regenerate()` with model hooks
4. **Test end-to-end**: Run `python example_usage.py` with all components

---

**Questions?** Check `integration.py` docstrings — each function documents input/output format.
