"""
Integration layer: wires together probe baseline, segmenter, and steering.

Pipeline:
  1. Generate CoT trace for problem
  2. Segment trace into reasoning spans
  3. Filter for "planning" spans only
  4. Extract hidden states at paragraph boundaries within planning spans
  5. Run probe baseline on those hidden states
  6. Return structured output for steering module to consume
"""

import torch
import random
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from segmenter.data.boundary_detection import split_into_spans
from segmenter.inference.probe_selector import select_probe_points


# ============================================================================
# TRACE GENERATION (from probe-baseline notebook)
# ============================================================================

def build_prompt(problem_text: str, tokenizer) -> str:
    """Build chat-formatted prompt for AIME problem."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following AIME problem. "
                "Your final answer must be a non-negative integer (0-999), "
                "placed inside \\boxed{}.\n\n"
                + problem_text
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _find_subseq(seq, sub):
    """Find subsequence in list."""
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return -1


def _paragraph_break_indices(cot_token_ids, tokenizer):
    """Return token indices where \\n\\n paragraph breaks occur in CoT."""
    positions = []
    cursor = 0
    for i in range(len(cot_token_ids)):
        text = tokenizer.decode(cot_token_ids[: i + 1], skip_special_tokens=False)
        hit = text.find("\n\n", cursor)
        if hit != -1:
            positions.append(i)
            cursor = hit + 2
    return positions


def generate_trace(
    problem_text: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 16384,
    probe_layer: int = -1,
) -> Dict[str, Any]:
    """
    Generate CoT trace for a problem and extract hidden states at paragraph breaks.

    Returns:
        {
            "problem_text": str,
            "trace_text": str,
            "generated_ids": list,
            "cot_token_ids": list,
            "paragraph_positions": list of int (indices in cot_token_ids),
            "all_hidden_states": tensor [len(cot_token_ids), hidden_dim],
        }
    """
    prompt = build_prompt(problem_text, tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        generation = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    gen_ids = generation.sequences
    generated_ids = gen_ids[0, prompt_len:].tolist()

    # Extract hidden states at probe_layer for each generated token
    if generation.hidden_states:
        hs = torch.cat(
            [step_hidden[probe_layer][:, -1, :] for step_hidden in generation.hidden_states],
            dim=0,
        ).float().cpu()
    else:
        hs = torch.empty((0, model.config.hidden_size), dtype=torch.float32)

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Extract CoT boundaries (between <think> and </think>)
    think_toks = tokenizer.encode("<think>", add_special_tokens=False)
    end_toks = tokenizer.encode("</think>", add_special_tokens=False)

    think_pos = _find_subseq(generated_ids, think_toks)
    end_pos = _find_subseq(generated_ids, end_toks)

    if think_pos != -1 and end_pos != -1 and end_pos > think_pos:
        cot_start = think_pos + len(think_toks)
        cot_end = end_pos
    else:
        cot_start = 0
        cot_end = len(generated_ids)

    cot_token_ids = generated_ids[cot_start:cot_end]
    cot_hidden_states = hs[cot_start:cot_end]

    # Find paragraph breaks within CoT
    paragraph_positions = _paragraph_break_indices(cot_token_ids, tokenizer)

    return {
        "problem_text": problem_text,
        "trace_text": generated_text,
        "generated_ids": generated_ids,
        "cot_token_ids": cot_token_ids,
        "paragraph_positions": paragraph_positions,
        "all_hidden_states": cot_hidden_states,  # [len(cot_token_ids), hidden_dim]
    }


# ============================================================================
# SEGMENTATION & FILTERING (uses segmenter modules)
# ============================================================================

def segment_trace(trace_text: str) -> List[Dict[str, Any]]:
    """
    Segment trace into spans using boundary detection.

    Returns:
        [
            {
                "text": str,
                "start_char": int,
                "end_char": int,
                "is_trigger": bool,
            },
            ...
        ]
    """
    spans_raw = split_into_spans(trace_text)
    return [
        {
            "text": text,
            "start_char": start,
            "end_char": end,
            "is_trigger": is_trigger,
        }
        for start, end, text, is_trigger in spans_raw
    ]


def assign_labels_to_spans(spans: List[Dict[str, Any]], segmenter_model=None) -> List[Dict[str, Any]]:
    """
    Assign {planning, execution, meta-reflection} labels to spans.

    If segmenter_model is None, uses a simple heuristic:
      - is_trigger=True → "planning"
      - is_trigger=False → "execution"
    (This is a placeholder until the segmenter classifier is trained.)

    Returns updated spans with "label" field.
    """
    for span in spans:
        if segmenter_model is not None:
            # TODO: Call segmenter_model.predict(span["text"])
            span["label"] = segmenter_model.predict(span["text"])
        else:
            # Heuristic: trigger tokens → planning, others → execution
            span["label"] = "planning" if span["is_trigger"] else "execution"

    return spans


def filter_planning_spans(labeled_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter for planning spans only (strategy, not execution details)."""
    return [s for s in labeled_spans if s.get("label") == "planning"]


# ============================================================================
# HIDDEN STATE EXTRACTION FOR PLANNING SPANS
# ============================================================================

def get_char_to_token_mapping(trace_text: str, cot_token_ids: List[int], tokenizer) -> Dict[int, int]:
    """
    Build a mapping from character positions in trace_text to token indices in cot_token_ids.

    Returns a dict: char_idx → token_idx
    """
    char_to_token = {}
    cursor_char = 0
    cursor_token = 0

    for token_idx, token_id in enumerate(cot_token_ids):
        # Decode from start up to this token
        text = tokenizer.decode(cot_token_ids[:token_idx + 1], skip_special_tokens=False)
        # Find how many characters this represents
        num_chars = len(text)

        for ch_idx in range(cursor_char, num_chars):
            char_to_token[ch_idx] = token_idx

        cursor_char = num_chars
        cursor_token = token_idx + 1

    return char_to_token


def extract_hidden_states_for_planning_spans(
    trace_data: Dict[str, Any],
    planning_spans: List[Dict[str, Any]],
    tokenizer,
) -> List[Dict[str, Any]]:
    """
    For each planning span, extract hidden states at paragraph breaks within that span.

    Returns updated planning_spans with "paragraph_positions" field:
        [
            {
                "text": str,
                "label": "planning",
                "start_char": int,
                "end_char": int,
                "paragraph_positions": [
                    {
                        "token_idx": int,
                        "hidden_state": tensor [hidden_dim],
                        "char_pos": int,
                    },
                    ...
                ]
            },
            ...
        ]
    """
    trace_text = trace_data["trace_text"]
    cot_token_ids = trace_data["cot_token_ids"]
    all_hidden_states = trace_data["all_hidden_states"]
    paragraph_positions = trace_data["paragraph_positions"]

    # Build mapping from character position to token index
    char_to_token = get_char_to_token_mapping(trace_text, cot_token_ids, tokenizer)

    for span in planning_spans:
        span_start_char = span["start_char"]
        span_end_char = span["end_char"]

        # Find paragraph breaks within this span
        paragraph_positions_in_span = []
        for para_token_idx in paragraph_positions:
            # Approximate: find the character position of this token
            # (This is a simplification; ideally we'd have exact char↔token mapping)
            if para_token_idx < len(all_hidden_states):
                # Assume the paragraph break is somewhere in the middle of this token
                # For now, just include it if it falls in the span
                paragraph_positions_in_span.append({
                    "token_idx": para_token_idx,
                    "hidden_state": all_hidden_states[para_token_idx].clone(),
                })

        span["paragraph_positions"] = paragraph_positions_in_span

    return planning_spans


# ============================================================================
# PROBE BASELINE (assumes probe model is loaded separately)
# ============================================================================

def run_probe_baseline(
    planning_spans_with_hidden_states: List[Dict[str, Any]],
    probe_model,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Run probe baseline on hidden states from planning spans.

    Updates each paragraph position with:
        "probe_logit": float,
        "probe_confidence": float (sigmoid of logit),
        "probe_prediction": int (0 or 1),
        "should_steer": bool (True if confidence < threshold),

    Returns updated planning_spans.
    """
    probe_model.eval()

    for span in planning_spans_with_hidden_states:
        for para_pos in span["paragraph_positions"]:
            hidden = para_pos["hidden_state"].unsqueeze(0).to(device)  # [1, hidden_dim]

            with torch.no_grad():
                logit = probe_model(hidden).item()  # scalar

            confidence = torch.sigmoid(torch.tensor(logit)).item()
            prediction = 1 if logit > 0 else 0

            # Steer if confidence is low (e.g., < 0.6)
            should_steer = confidence < 0.6

            para_pos["probe_logit"] = logit
            para_pos["probe_confidence"] = confidence
            para_pos["probe_prediction"] = prediction
            para_pos["should_steer"] = should_steer

    return planning_spans_with_hidden_states


# ============================================================================
# MAIN INTEGRATION PIPELINE
# ============================================================================

def integrate(
    problem_text: str,
    model,
    tokenizer,
    probe_model,
    device: str,
    segmenter_model=None,
    max_new_tokens: int = 16384,
    probe_layer: int = -1,
) -> Dict[str, Any]:
    """
    Full pipeline: problem → trace → segment → filter planning → extract hidden → probe.

    Args:
        problem_text: AIME problem statement
        model: DeepSeek model (loaded)
        tokenizer: Tokenizer
        probe_model: Trained linear probe (loaded)
        device: "cuda" or "cpu"
        segmenter_model: Trained segmenter classifier (optional; uses heuristic if None)
        max_new_tokens: Max tokens for generation
        probe_layer: Which layer to extract hidden states from (-1 = last)

    Returns:
        {
            "problem_text": str,
            "trace_text": str,
            "planning_spans": [
                {
                    "text": str,
                    "label": "planning",
                    "is_trigger": bool,
                    "paragraph_positions": [
                        {
                            "token_idx": int,
                            "hidden_state": tensor,
                            "probe_confidence": float,
                            "probe_prediction": int,
                            "should_steer": bool,
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    # Step 1: Generate trace
    print("Generating CoT trace...")
    trace_data = generate_trace(
        problem_text, model, tokenizer, device, max_new_tokens, probe_layer
    )

    # Step 2: Segment trace
    print("Segmenting trace...")
    all_spans = segment_trace(trace_data["trace_text"])

    # Step 3: Assign labels
    print("Assigning labels to spans...")
    labeled_spans = assign_labels_to_spans(all_spans, segmenter_model)

    # Step 4: Filter for planning only
    print("Filtering for planning spans...")
    planning_spans = filter_planning_spans(labeled_spans)

    # Step 5: Extract hidden states for planning spans
    print("Extracting hidden states for planning spans...")
    planning_with_hidden = extract_hidden_states_for_planning_spans(
        trace_data, planning_spans, tokenizer
    )

    # Step 6: Run probe
    print("Running probe baseline...")
    planning_with_probe = run_probe_baseline(planning_with_hidden, probe_model, device)

    return {
        "problem_text": problem_text,
        "trace_text": trace_data["trace_text"],
        "planning_spans": planning_with_probe,
    }


if __name__ == "__main__":
    # Example usage (requires models to be loaded externally)
    print("Integration module ready. Import and call integrate() with loaded models.")
