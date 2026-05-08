"""
Steering module: applies activation steering based on probe confidence.

This module receives output from integration.py and:
  1. Identifies which planning spans have low probe confidence
  2. Computes steering vector v = mean(h+) - mean(h-)
  3. Applies α * v to hidden states in middle-late layers
  4. Re-generates or continues reasoning from the steered state

(Stub - to be implemented by steering team)
"""

import torch
from typing import Dict, List, Any


# ============================================================================
# STEERING VECTOR COMPUTATION
# ============================================================================

def compute_steering_vector(
    correct_hidden_states: torch.Tensor,
    incorrect_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute steering vector v = mean(h+) - mean(h-).

    Args:
        correct_hidden_states: [N_correct, hidden_dim] - hidden states from correct traces
        incorrect_hidden_states: [N_incorrect, hidden_dim] - hidden states from incorrect traces

    Returns:
        v: [hidden_dim] - steering direction vector
    """
    v = correct_hidden_states.mean(dim=0) - incorrect_hidden_states.mean(dim=0)
    return v


# ============================================================================
# STEERING APPLICATION
# ============================================================================

def apply_steering_to_hidden_state(
    hidden_state: torch.Tensor,
    steering_vector: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Apply steering: h_steered = h + α * v

    Args:
        hidden_state: [hidden_dim] - original hidden state
        steering_vector: [hidden_dim] - direction to move
        alpha: float - scaling factor (how much to steer)

    Returns:
        hidden_state_steered: [hidden_dim]
    """
    return hidden_state + alpha * steering_vector


# ============================================================================
# IDENTIFYING STEER-WORTHY POSITIONS
# ============================================================================

def get_steer_positions(integration_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From integration output, extract positions where probe confidence is low.

    Returns:
        [
            {
                "span_text": str,
                "token_idx": int,
                "hidden_state": tensor,
                "probe_confidence": float,
                "char_position": int,
            },
            ...
        ]
    """
    steer_positions = []

    for span in integration_output["planning_spans"]:
        for para_pos in span["paragraph_positions"]:
            if para_pos.get("should_steer", False):
                steer_positions.append({
                    "span_text": span["text"],
                    "token_idx": para_pos["token_idx"],
                    "hidden_state": para_pos["hidden_state"],
                    "probe_confidence": para_pos["probe_confidence"],
                    "probe_prediction": para_pos["probe_prediction"],
                })

    return steer_positions


# ============================================================================
# FULL STEERING PIPELINE (stub)
# ============================================================================

def steer_and_regenerate(
    integration_output: Dict[str, Any],
    model,
    tokenizer,
    device: str,
    steering_vector: torch.Tensor,
    steer_layer: int = 14,  # middle-late layer
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Given integration output (with probe results), identify low-confidence positions,
    apply steering, and re-generate from steered state.

    Args:
        integration_output: Output from integration.integrate()
        model: DeepSeek model
        tokenizer: Tokenizer
        device: "cuda" or "cpu"
        steering_vector: Pre-computed v = mean(h+) - mean(h-)
        steer_layer: Which layer to apply steering to (middle-late, e.g. 14-20)
        alpha: Scaling factor for steering magnitude

    Returns:
        {
            "original_trace": str,
            "steered_positions": [...],
            "regenerated_trace": str,
            "probe_results_after_steering": [...],
        }
    """
    steer_positions = get_steer_positions(integration_output)

    if not steer_positions:
        return {
            "original_trace": integration_output["trace_text"],
            "steered_positions": [],
            "regenerated_trace": integration_output["trace_text"],
            "note": "No low-confidence positions to steer",
        }

    # TODO: Implement steering in model forward pass
    # This requires intervening during generation to modify hidden states at steer_layer
    # Pseudo-code:
    #   1. Hook into model's forward at steer_layer
    #   2. When we reach a steer position, apply: h += α * v
    #   3. Continue generation from steered state
    #   4. Collect new hidden states and probe results

    return {
        "original_trace": integration_output["trace_text"],
        "steered_positions": steer_positions,
        "regenerated_trace": "<TODO: implement steering in forward pass>",
        "probe_results_after_steering": [],
    }


if __name__ == "__main__":
    print("Steering module stub ready. Implement steer_and_regenerate() with model hooks.")
