"""
Example: How to use integration + steering modules together.

This shows the full workflow:
  1. Load models (DeepSeek, probe, optionally segmenter)
  2. Run integration pipeline (problem → trace → segment → probe)
  3. Pass results to steering (identify low-confidence positions, apply steering)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our modules
from integration import integrate
from steering import steer_and_regenerate, compute_steering_vector


def main():
    # ========================================================================
    # SETUP: Load models
    # ========================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load DeepSeek model
    print("Loading DeepSeek-R1-Distill-Qwen-7B...")
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    print(f"✓ Model loaded\n")

    # Load probe baseline
    print("Loading probe baseline...")
    from probe_baseline_module import LinearProbe  # TODO: extract from notebook

    probe_model = LinearProbe(hidden_dim=3584)  # DeepSeek hidden dim
    # probe_model.load_state_dict(torch.load("probe-baseline/linear_probe.pt", map_location=device))
    # TODO: Path depends on where probe is saved
    probe_model = probe_model.to(device)
    probe_model.eval()
    print(f"✓ Probe loaded (or initialize fresh)\n")

    # Optionally load segmenter (if trained)
    segmenter_model = None  # TODO: Load trained segmenter here
    # if segmenter_model exists:
    #   from segmenter import load_segmenter
    #   segmenter_model = load_segmenter("segmenter/4_checkpoints/segmenter.pt")

    # ========================================================================
    # STEP 1: Run integration pipeline
    # ========================================================================

    problem_text = """
    Find the sum of the solutions to the equation
    $$\\frac{5}{x} + \\frac{x}{5} = \\frac{5}{2}$$
    """

    print("=" * 70)
    print("STEP 1: Integration Pipeline (trace → segment → probe)")
    print("=" * 70)

    integration_output = integrate(
        problem_text=problem_text,
        model=model,
        tokenizer=tokenizer,
        probe_model=probe_model,
        device=device,
        segmenter_model=segmenter_model,
        max_new_tokens=4096,
        probe_layer=-1,  # Use last layer (layer 28)
    )

    print(f"\n✓ Integration complete")
    print(f"  Planning spans found: {len(integration_output['planning_spans'])}")
    print(f"  Original trace length: {len(integration_output['trace_text'])} chars\n")

    # Summarize probe results
    low_conf_count = 0
    for span in integration_output["planning_spans"]:
        for para_pos in span["paragraph_positions"]:
            if para_pos.get("should_steer", False):
                low_conf_count += 1

    print(f"  Low-confidence positions (should_steer=True): {low_conf_count}")

    # ========================================================================
    # STEP 2: (Optional) Compute steering vector
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Steering Preparation (compute steering vector)")
    print("=" * 70)

    # In practice, you'd collect correct/incorrect hidden states from a training set
    # For now, just a stub:
    print("(Steering vector normally computed from training set of correct/incorrect traces)")
    print("For demo, skipping actual steering application.\n")

    # Pseudo-code:
    # correct_hs = torch.load("steering/correct_hidden_states.pt")  # [N, hidden_dim]
    # incorrect_hs = torch.load("steering/incorrect_hidden_states.pt")  # [M, hidden_dim]
    # steering_vector = compute_steering_vector(correct_hs, incorrect_hs)

    # ========================================================================
    # STEP 3: (Optional) Apply steering and regenerate
    # ========================================================================

    print("=" * 70)
    print("STEP 3: Steering Application (would regenerate with steered hidden states)")
    print("=" * 70)

    # TODO: Once steering is implemented, uncomment:
    # steering_output = steer_and_regenerate(
    #     integration_output=integration_output,
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     steering_vector=steering_vector,
    #     steer_layer=14,  # middle-late layer
    #     alpha=0.5,
    # )
    #
    # print(f"\n✓ Steering complete")
    # print(f"  Steered positions: {len(steering_output['steered_positions'])}")
    # print(f"  Regenerated trace length: {len(steering_output['regenerated_trace'])} chars")

    print("(Steering application deferred — requires model forward hook implementation)\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Data flow:
  1. Problem text
     ↓
  2. integration.integrate() → Integration output
     - Trace + segments + probe results
     ↓
  3. steering.steer_and_regenerate() → Steering output
     - Steered trace + re-probed results
     ↓
  4. (Optional) Evaluate final answer

Integration output structure:
{integration_output.keys()}

Planning span structure (first span if exists):
{integration_output['planning_spans'][0].keys() if integration_output['planning_spans'] else 'No planning spans'}
""")


if __name__ == "__main__":
    main()
