"""
Generate reasoning traces from a reasoning model.

This is the raw material for everything downstream. You only need this to run once
(per problem set). The output is a JSONL file where each row is one trace.

Notes on scale:
  - The proposal calls for ~2000 problems * 20 traces = 40,000 traces.
  - For the SEGMENTER specifically you do not need anywhere near that many.
    500 traces is plenty, since the segmenter only needs diverse reasoning text
    and does not use hidden states at all.
  - Generate the big 40k run later, after the segmenter works, because that run
    also needs to capture hidden states (much heavier IO).

Usage:
    python -m data.generate_traces --n-problems 200 --traces-per 3
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from config import REASONING_MODEL, RAW_TRACES_PATH


def build_prompt(problem: str) -> str:
    """Wrap a math problem in the DeepSeek-R1 thinking format."""
    return (
        "Solve the following problem. Think step by step inside <think>...</think>, "
        "then give the final answer after </think>.\n\n"
        f"Problem: {problem}\n\n<think>\n"
    )


def extract_thinking(full_output: str) -> str:
    """Return only the content inside <think>...</think>."""
    start = full_output.find("<think>")
    end = full_output.find("</think>")
    if start == -1:
        return full_output
    start += len("<think>")
    if end == -1:
        return full_output[start:].strip()
    return full_output[start:end].strip()


def check_correct(generated: str, gold_answer: str) -> bool:
    """Very loose correctness check. Good enough for building segmenter data.
    Replace with the MATH benchmark's official grader when you train the probe."""
    gold = gold_answer.strip().rstrip(".")
    return gold in generated[-500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--traces-per", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=RAW_TRACES_PATH)
    ap.add_argument("--dataset", default="hendrycks/competition_math")
    args = ap.parse_args()

    print(f"Loading {REASONING_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(REASONING_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REASONING_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading dataset {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=0).select(range(args.n_problems))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w") as fout:
        for ex in tqdm(ds, desc="problems"):
            problem = ex["problem"]
            gold = ex.get("solution", "") or ex.get("answer", "")
            prompt = build_prompt(problem)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            for trace_i in range(args.traces_per):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=tok.eos_token_id,
                    )
                full = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
                thinking = extract_thinking(full)
                correct = check_correct(full, gold)

                row = {
                    "trace_id": f"p{n_written}",
                    "problem": problem,
                    "gold": gold,
                    "raw_output": full,
                    "thinking": thinking,
                    "correct": bool(correct),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_written += 1

    print(f"Wrote {n_written} traces to {args.out}")


if __name__ == "__main__":
    main()
