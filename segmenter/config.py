"""
Central configuration for the segmenter pipeline.
Edit the paths and model names here rather than scattering them through scripts.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data_out"
DATA_DIR.mkdir(exist_ok=True)

# Label set. Start with 2 classes, flip USE_META to True later to add meta-reflection.
USE_META = False

if USE_META:
    LABELS = ["strategy", "execution", "reflection"]
else:
    LABELS = ["strategy", "execution"]

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# Base encoder for the distilled segmenter. Swap to a bigger model if you have GPU budget.
BASE_MODEL = "distilbert-base-uncased"
# Alternatives you can try with no other code changes:
#   "microsoft/deberta-v3-small"
#   "roberta-base"

# Reasoning model used to generate traces
REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# LLM judge (used only for offline data labeling, not at inference)
JUDGE_PROVIDER = "anthropic"   # or "openai"
JUDGE_MODEL_ANTHROPIC = "claude-sonnet-4-5"
JUDGE_MODEL_OPENAI = "gpt-4o-mini"

# File paths produced by the pipeline
RAW_TRACES_PATH = DATA_DIR / "raw_traces.jsonl"
SPANS_PATH = DATA_DIR / "spans.jsonl"
SEED_LABELS_PATH = DATA_DIR / "spans_seed_labeled.jsonl"
JUDGE_LABELS_PATH = DATA_DIR / "spans_judge_labeled.jsonl"
MODEL_DIR = ROOT / "checkpoints" / "segmenter"
MODEL_DIR.parent.mkdir(exist_ok=True)

# Training hyperparameters
MAX_LEN = 256
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 4
SEED = 42
