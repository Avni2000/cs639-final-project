"""
End-to-end inference: given a reasoning trace, return labeled spans.

This is what the Probe calls downstream. The important field for Probe training
is `end_char`, which is the precise point at the end of each strategy span where
the reasoning model's hidden state should be captured.

Usage:
    from inference.segmenter_pipeline import Segmenter
    seg = Segmenter()
    spans = seg("Let me think. First, I'll try substitution...")
    for s in spans:
        print(s.label, s.end_char, s.text[:60])
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_DIR
from data.boundary_detection import detect_spans, Span
from model.evaluate_segmenter import SegmenterClassifier


@dataclass
class LabeledSpan:
    trace_id: str
    span_idx: int
    start_char: int
    end_char: int
    text: str
    trigger: str | None
    label: str


class Segmenter:
    """Runs boundary detection then classification to produce labeled spans."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No trained segmenter at {model_dir}. Run model/train_segmenter.py first."
            )
        self.clf = SegmenterClassifier(model_dir)

    def __call__(self, trace_text: str, trace_id: str = "0") -> List[LabeledSpan]:
        spans = detect_spans(trace_text, trace_id=trace_id)
        if not spans:
            return []
        texts = [s.text for s in spans]
        labels = self.clf.predict(texts)
        return [
            LabeledSpan(
                trace_id=s.trace_id,
                span_idx=s.span_idx,
                start_char=s.start_char,
                end_char=s.end_char,
                text=s.text,
                trigger=s.trigger,
                label=lab,
            )
            for s, lab in zip(spans, labels)
        ]

    def strategy_boundaries(self, trace_text: str) -> List[int]:
        """Return just the char offsets at the END of each strategy span.
        These are the points the Probe wants to capture hidden states at."""
        return [sp.end_char for sp in self(trace_text) if sp.label == "strategy"]


if __name__ == "__main__":
    demo = (
        "Let me think about this. First, I'll try substitution since we have two equations. "
        "Substituting x = y + 2 into the second equation gives 3(y+2) + y = 14. "
        "Simplifying: 3y + 6 + y = 14, so 4y = 8, y = 2. "
        "Wait, let me double-check. If y = 2, then x = 4. "
        "Plugging back in: 3(4) + 2 = 14. Yes, that works."
    )
    seg = Segmenter()
    for sp in seg(demo):
        print(f"[{sp.label:>9}] end={sp.end_char:>4}  {sp.text[:70]}...")
