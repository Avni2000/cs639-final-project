"""
Stage 1 of the segmenter: boundary detection.

Given a reasoning trace (the text between <think>...</think>, or just the full CoT),
slice it into candidate spans. The classifier (stage 2) later assigns a label to each span.

Design: we combine two signals for cuts.
  1. Sentence boundaries. Every sentence ends a span.
  2. Reflection tokens (from arXiv:2506.12963 Table A.2). Sentences that START with
     one of these also trigger a cut, because reasoning models use them as pivots.

We intentionally overproduce boundaries. False positives are cheap: the classifier
will just label an extra execution span. False negatives (missing a real transition)
are the expensive kind, so we err toward more cuts.

We use a regex-based sentence splitter rather than NLTK. It is less clever than
punkt but has no data-download dependency, which matters for reproducibility on
cluster machines with no internet.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import List

# From arXiv:2506.12963 Appendix A.2, plus a few common planning markers.
REFLECTION_TOKENS = {
    "<think>", "wait", "but", "okay", "hmm", "albeit", "however", "yet",
    "still", "nevertheless", "though", "meanwhile", "whereas", "alternatively",
    # Planning / strategy markers (added on top of the paper's list)
    "first", "next", "instead", "therefore", "so", "actually", "let me",
    "let's", "i'll", "i will", "i should", "i need", "another approach",
    "on second thought", "hold on",
}

# Regex that matches any trigger phrase at the start of a sentence.
_TRIGGER_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(tok) for tok in sorted(REFLECTION_TOKENS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Simple sentence splitter. Splits on .!? followed by whitespace + capital letter,
# plus double newlines. Not perfect on abbreviations, but reasoning traces are
# fairly clean so this works well in practice.
_SENT_END = re.compile(r"([.!?])\s+(?=[A-Z<\[\(])|\n{2,}")


@dataclass
class Span:
    """A candidate reasoning span produced by boundary detection."""
    trace_id: str
    span_idx: int
    start_char: int
    end_char: int
    text: str
    trigger: str | None

    def to_dict(self):
        return asdict(self)


def _sentence_tokenize(text: str) -> List[tuple[int, int, str]]:
    """Return a list of (start_char, end_char, sentence_text) tuples."""
    sentences = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1) if m.group(1) else m.start()
        chunk = text[start:end]
        sent = chunk.strip()
        if sent:
            lead = len(chunk) - len(chunk.lstrip())
            actual_start = start + lead
            actual_end = actual_start + len(sent)
            sentences.append((actual_start, actual_end, sent))
        start = m.end()
    if start < len(text):
        tail = text[start:]
        sent = tail.strip()
        if sent:
            lead = len(tail) - len(tail.lstrip())
            actual_start = start + lead
            actual_end = actual_start + len(sent)
            sentences.append((actual_start, actual_end, sent))
    return sentences


def detect_spans(trace_text: str, trace_id: str = "0") -> List[Span]:
    """Split a trace into candidate spans.

    A new span starts whenever a sentence begins with a reflection token.
    Consecutive non-trigger sentences merge into a single span so execution runs
    (algebra, arithmetic) stay together.
    """
    sentences = _sentence_tokenize(trace_text)
    if not sentences:
        return []

    spans: List[Span] = []
    buf: List[tuple[int, int, str]] = []
    buf_trigger: str | None = None

    def flush():
        nonlocal buf, buf_trigger
        if not buf:
            return
        s = buf[0][0]
        e = buf[-1][1]
        txt = trace_text[s:e].strip()
        if txt:
            spans.append(Span(
                trace_id=trace_id,
                span_idx=len(spans),
                start_char=s,
                end_char=e,
                text=txt,
                trigger=buf_trigger,
            ))
        buf = []
        buf_trigger = None

    for (s, e, sent) in sentences:
        m = _TRIGGER_RE.match(sent)
        if m:
            flush()
            buf_trigger = m.group(0).strip()
        buf.append((s, e, sent))

    flush()
    return spans


if __name__ == "__main__":
    demo = (
        "Let me think about this. First, I'll try substitution since we have two equations. "
        "Substituting x = y + 2 into the second equation gives 3(y+2) + y = 14. "
        "Simplifying: 3y + 6 + y = 14, so 4y = 8, y = 2. "
        "Wait, let me double-check. If y = 2, then x = 4. "
        "Plugging back in: 3(4) + 2 = 14. Yes, that works."
    )
    for sp in detect_spans(demo, trace_id="demo"):
        print(f"[{sp.span_idx}] trigger={sp.trigger!r}")
        print(f"    {sp.text}")
        print()
