"""
Stage 1 of the segmenter: boundary detection.

Design: we combine two signals for cuts.
  1. Sentence boundaries. Every sentence ends a span.
  2. Reflection tokens. Sentences that START with
     one of these transitions words that triggers a cut b/c reasoning models use them as pivot points
     for reflection and strategy shifts.

We use a regex-based sentence splitter.

"""
import re
from typing import List, Tuple

# List + some additional marker from paper.
REFLECTION_TOKENS = {
    "<think>", "wait", "but", "okay", "hmm", "albeit", "however", "yet",
    "still", "nevertheless", "though", "meanwhile", "whereas", "alternatively",
    # Planning / strategy markers (added on top of the paper's list)
    "first", "next", "instead", "therefore", "so", "actually", "let me",
    "let's", "i'll", "i will", "i should", "i need", "another approach",
    "on second thought", "hold on",
}

# Regex that matches any trigger phrase at the start of a sentence
_TRIGGER_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(tok) for tok in sorted(REFLECTION_TOKENS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Simple sentence splitter. Splits on .!? followed by whitespace + capital letter,
# plus double newlines. Not perfect tho
_SENT_END = re.compile(r"([.!?])\s+(?=[A-Z<\[\(])|\n{2,}")



def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into sentences using regex.
    Returns a list of sentences.
    """
    sentences = []
    start_idx = 0
    
    # Split text into sentences based on the defined regex pattern.
    for match in _SENT_END.finditer(text):
        raw = text[start_idx:match.end()]
        sentence = raw.strip()
        
        if sentence:
            lead = len(raw) - len(raw.lstrip())
            span_start_idx = start_idx + lead
            span_end_idx = span_start_idx + len(sentence)
            sentences.append((span_start_idx, span_end_idx, sentence))
        
        start_idx = match.end()
   
    # Then handle the last sentence after the final punctuation, if any.
    if start_idx < len(text):
        
        raw = text[start_idx:]
        sentence = raw.strip()
        
        if sentence:
            lead = len(raw) - len(raw.lstrip())
            sen_start_idx = start_idx + lead
            sen_end_idx = sen_start_idx + len(sentence)
            sentences.append((sen_start_idx, sen_end_idx, sentence))
    return sentences




def is_trigger_sentence(sentence: str) -> bool:
    """
    Determine if a sentence is a trigger sentence based on 
    whether it starts with any of the REFLECTION_TOKENS.
    """
    return bool(_TRIGGER_RE.match(sentence))




def detect_spans(text: str) -> List[tuple[int, int, str, bool]]:
    """
    Detect candidate reasoning spans in the text, or simply put it,
    detect sentence boundaries => split text into sentences, 
    and mark which sentences are trigger sentences based on 
    REFLECTION_TOKENS.

    Returns a list of tuples: (start_char, end_char, sentence_text, is_trigger).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    else:
        spans = []
        curr_start, curr_end, curr_text = sentences[0]
        curr_trigger = is_trigger_sentence(curr_text)
    
        for start_char, end_char, sentence in sentences[1:]:
            if is_trigger_sentence(sentence):
                # Start a new span at this sentence.
                spans.append((curr_start, curr_end, curr_text, curr_trigger))
                curr_start, curr_end, curr_text = start_char, end_char, sentence
                curr_trigger = True
            else:
                # Merge this sentence into the current span.
                curr_end = end_char
                curr_text += " " + sentence
        spans.append((curr_start, curr_end, curr_text, curr_trigger))
        return spans




if __name__ == "__main__":    # Example usage
    example_text = """
    First, we need to find the derivative. The function is f(x) = x^2.
    So, the derivative is f'(x) = 2x. However, we also need to consider the second derivative.
    The second derivative is f''(x) = 2.
    """
    spans = detect_spans(example_text)
    for start, end, span, is_trigger in spans:
        print(f"Span: '{span}' (Trigger: {is_trigger})")
        print(f"Chars: ({start}, {end})")
        print()