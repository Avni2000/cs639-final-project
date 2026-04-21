

from segmenter.data.boundary_detection import split_into_spans

def run_segmenter(trace, model):
    """
    Input:
        trace (str)
        model (trained classifier)

    Output:
        list of dicts:
        [
            {"span": str, "label": str},
            ...
        ]
    """
    spans = split_into_spans(trace)

    results = []
    for span in spans:
        label = model.predict(span)  # placeholder
        results.append({"span": span, "label": label})

    return results