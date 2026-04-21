
def select_probe_points(labeled_spans):
    """
    Input:
        labeled_spans = [{"span": str, "label": str}]

    Output:
        subset of spans to probe
    """
    selected = []

    for item in labeled_spans:
        if item["label"] in ["planning", "meta-reflection"]:
            selected.append(item)

    return selected