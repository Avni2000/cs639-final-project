

def compute_accuracy(preds, labels):
    correct = sum(p == y for p, y in zip(preds, labels))
    return correct / len(labels)