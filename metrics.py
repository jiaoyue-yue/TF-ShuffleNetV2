"""Positive-class metrics and rank-based AUC, using infested = 1."""
import math


def binary_metrics(labels, probabilities):
    labels = list(labels)
    probabilities = [float(p) for p in probabilities]
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same nonzero length")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be 0 or 1")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Expected finite class probabilities in [0, 1]")
    tp = sum(y == 1 and p >= .5 for y, p in zip(labels, probabilities))
    fn = sum(y == 1 and p < .5 for y, p in zip(labels, probabilities))
    tn = sum(y == 0 and p < .5 for y, p in zip(labels, probabilities))
    fp = sum(y == 0 and p >= .5 for y, p in zip(labels, probabilities))
    div = lambda a, b: a / b if b else 0.0
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    auc = None
    if tp + fn and tn + fp:
        ordered = sorted(zip(probabilities, labels))
        rank_sum = 0.0
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            rank_sum += (start + 1 + end) / 2 * sum(y for _, y in ordered[start:end])
            start = end
        positives, negatives = tp + fn, tn + fp
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, accuracy=div(tp + tn, len(labels)),
                precision=div(tp, tp + fp), recall=recall, specificity=specificity,
                f1=div(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(recall + specificity) / 2, roc_auc=auc)
