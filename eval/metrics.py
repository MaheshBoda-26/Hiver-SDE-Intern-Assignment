"""Evaluation metrics for the customer support agent."""

from typing import List, Dict, Any


def calculate_classification_metrics(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, float]:
    """Calculate accuracy for categorical predictions (e.g., intent)."""
    if not y_true:
        return {"accuracy": 0.0, "total": 0}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt.lower() == yp.lower())
    return {
        "accuracy": round(correct / len(y_true), 4),
        "total_samples": len(y_true),
        "correct_predictions": correct,
    }


def calculate_escalation_metrics(
    y_true_escalate: List[bool], y_pred_escalate: List[bool]
) -> Dict[str, float]:
    """Calculate Precision, Recall, F1, False Escalation Rate, and Missed Escalation Rate."""
    tp = sum(1 for yt, yp in zip(y_true_escalate, y_pred_escalate) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true_escalate, y_pred_escalate) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_escalate, y_pred_escalate) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true_escalate, y_pred_escalate) if not yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true_escalate) if y_true_escalate else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_escalation_rate": round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
        "missed_escalation_rate": round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
    }


def compute_grounding_overlap(prediction: str, reference: str) -> float:
    """Compute word token overlap Jaccard coefficient between prediction and reference answer."""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())

    if not ref_words:
        return 0.0

    overlap = pred_words.intersection(ref_words)
    return round(len(overlap) / len(pred_words.union(ref_words)), 4)
