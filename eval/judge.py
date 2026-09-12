"""Evaluation runner and LLM judge for customer support pipeline."""

import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline import SupportAgentPipeline
from eval.metrics import (
    calculate_classification_metrics,
    calculate_escalation_metrics,
    compute_grounding_overlap,
)


def run_evaluation(
    golden_csv_path: str = "eval/golden_set.csv",
    output_report_path: str = "eval/eval_results.json",
) -> Dict[str, Any]:
    """Runs pipeline over the golden benchmark set and records evaluation metrics."""
    golden_path = Path(golden_csv_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found at: {golden_path}")

    pipeline = SupportAgentPipeline()

    records = []
    with open(golden_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Starting evaluation on {len(records)} golden benchmark samples...\n")

    y_true_intent: List[str] = []
    y_pred_intent: List[str] = []
    y_true_escalate: List[bool] = []
    y_pred_escalate: List[bool] = []
    grounding_scores: List[float] = []
    latencies: List[float] = []

    detailed_results = []

    for row in records:
        query = row["query"]
        expected_intent = row["expected_intent"]
        expected_escalate = row["expected_escalate"].strip().lower() == "true"
        reference_answer = row.get("reference_answer", "")

        start_time = time.time()
        res = pipeline.process(query)
        latency = round(time.time() - start_time, 4)
        latencies.append(latency)

        y_true_intent.append(expected_intent)
        y_pred_intent.append(res.intent)
        y_true_escalate.append(expected_escalate)
        y_pred_escalate.append(res.is_escalated)

        overlap = compute_grounding_overlap(res.reply, reference_answer)
        grounding_scores.append(overlap)

        detailed_results.append({
            "id": row.get("id"),
            "query": query,
            "expected_intent": expected_intent,
            "predicted_intent": res.intent,
            "expected_escalate": expected_escalate,
            "predicted_escalate": res.is_escalated,
            "escalation_reason": res.escalation_reason,
            "escalation_queue": res.escalation_queue,
            "reply": res.reply,
            "grounding_overlap": overlap,
            "latency_sec": latency,
        })

    intent_metrics = calculate_classification_metrics(y_true_intent, y_pred_intent)
    escalate_metrics = calculate_escalation_metrics(y_true_escalate, y_pred_escalate)
    avg_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0
    avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.0

    summary = {
        "benchmark_samples": len(records),
        "intent_accuracy": intent_metrics["accuracy"],
        "escalation_precision": escalate_metrics["precision"],
        "escalation_recall": escalate_metrics["recall"],
        "escalation_f1": escalate_metrics["f1_score"],
        "false_escalation_rate": escalate_metrics["false_escalation_rate"],
        "missed_escalation_rate": escalate_metrics["missed_escalation_rate"],
        "avg_grounding_overlap": avg_grounding,
        "avg_latency_seconds": avg_latency,
        "metrics_breakdown": {
            "intent": intent_metrics,
            "escalation": escalate_metrics,
        },
        "sample_details": detailed_results,
    }

    # Save results to json
    output_path = Path(output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Samples Evaluated:      {len(records)}")
    print(f"Intent Classification Acc:   {intent_metrics['accuracy'] * 100:.1f}%")
    print(f"Escalation Precision:         {escalate_metrics['precision'] * 100:.1f}%")
    print(f"Escalation Recall:            {escalate_metrics['recall'] * 100:.1f}%")
    print(f"Escalation F1-Score:          {escalate_metrics['f1_score'] * 100:.1f}%")
    print(f"False Escalation Rate:        {escalate_metrics['false_escalation_rate'] * 100:.1f}%")
    print(f"Missed Escalation Rate:       {escalate_metrics['missed_escalation_rate'] * 100:.1f}%")
    print(f"Average Latency:              {avg_latency * 1000:.2f} ms")
    print(f"Results saved to:             {output_report_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_evaluation()
