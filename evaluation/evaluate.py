"""
evaluate.py — FNOL Triage Evaluation Pipeline

Runs against the held-out test set and produces:
  1. Per-class P/R/F1 for all four output dimensions
  2. Macro and weighted F1
  3. Confusion matrices as seaborn heatmaps
  4. Asymmetric cost scores (routing + severity)
  5. Domain signal detection report (5 failure modes)
  6. HTML evaluation report readable by a non-technical claims manager

Usage:
    python evaluate.py \\
        --test-data ../data/test_split.jsonl \\
        --adapter-path ../training/adapters/final \\
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --output evaluation_report.html
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from cost_matrix import (
    routing_cost, severity_cost,
    ROUTING_COST_MATRIX, SEVERITY_COST_MATRIX,
    ROUTING_LABELS as ROUTING_COST_LABELS,
    SEVERITY_LABELS as SEVERITY_COST_LABELS,
)

# Output dimensions we evaluate
DIMENSIONS = ["coverage_line", "cause_code", "severity_tier", "routing"]

# The five failure modes from the spec
DOMAIN_SIGNALS = [
    "reservation_of_rights",
    "subrogation_indicator",
    "cat_event",
    "occurrence_claims_made_ambiguity",
    "named_additional_insured_confusion",
]


def load_model(base_model: str, adapter_path: str):
    """Load quantized base model + LoRA adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    return model, tokenizer


def run_inference(model, tokenizer, record: dict) -> dict:
    """Run model on a single FNOL record and parse JSON output."""
    import torch
    from training.finetune import INFERENCE_PROMPT  # reuse inference prompt

    prompt = INFERENCE_PROMPT.format(
        channel=record["submission_channel"],
        narrative=record["raw_narrative"],
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,    # low temp for deterministic structured output
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Parse JSON from generated text
    try:
        # Find first { ... } block
        match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: return empty dict (will count as wrong on all dimensions)
    return {}


def compute_dimension_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Compute P/R/F1 per class and macro/weighted averages."""
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    return report


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Render confusion matrix as seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def compute_domain_signal_metrics(records: list[dict], predictions: list[dict]) -> dict:
    """
    Compute precision and recall for each of the five domain signals.
    A signal is 'detected' if it appears in the predicted domain_signals list.
    """
    results = {}
    for signal in DOMAIN_SIGNALS:
        y_true = [int(signal in r.get("domain_signals", [])) for r in records]
        y_pred = [int(signal in p.get("domain_signals", [])) for p in predictions]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[signal] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(y_true),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return results


def plain_language_summary(metrics: dict, cost_metrics: dict, signal_metrics: dict) -> str:
    """
    Generate a plain-language interpretation for a non-technical claims manager.
    This is the automated narrative the spec asks for as a bonus.
    """
    routing_f1 = metrics.get("routing", {}).get("weighted avg", {}).get("f1-score", 0)
    severity_f1 = metrics.get("severity_tier", {}).get("weighted avg", {}).get("f1-score", 0)
    r_cost = cost_metrics.get("routing_cost", 0)

    stp_c1_errors = cost_metrics.get("c1_to_stp_count", 0)

    summary_parts = []

    # Overall routing quality
    if routing_f1 >= 0.90:
        summary_parts.append(
            f"<strong>Routing accuracy is strong</strong> (weighted F1: {routing_f1:.0%}). "
            "In 9 out of 10 cases, the model sends claims to the right team."
        )
    elif routing_f1 >= 0.75:
        summary_parts.append(
            f"<strong>Routing accuracy is acceptable but has room to improve</strong> "
            f"(weighted F1: {routing_f1:.0%}). Approximately 1 in 4 claims may be misrouted."
        )
    else:
        summary_parts.append(
            f"<strong>Routing accuracy needs improvement</strong> (weighted F1: {routing_f1:.0%}). "
            "Human review should remain mandatory until accuracy improves."
        )

    # C1 safety
    if stp_c1_errors == 0:
        summary_parts.append(
            "✅ <strong>No C1 (large loss) claims were routed to straight-through processing</strong> "
            "during evaluation. This is the most critical safety requirement and it was met."
        )
    else:
        summary_parts.append(
            f"⚠️ <strong>{stp_c1_errors} large loss claim(s) were incorrectly routed to "
            "automated settlement</strong>. This is a critical failure — human review must be "
            "enforced for all claims until this is resolved."
        )

    # Domain signals
    ror_recall = signal_metrics.get("reservation_of_rights", {}).get("recall", 0)
    sub_recall = signal_metrics.get("subrogation_indicator", {}).get("recall", 0)
    cat_recall = signal_metrics.get("cat_event", {}).get("recall", 0)

    summary_parts.append(
        f"The model detects reservation of rights language in {ror_recall:.0%} of cases "
        f"where it is present, subrogation indicators in {sub_recall:.0%} of cases, "
        f"and CAT event linkage in {cat_recall:.0%} of cases."
    )

    return " ".join(summary_parts)


def render_html_report(
    metrics: dict,
    cost_metrics: dict,
    signal_metrics: dict,
    chart_paths: dict,
    output_path: str,
    generated_at: str,
):
    """Render the evaluation results as a standalone HTML report."""

    signal_rows = ""
    for signal, m in signal_metrics.items():
        label = signal.replace("_", " ").title()
        color = "#2e7d32" if m["recall"] >= 0.75 else "#f57c00" if m["recall"] >= 0.5 else "#c62828"
        signal_rows += f"""
        <tr>
          <td>{label}</td>
          <td>{m['support']}</td>
          <td>{m['precision']:.3f}</td>
          <td style="color:{color};font-weight:bold">{m['recall']:.3f}</td>
          <td>{m['f1']:.3f}</td>
          <td>{m['tp']} / {m['tp']+m['fn']}</td>
        </tr>"""

    dimension_tables = ""
    for dim in DIMENSIONS:
        if dim not in metrics:
            continue
        report = metrics[dim]
        rows = ""
        for label, vals in report.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            rows += f"""
            <tr>
              <td>{label}</td>
              <td>{vals.get('precision', 0):.3f}</td>
              <td>{vals.get('recall', 0):.3f}</td>
              <td>{vals.get('f1-score', 0):.3f}</td>
              <td>{int(vals.get('support', 0))}</td>
            </tr>"""
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        dimension_tables += f"""
        <h3>{dim.replace('_', ' ').title()}</h3>
        <table>
          <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
          {rows}
          <tr style="background:#e8f5e9;font-weight:bold">
            <td>Macro Avg</td>
            <td>{macro.get('precision',0):.3f}</td>
            <td>{macro.get('recall',0):.3f}</td>
            <td>{macro.get('f1-score',0):.3f}</td>
            <td>—</td>
          </tr>
          <tr style="background:#e3f2fd;font-weight:bold">
            <td>Weighted Avg</td>
            <td>{weighted.get('precision',0):.3f}</td>
            <td>{weighted.get('recall',0):.3f}</td>
            <td>{weighted.get('f1-score',0):.3f}</td>
            <td>—</td>
          </tr>
        </table>"""

    def _render_cost_matrix_rows(matrix, labels):
        rows = ""
        for i, true_label in enumerate(labels):
            cells = ""
            for j, _ in enumerate(labels):
                v = int(matrix[i][j])
                if v == 0:
                    style = "background:#e8f5e9;color:#2e7d32;font-weight:bold"
                elif v >= 50:
                    style = "background:#ffebee;color:#c62828;font-weight:bold"
                elif v >= 20:
                    style = "background:#fff3e0;color:#e65100"
                else:
                    style = "color:#616161"
                cells += f'<td style="{style};text-align:center">{v}</td>'
            rows += f'<tr><th style="text-align:left">{true_label}</th>{cells}</tr>'
        return rows

    routing_matrix_rows = _render_cost_matrix_rows(ROUTING_COST_MATRIX, ROUTING_COST_LABELS)
    severity_matrix_rows = _render_cost_matrix_rows(SEVERITY_COST_MATRIX, SEVERITY_COST_LABELS)

    summary_text = plain_language_summary(metrics, cost_metrics, signal_metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FNOL Triage Model Evaluation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #212121; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
    h2 {{ color: #283593; margin-top: 40px; }}
    h3 {{ color: #3949ab; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
    th {{ background: #3949ab; color: white; padding: 8px 12px; text-align: left; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #f5f5f5; }}
    .summary-box {{ background: #e8eaf6; border-left: 4px solid #3949ab;
                   padding: 16px 20px; margin: 20px 0; border-radius: 4px; }}
    .cost-box {{ background: #fff3e0; border-left: 4px solid #e65100;
                padding: 16px 20px; margin: 20px 0; border-radius: 4px; }}
    .metric {{ display: inline-block; background: white; border: 1px solid #e0e0e0;
               border-radius: 6px; padding: 12px 20px; margin: 8px; text-align: center; }}
    .metric-value {{ font-size: 28px; font-weight: bold; color: #1a237e; }}
    .metric-label {{ font-size: 12px; color: #757575; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; margin: 12px 0; }}
    .generated {{ color: #9e9e9e; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>FNOL Triage Model — Evaluation Report</h1>
  <p class="generated">Generated: {generated_at} | Test set size: {cost_metrics.get('n_test', '?')} records</p>

  <h2>Executive Summary</h2>
  <div class="summary-box">{summary_text}</div>

  <h2>Key Metrics</h2>
  <div>
    <div class="metric">
      <div class="metric-value">{metrics.get('routing', {}).get('weighted avg', {}).get('f1-score', 0):.1%}</div>
      <div class="metric-label">Routing F1 (weighted)</div>
    </div>
    <div class="metric">
      <div class="metric-value">{metrics.get('severity_tier', {}).get('weighted avg', {}).get('f1-score', 0):.1%}</div>
      <div class="metric-label">Severity F1 (weighted)</div>
    </div>
    <div class="metric">
      <div class="metric-value">{cost_metrics.get('routing_cost', 0):.2f}</div>
      <div class="metric-label">Mean Routing Cost</div>
    </div>
    <div class="metric">
      <div class="metric-value">{cost_metrics.get('severity_cost', 0):.2f}</div>
      <div class="metric-label">Mean Severity Cost</div>
    </div>
    <div class="metric">
      <div class="metric-value">{cost_metrics.get('c1_to_stp_count', 0)}</div>
      <div class="metric-label">C1→STP Errors (must be 0)</div>
    </div>
  </div>

  <div class="cost-box">
    <strong>Cost-weighted scoring note:</strong> Standard F1 treats all misclassifications equally.
    The cost-weighted score reflects real business consequences — a C1 claim routed to STP
    carries 100x the penalty of a C3 claim routed to LLU. A model with high F1 but poor
    cost score is operationally dangerous despite appearing accurate by standard metrics.
    <br><br>
    Standard weighted F1 (routing): <strong>{metrics.get('routing', {}).get('weighted avg', {}).get('f1-score', 0):.3f}</strong>
    &nbsp;|&nbsp;
    Mean cost score (routing): <strong>{cost_metrics.get('routing_cost', 0):.2f}</strong>
    &nbsp;(lower is better; 0 = perfect)
    <br>
    Standard weighted F1 (severity): <strong>{metrics.get('severity_tier', {}).get('weighted avg', {}).get('f1-score', 0):.3f}</strong>
    &nbsp;|&nbsp;
    Mean cost score (severity): <strong>{cost_metrics.get('severity_cost', 0):.2f}</strong>
    &nbsp;(lower is better; 0 = perfect)
  </div>

  <h2>Cost Matrices</h2>
  <p>Asymmetric penalties — direction of the error matters. C1→STP (catastrophic auto-settlement
     of a large-loss claim) is penalized 100× more than C3→LLU (minor operational waste).
     Diagonal entries are 0 (correct prediction).</p>

  <h3>Routing Cost Matrix</h3>
  <table>
    <tr><th>True ↓ / Pred →</th>{''.join(f'<th>{l}</th>' for l in ROUTING_COST_LABELS)}</tr>
    {routing_matrix_rows}
  </table>

  <h3>Severity Cost Matrix</h3>
  <table>
    <tr><th>True ↓ / Pred →</th>{''.join(f'<th>{l}</th>' for l in SEVERITY_COST_LABELS)}</tr>
    {severity_matrix_rows}
  </table>

  <h2>Per-Dimension Classification Metrics</h2>
  {dimension_tables}

  <h2>Domain Signal Detection Report</h2>
  <p>These are the five failure modes called out in the system specification.
     Recall is the most important metric here — a missed signal means the model
     failed to route a claim correctly.</p>
  <table>
    <tr>
      <th>Signal</th><th>Support (test)</th>
      <th>Precision</th><th>Recall ⭐</th><th>F1</th><th>Detected / Total</th>
    </tr>
    {signal_rows}
  </table>

  <h2>Confusion Matrices</h2>
  {''.join(f'<img src="{p}" alt="{k} confusion matrix">' for k, p in chart_paths.items())}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="../data/test_split.jsonl")
    parser.add_argument("--adapter-path", default="../training/adapters/final")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="evaluation_report.html")
    parser.add_argument("--use-cached-predictions", default=None,
                        help="Path to cached predictions JSONL — skips model inference")
    args = parser.parse_args()

    # Load test records
    with open(args.test_data) as f:
        test_records = [json.loads(l) for l in f]
    print(f"Loaded {len(test_records)} test records")

    # Run inference (or load cached)
    if args.use_cached_predictions:
        with open(args.use_cached_predictions) as f:
            predictions = [json.loads(l) for l in f]
        print(f"Loaded {len(predictions)} cached predictions")
    else:
        print("Loading model for inference...")
        model, tokenizer = load_model(args.base_model, args.adapter_path)
        predictions = []
        for i, record in enumerate(test_records):
            pred = run_inference(model, tokenizer, record)
            predictions.append(pred)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(test_records)}] inference done")

        # Cache predictions
        cache_path = Path(args.output).parent / "predictions_cache.jsonl"
        with open(cache_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        print(f"Predictions cached to {cache_path}")

    # Compute metrics per dimension
    metrics = {}
    for dim in DIMENSIONS:
        y_true = [r.get(dim, "UNKNOWN") for r in test_records]
        y_pred = [p.get(dim, "UNKNOWN") for p in predictions]
        labels = sorted(set(y_true))
        metrics[dim] = compute_dimension_metrics(y_true, y_pred, labels)

    # Cost metrics
    y_true_routing = [r.get("routing", "") for r in test_records]
    y_pred_routing = [p.get("routing", "") for p in predictions]
    y_true_severity = [r.get("severity_tier", "") for r in test_records]
    y_pred_severity = [p.get("severity_tier", "") for p in predictions]

    c1_to_stp = sum(1 for t, p in zip(y_true_routing, y_pred_routing)
                    if t == "LLU" and p == "STP")

    cost_metrics = {
        "routing_cost": routing_cost(y_true_routing, y_pred_routing),
        "severity_cost": severity_cost(y_true_severity, y_pred_severity),
        "c1_to_stp_count": c1_to_stp,
        "n_test": len(test_records),
    }

    # Domain signal metrics
    signal_metrics = compute_domain_signal_metrics(test_records, predictions)

    # Confusion matrix plots
    chart_paths = {}
    charts_dir = Path(args.output).parent / "charts"
    charts_dir.mkdir(exist_ok=True)

    for dim in ["routing", "severity_tier", "coverage_line"]:
        y_true = [r.get(dim, "?") for r in test_records]
        y_pred = [p.get(dim, "?") for p in predictions]
        labels = sorted(set(y_true))
        chart_path = charts_dir / f"cm_{dim}.png"
        plot_confusion_matrix(y_true, y_pred, labels,
                              f"Confusion Matrix — {dim.replace('_', ' ').title()}",
                              chart_path)
        chart_paths[dim] = str(chart_path)

    # Render report
    render_html_report(
        metrics, cost_metrics, signal_metrics, chart_paths,
        args.output,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # Print summary to stdout
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for dim in DIMENSIONS:
        w_f1 = metrics[dim].get("weighted avg", {}).get("f1-score", 0)
        m_f1 = metrics[dim].get("macro avg", {}).get("f1-score", 0)
        print(f"  {dim:20s} weighted_f1={w_f1:.3f}  macro_f1={m_f1:.3f}")
    print(f"\n  Routing cost score:  {cost_metrics['routing_cost']:.2f}")
    print(f"  Severity cost score: {cost_metrics['severity_cost']:.2f}")
    print(f"  C1->STP errors:      {c1_to_stp} (MUST BE 0)")
    print(f"\nDomain signals:")
    for signal, m in signal_metrics.items():
        print(f"  {signal:40s} P={m['precision']:.2f}  R={m['recall']:.2f}")


if __name__ == "__main__":
    main()
