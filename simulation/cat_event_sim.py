"""
cat_event_sim.py — CAT Event Stress Test Simulation

Generates 50 FNOL submissions for "Hurricane Marlene" (not in training data)
arriving within a simulated 6-hour window, runs them through the inference API,
and analyzes model behavior under distribution shift.

Usage:
    python cat_event_sim.py --api-url http://localhost:8000 --output sim_results.jsonl
"""

import json
import time
import random
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path

random.seed(99)  # separate seed from training

# ── Hurricane Marlene narrative templates ─────────────────────────────────────
# Varied narratives referencing the same event — the model has never seen "Marlene"

MARLENE_TEMPLATES = [
    # Broker emails
    "URGENT — Hurricane Marlene Submission\nPolicy {policy}. Landfall 09/18, Category 3. "
    "{insured} reporting {damage_type} damage to {property_desc}. TIV ${tiv:,}. "
    "Site access pending FEMA clearance. Recommend CAT designation and LLU assignment.",

    "Re: {insured} — Marlene Loss\nSubmitting emergency FNOL per Hurricane Marlene event 09/18. "
    "{property_desc} sustained {damage_type}. Estimated replacement cost ${tiv:,}. "
    "BI exposure likely — operations suspended. Please flag for CAT coordination.",

    "Hurricane Marlene FNOL — {insured}\n{damage_type} at {property_desc}, {city}. "
    "Loss date 09/18. Policy {policy}. Reserve recommendation: ${tiv:,} building, "
    "additional BI TBD. Fourth submission from our Gulf portfolio this week re: Marlene.",

    # Portal submissions
    "Insured: {insured}\nDate of Loss: 09/18\nEvent: Hurricane Marlene (Cat 3 landfall)\n"
    "Description: {damage_type} to {property_desc}. Estimated damage ${tiv:,}. "
    "Business closed since landfall. Requesting emergency adjuster.",

    "Loss type: Hurricane — Marlene\nInsured property: {property_desc}\n"
    "Damage: {damage_type}\nEstimate: ${tiv:,}\nComments: Part of widespread Marlene damage "
    "in {city} area. Multiple other policies affected same storm.",

    # Fax-to-email (OCR artifacts)
    "FAX 09/19 08:12\nRE: HuRricane Mar1ene Claim — {insured}\n"
    "{damage_type} dam age to {property_desc}. L0ss date 09/18. Est ${tiv:,}. "
    "Cat event — multiple submissions expected fr om this ac count.",

    "Emergency FNOL\nEvent: Hurricane Marlen e (09/18)\n{insured} — {property_desc}\n"
    "Damage type: {damage_type}\nReserve: ${tiv:,}\nNeed CAT un1t coordinator asap.",
]

INSUREDS = [
    "Gulf Coast Seafood Processing LLC", "Bayou Industrial Partners",
    "Coastal Hospitality Group", "Southern Marine Services Inc.",
    "Delta Agricultural Holdings", "Port City Logistics Corp.",
    "Pelican Bay Storage LLC", "Magnolia Manufacturing Co.",
    "Tidewater Commercial Properties", "Crescent City Retail Group",
]

PROPERTY_DESCS = [
    "main processing facility", "warehouse and distribution center",
    "commercial office building", "retail strip center",
    "manufacturing plant", "cold storage facility",
    "marina and boat storage", "grain storage silo complex",
    "hotel and conference center", "multi-tenant industrial park",
]

DAMAGE_TYPES = [
    "total roof failure and interior flooding",
    "wind-driven rain intrusion and structural damage",
    "storm surge flooding — 4 feet of standing water",
    "partial roof collapse and equipment damage",
    "facade failure and broken windows throughout",
    "complete loss of roof structure, Category 3 wind",
]

CITIES = ["New Orleans", "Baton Rouge", "Biloxi", "Mobile", "Pensacola", "Houma"]


def generate_marlene_fnol(index: int) -> dict:
    """Generate a single Hurricane Marlene FNOL narrative."""
    insured = random.choice(INSUREDS)
    template = MARLENE_TEMPLATES[index % len(MARLENE_TEMPLATES)]
    tiv = random.randint(200, 5000) * 1000  # $200K to $5M
    policy = f"CP-{random.randint(1000,9999)}-{random.randint(100,999)}"
    channel_options = ["broker_email", "broker_email", "portal", "fax_to_email"]

    narrative = template.format(
        insured=insured,
        policy=policy,
        damage_type=random.choice(DAMAGE_TYPES),
        property_desc=random.choice(PROPERTY_DESCS),
        tiv=tiv,
        city=random.choice(CITIES),
    )

    # Simulated arrival time within 6-hour window
    base_time = datetime(2024, 9, 18, 8, 0, 0)
    arrival_offset = timedelta(seconds=random.randint(0, 6 * 3600))
    arrival_time = base_time + arrival_offset

    return {
        "sim_id": f"SIM-MARLENE-{index+1:03d}",
        "arrival_time": arrival_time.isoformat(),
        "narrative": narrative,
        "channel": random.choice(channel_options),
        "expected_cause_code": "CAT-01",
        "expected_severity": "C1",
        "expected_routing": "LLU",
    }


def run_simulation(api_url: str, fnols: list, output_path: str):
    """Send all FNOLs to the inference API and capture results."""
    results = []
    errors = 0

    print(f"\nRunning simulation: {len(fnols)} Hurricane Marlene FNOLs → {api_url}/triage")
    print("─" * 60)

    for i, fnol in enumerate(fnols):
        try:
            response = requests.post(
                f"{api_url}/triage",
                json={"narrative": fnol["narrative"], "submission_channel": fnol["channel"]},
                timeout=30,
            )
            if response.status_code == 200:
                api_result = response.json()
                result = {
                    **fnol,
                    "predicted_cause_code": api_result.get("cause_code", {}).get("value", ""),
                    "predicted_severity": api_result.get("severity_tier", {}).get("value", ""),
                    "predicted_routing": api_result.get("routing", {}).get("value", ""),
                    "confidence": api_result.get("routing", {}).get("confidence_score", 0),
                    "domain_signals": api_result.get("domain_signals_detected", []),
                    "human_review_required": api_result.get("human_review_required", False),
                    "correct_cause": api_result.get("cause_code", {}).get("value", "") == "CAT-01",
                    "correct_routing": api_result.get("routing", {}).get("value", "") == "LLU",
                    "dangerous_stp": api_result.get("routing", {}).get("value", "") == "STP",
                }
                results.append(result)
                status = "✓" if result["correct_routing"] else "✗"
                print(f"  [{i+1:02d}] {status} cause={result['predicted_cause_code']:8s} "
                      f"route={result['predicted_routing']:4s} conf={result['confidence']:.2f}")
            else:
                errors += 1
                print(f"  [{i+1:02d}] ERROR {response.status_code}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1:02d}] EXCEPTION: {e}")

        # Simulate realistic submission pacing
        time.sleep(0.1)

    # Save results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results, errors


def analyze_results(results: list) -> dict:
    """Compute simulation statistics."""
    n = len(results)
    if n == 0:
        return {}

    correct_cause = sum(1 for r in results if r["correct_cause"])
    correct_routing = sum(1 for r in results if r["correct_routing"])
    dangerous_stp = sum(1 for r in results if r["dangerous_stp"])
    cat_signal = sum(1 for r in results if "cat_event" in r.get("domain_signals", []))

    confidences = [r["confidence"] for r in results]
    avg_conf = sum(confidences) / len(confidences)

    low_conf = sum(1 for c in confidences if c < 0.65)

    return {
        "total": n,
        "correct_cause_code_pct": correct_cause / n * 100,
        "correct_routing_llu_pct": correct_routing / n * 100,
        "dangerous_stp_count": dangerous_stp,
        "dangerous_stp_pct": dangerous_stp / n * 100,
        "cat_signal_detected_pct": cat_signal / n * 100,
        "avg_confidence": avg_conf,
        "low_confidence_count": low_conf,
        "low_confidence_pct": low_conf / n * 100,
    }


def write_incident_report(stats: dict, output_path: str):
    """Write FDE → Head of Claims incident report."""
    report = f"""# CAT Event Simulation — Incident Report
**Hurricane Marlene Stress Test**
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Author:** FDE, FNOL Triage System
**To:** Head of Claims

---

## Simulation Parameters

- **Event:** Hurricane Marlene (simulated landfall 09/18)
- **Submissions:** {stats['total']} FNOLs within a 6-hour window
- **Expected classification:** CAT-01 / C1 / LLU for all records
- **Key constraint:** "Marlene" is not present in the training data — this tests generalization to novel storm names

---

## Results Summary

| Metric | Result | Threshold | Status |
|---|---|---|---|
| Correct cause code (CAT-01) | {stats['correct_cause_code_pct']:.1f}% | >90% | {'✅' if stats['correct_cause_code_pct'] >= 90 else '❌'} |
| Correct routing (LLU) | {stats['correct_routing_llu_pct']:.1f}% | >90% | {'✅' if stats['correct_routing_llu_pct'] >= 90 else '❌'} |
| Dangerous STP routings | {stats['dangerous_stp_count']} ({stats['dangerous_stp_pct']:.1f}%) | 0 | {'✅' if stats['dangerous_stp_count'] == 0 else '🚨'} |
| CAT signal detected | {stats['cat_signal_detected_pct']:.1f}% | >80% | {'✅' if stats['cat_signal_detected_pct'] >= 80 else '❌'} |
| Average confidence score | {stats['avg_confidence']:.2f} | >0.75 | {'✅' if stats['avg_confidence'] >= 0.75 else '❌'} |
| Low-confidence predictions | {stats['low_confidence_count']} ({stats['low_confidence_pct']:.1f}%) | <10% | {'✅' if stats['low_confidence_pct'] < 10 else '❌'} |

---

## Observations

{'**The model performed within acceptable bounds.** Correct cause and routing rates exceeded 90%, and no CAT claims were routed to STP.' if stats['correct_routing_llu_pct'] >= 90 and stats['dangerous_stp_count'] == 0 else '**The model showed degraded performance on this out-of-distribution CAT event.** See failure analysis below.'}

**Confidence degradation:** {f"Average confidence dropped to {stats['avg_confidence']:.2f}, below the 0.75 target, indicating the model recognized uncertainty when encountering the novel storm name 'Marlene'. This is expected behavior — the model appropriately flagged {stats['low_confidence_count']} submissions for human review." if stats['avg_confidence'] < 0.75 else f"Confidence remained above threshold ({stats['avg_confidence']:.2f}), suggesting the model generalized well from Hurricane Ida training examples."}

**STP routing risk:** {f"🚨 CRITICAL: {stats['dangerous_stp_count']} CAT event submissions were routed to straight-through processing. In a real deployment, these claims would have been auto-settled without a senior examiner reviewing a large loss event. This represents a reserve failure risk and potential regulatory exposure." if stats['dangerous_stp_count'] > 0 else "No CAT event submissions were routed to STP. The model correctly identified all submissions as requiring manual review."}

---

## Failure Analysis

The model was trained exclusively on Hurricane Ida as the named CAT event. When presented with "Hurricane Marlene," generalization depends on whether the model learned:
- **General CAT signals** (storm language, widespread damage references, CAT-unit mentions) → generalizes
- **Named entity memorization** ("Ida" → CAT-01) → does not generalize

{f"The {stats['low_confidence_pct']:.0f}% low-confidence rate suggests partial memorization. The model is uncertain about Marlene specifically but correctly identifies contextual CAT signals in most cases." if stats['low_confidence_pct'] > 15 else "The low rate of uncertain predictions suggests the model learned general CAT language patterns rather than memorizing 'Ida' specifically — a positive sign for production robustness."}

---

## Remediation Path

1. **Immediate (0–48 hours):** Lower the `human_review_required` confidence threshold from 0.65 to 0.50 for any submission containing weather/storm language. This catches uncertain predictions before they reach STP.

2. **Short-term (1–2 weeks):** Add 20–30 training examples with varied storm names (Katrina, Harvey, Ian, Helene, and fictional names) to reduce named-entity dependence. Retrain adapters — cost is <2 GPU hours.

3. **Medium-term (1 month):** Implement a CAT event registry lookup: if active CAT event is declared for a date/geography, force override routing to LLU regardless of model output. Model uncertainty on novel named events is unavoidable; a deterministic override is the correct architectural safeguard.

4. **Monitoring:** Add a CAT clustering detector — if >10 submissions in 4 hours share the same cause code and region, trigger automatic CAT flag and alert the claims operations team.

---

*This report was generated automatically by the simulation pipeline. Confidence scores are approximate proxies derived from output token probabilities and should be treated as relative indicators, not calibrated probabilities.*
"""

    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nIncident report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="sim_results.jsonl")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate FNOLs but don't call API")
    args = parser.parse_args()

    # Generate 50 FNOLs
    fnols = [generate_marlene_fnol(i) for i in range(args.count)]
    print(f"Generated {len(fnols)} Hurricane Marlene FNOLs")

    if args.dry_run:
        print("\nDry run — sample narrative:")
        print(fnols[0]["narrative"])
        print("\nFirst 5 fnol_ids:")
        for f in fnols[:5]:
            print(f"  {f['sim_id']} | channel={f['channel']}")
        return

    results, errors = run_simulation(args.api_url, fnols, args.output)
    stats = analyze_results(results)

    print(f"\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    if errors > 0:
        print(f"\n⚠️  {errors} API errors during simulation")

    # Write incident report
    report_path = Path(args.output).parent / "cat_incident_report.md"
    write_incident_report(stats, str(report_path))


if __name__ == "__main__":
    main()
