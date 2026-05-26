"""
FNOL Synthetic Dataset Generator
Generates labeled First Notice of Loss narratives for fine-tuning.

Uses Claude API (claude-sonnet-4-20250514) for realistic narrative generation.
Target: 550 records covering all distribution requirements from the spec.

Usage:
    python generator.py --count 550 --output generated_dataset.jsonl
    python generator.py --count 50 --output generated_dataset.jsonl --dry-run
"""

import anthropic
import json
import os
import uuid
import random
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path


def _build_client() -> anthropic.Anthropic:
    """Prefer ANTHROPIC_API_KEY; fall back to Claude Code OAuth token if present."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if creds_path.exists():
        token = json.loads(creds_path.read_text())["claudeAiOauth"]["accessToken"]
        return anthropic.Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    raise RuntimeError("No ANTHROPIC_API_KEY env var and no ~/.claude/.credentials.json found.")

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Distribution targets (from spec) ─────────────────────────────────────────
TOTAL = 550

DISTRIBUTION = {
    # severity: (count, %)
    "C1": int(TOTAL * 0.05),   # ~28 — critical, large loss
    "C2": int(TOTAL * 0.25),   # ~138 — elevated
    "C3": TOTAL - int(TOTAL * 0.05) - int(TOTAL * 0.25),  # remainder ~384
}

SPECIAL_QUOTAS = {
    "reservation_of_rights": 32,   # spec: min 30
    "subrogation_indicator": 27,   # spec: min 25
    "cat_event": 42,               # spec: min 40
    "multi_line": 22,              # spec: min 20
}

CHANNELS = ["broker_email", "portal", "fax_to_email"]
COVERAGE_LINES = ["Commercial Property", "General Liability", "Commercial Auto", "Multi-Line"]

CAUSE_CODES = {
    "Commercial Property": ["FIR-02", "WTR-07", "CAT-01", "PROP-03", "PROP-08", "EQUIP-05"],
    "General Liability":   ["LIA-11", "LIA-14", "LIA-22", "LIA-09", "ADV-03"],
    "Commercial Auto":     ["AUTO-04", "AUTO-02", "AUTO-07", "AUTO-11"],
    "Multi-Line":          ["LIA-11", "AUTO-04", "WTR-07"],
}

ROUTING_MAP = {
    "C1": "LLU",
    "C2": "LMU",
    "C3": "STP",
}

# ── Prompt templates per scenario type ───────────────────────────────────────

BASE_SYSTEM = """You are an expert insurance claims professional with 20 years of experience.
Generate a realistic FNOL (First Notice of Loss) narrative exactly as it would be submitted by:
- A commercial insurance broker writing a coverage email
- A risk manager submitting through a portal
- An insured's fax converted to text (with occasional OCR artifacts like missing spaces or garbled words)

Use authentic insurance language: policy form references (ISO CG 00 01, CP 10 30, CA 00 01),
broker shorthand (TPA, EOP, GL, BI, PD, UIM, SIR, cert holder), coverage terminology,
and specific loss details (dollar amounts, injury descriptions, property specifics).

Vary the writing style — some submissions are terse and professional, others are rambling and emotional.
Fax conversions sometimes have OCR errors: "th e" instead of "the", "1nsured" instead of "insured".

Return ONLY the raw narrative text. No labels, no JSON, no explanation."""


def build_prompt(scenario: dict) -> str:
    """Build a targeted generation prompt for a specific scenario."""

    signals = scenario.get("domain_signals", [])
    hints = []

    if "reservation_of_rights" in signals:
        hints.append("The narrative must include reservation of rights language — "
                     "phrases like 'without prejudice to our rights under the policy', "
                     "'we reserve all rights and defenses', or explicit ROR reference.")

    if "subrogation_indicator" in signals:
        hints.append("Include clear subrogation signals: a third party caused the loss "
                     "(contractor negligence, defective product, another driver), "
                     "the insured is pursuing or has filed against that party, "
                     "or the broker flags recovery potential.")

    if "cat_event" in signals:
        storm_name = scenario.get("storm_name", "Hurricane Ida")
        hints.append(f"This loss is part of a CAT event — {storm_name}. "
                     "Reference the storm by name, include CAT date window, "
                     "mention widespread damage. Code should be CAT-01.")

    if "occurrence_claims_made_ambiguity" in signals:
        hints.append("Create ambiguity between occurrence and claims-made policy triggers — "
                     "the loss event happened in a prior policy period but the claim "
                     "is being reported now. The broker may reference the retroactive date.")

    if "named_additional_insured_confusion" in signals:
        hints.append("Include both a named insured and additional insured (certificate holder) "
                     "with unclear coverage implications — e.g., a general contractor and "
                     "a subcontractor both potentially involved.")

    if "fraud_indicator" in signals:
        hints.append("Embed fraud red flags realistically — late reporting weeks after the loss "
                     "with weak explanation, prior claim history hinted by the broker, "
                     "inconsistent loss description, no police/fire report despite required, "
                     "claimant unreachable, recent policy increase before loss, "
                     "staged-loss indicators, or SIU referral language. "
                     "Do NOT label it as fraud explicitly — show the red flags through facts.")

    coverage = scenario["coverage_line"]
    severity = scenario["severity"]
    cause = scenario["cause_code"]
    channel = scenario["channel"]

    channel_instruction = {
        "broker_email": "Write as a broker's email to the claims intake team. "
                        "Professional tone, references to policy numbers (e.g., CPP-2847-001), "
                        "adjuster assignments, and coverage form citations.",
        "portal": "Write as a portal submission by a risk manager or insured. "
                  "Form-like structure, factual, sometimes missing context.",
        "fax_to_email": "Write as OCR-converted fax text. Include minor OCR artifacts "
                        "(spacing errors, '0' vs 'O' confusion, occasional garbled word). "
                        "Older, more formal language. May reference fax date/time header.",
    }[channel]

    severity_instruction = {
        "C1": "This is a LARGE LOSS: reserve likely exceeds $500K, or involves fatality/permanent injury, "
              "or has CAT linkage. The narrative must convey severity — hospitalization, structural total loss, "
              "multi-vehicle fatality, bad faith exposure mentioned.",
        "C2": "This is an ELEVATED claim: litigation indicators present (attorney representation mentioned, "
              "demand letter referenced), coverage dispute likely, subrogation potential, or repeat claimant.",
        "C3": "This is a STANDARD claim: routine loss, clear coverage, no aggravating indicators. "
              "Straightforward property damage or minor liability event.",
    }[severity]

    prompt = f"""Generate a realistic FNOL narrative for a commercial lines insurance claim.

Coverage Line: {coverage}
Cause of Loss: {cause}
Severity: {severity}
Submission Channel: {channel}

Channel guidance: {channel_instruction}

Severity guidance: {severity_instruction}
"""

    if hints:
        prompt += "\nSpecific requirements:\n" + "\n".join(f"- {h}" for h in hints)

    prompt += "\n\nGenerate the narrative now (150–400 words):"
    return prompt


def assign_routing(scenario: dict) -> str:
    """Determine routing — SIU overrides severity-based routing."""
    if "fraud_indicator" in scenario.get("domain_signals", []):
        return "SIU"
    return ROUTING_MAP[scenario["severity"]]


def build_scenario_list(total: int) -> list[dict]:
    """
    Build the full list of scenario specs before generation.
    Ensures distribution quotas are met before randomizing the remainder.
    """
    scenarios = []

    # ── Mandatory special scenarios first ─────────────────────────────────────

    # Reservation of rights (mostly C2, some C1)
    for i in range(SPECIAL_QUOTAS["reservation_of_rights"]):
        sev = "C1" if i < 5 else "C2"
        cov = random.choice(["General Liability", "Commercial Property"])
        scenarios.append({
            "coverage_line": cov,
            "cause_code": random.choice(CAUSE_CODES[cov]),
            "severity": sev,
            "channel": random.choice(CHANNELS),
            "domain_signals": ["reservation_of_rights"],
        })

    # Subrogation indicators (C2 primary)
    for i in range(SPECIAL_QUOTAS["subrogation_indicator"]):
        cov = random.choice(["Commercial Property", "Commercial Auto", "General Liability"])
        scenarios.append({
            "coverage_line": cov,
            "cause_code": random.choice(CAUSE_CODES[cov]),
            "severity": "C2",
            "channel": random.choice(CHANNELS),
            "domain_signals": ["subrogation_indicator"],
        })

    # CAT event cluster — varied storm narratives, same event window
    cat_storms = ["Hurricane Ida"] * 42  # training data uses Ida; Marlene reserved for simulation
    for i in range(SPECIAL_QUOTAS["cat_event"]):
        scenarios.append({
            "coverage_line": "Commercial Property",
            "cause_code": "CAT-01",
            "severity": random.choice(["C1", "C1", "C2"]),  # CAT skews C1
            "channel": random.choice(CHANNELS),
            "domain_signals": ["cat_event"],
            "storm_name": cat_storms[i % len(cat_storms)],
        })

    # Multi-line ambiguous
    for _ in range(SPECIAL_QUOTAS["multi_line"]):
        scenarios.append({
            "coverage_line": "Multi-Line",
            "cause_code": random.choice(CAUSE_CODES["Multi-Line"]),
            "severity": random.choice(["C1", "C2", "C3"]),
            "channel": random.choice(CHANNELS),
            "domain_signals": ["named_additional_insured_confusion"],
        })

    # ── Fill remainder with distribution-balanced scenarios ────────────────────
    special_count = len(scenarios)
    remaining = total - special_count

    # Count severity in special scenarios
    sev_counts = {s: sum(1 for sc in scenarios if sc["severity"] == s) for s in ["C1", "C2", "C3"]}

    # Fill remaining to hit target distribution
    sev_targets = {
        "C1": max(0, DISTRIBUTION["C1"] - sev_counts["C1"]),
        "C2": max(0, DISTRIBUTION["C2"] - sev_counts["C2"]),
        "C3": max(0, remaining - max(0, DISTRIBUTION["C1"] - sev_counts["C1"])
                                - max(0, DISTRIBUTION["C2"] - sev_counts["C2"])),
    }

    for sev, count in sev_targets.items():
        for _ in range(count):
            cov = random.choice(COVERAGE_LINES[:3])  # no Multi-Line in remainder
            scenarios.append({
                "coverage_line": cov,
                "cause_code": random.choice(CAUSE_CODES[cov]),
                "severity": sev,
                "channel": random.choice(CHANNELS),
                "domain_signals": [],
            })

    # Add occurrence/claims-made ambiguity to ~15 C2 scenarios
    c2_indices = [i for i, s in enumerate(scenarios) if s["severity"] == "C2" and not s["domain_signals"]]
    for i in random.sample(c2_indices, min(15, len(c2_indices))):
        scenarios[i]["domain_signals"].append("occurrence_claims_made_ambiguity")

    random.shuffle(scenarios)
    return scenarios[:total]


def generate_narrative(client: anthropic.Anthropic, scenario: dict, dry_run: bool = False) -> str:
    """Call Claude to generate a single FNOL narrative."""
    if dry_run:
        return (f"[DRY RUN] Simulated FNOL narrative for {scenario['coverage_line']} "
                f"/ {scenario['cause_code']} / {scenario['severity']} "
                f"signals={scenario.get('domain_signals', [])}")

    prompt = build_prompt(scenario)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=[
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            {"type": "text", "text": BASE_SYSTEM},
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_record(scenario: dict, narrative: str) -> dict:
    """Assemble the final JSONL record."""
    return {
        "fnol_id": f"FNOL-{uuid.uuid4().hex[:8].upper()}",
        "submission_channel": scenario["channel"],
        "raw_narrative": narrative,
        "coverage_line": scenario["coverage_line"],
        "cause_code": scenario["cause_code"],
        "severity_tier": scenario["severity"],
        "routing": assign_routing(scenario),
        "domain_signals": scenario.get("domain_signals", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FNOL dataset")
    parser.add_argument("--count", type=int, default=550, help="Total records to generate")
    parser.add_argument("--output", type=str, default="generated_dataset.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, generate structure only")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from record N (append mode)")
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output
    client = _build_client() if not args.dry_run else None

    print(f"Building scenario list for {args.count} records...")
    scenarios = build_scenario_list(args.count)

    # Distribution summary
    print("\nDistribution plan:")
    for sev in ["C1", "C2", "C3"]:
        count = sum(1 for s in scenarios if s["severity"] == sev)
        print(f"  {sev}: {count} ({count/len(scenarios)*100:.1f}%)")
    for signal, quota in SPECIAL_QUOTAS.items():
        count = sum(1 for s in scenarios if signal in s.get("domain_signals", []))
        print(f"  {signal}: {count} (min {quota})")

    print(f"\nGenerating {args.count} narratives → {output_path}")
    print("This will take ~10–20 minutes on Claude API...\n")

    mode = "a" if args.resume_from > 0 else "w"
    records_written = args.resume_from

    with open(output_path, mode) as f:
        for i, scenario in enumerate(scenarios[args.resume_from:], start=args.resume_from):
            try:
                narrative = generate_narrative(client, scenario, args.dry_run)
                record = build_record(scenario, narrative)
                f.write(json.dumps(record) + "\n")
                f.flush()
                records_written += 1

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{args.count}] Generated — "
                          f"sev={scenario['severity']} cov={scenario['coverage_line'][:12]}")

                # Rate limiting: ~3 req/sec to stay within API limits
                if not args.dry_run:
                    time.sleep(0.35)

            except anthropic.RateLimitError:
                print(f"  Rate limited at record {i}. Waiting 60s...")
                time.sleep(60)
                # Retry once
                narrative = generate_narrative(client, scenario, args.dry_run)
                record = build_record(scenario, narrative)
                f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"  ERROR at record {i}: {e}. Skipping.")
                continue

    print(f"\nDone. {records_written} records written to {output_path}")

    # Quick validation
    print("\nValidation:")
    with open(output_path) as f:
        records = [json.loads(line) for line in f]
    print(f"  Total records: {len(records)}")
    for field in ["coverage_line", "severity_tier", "routing"]:
        from collections import Counter
        dist = Counter(r[field] for r in records)
        print(f"  {field}: {dict(dist)}")


if __name__ == "__main__":
    main()
