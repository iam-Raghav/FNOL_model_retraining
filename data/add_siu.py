"""
One-off: generate SIU (fraud-indicator) FNOL records and append to generated_dataset.jsonl.

Spec lists SIU as the fourth routing class, but the main generator did not produce any.
This script fills the gap with ~12 fraud-indicator records spread across coverage lines
and severity tiers (SIU routes regardless of severity).
"""
import json
import random
import time
from pathlib import Path

from generator import (
    CAUSE_CODES,
    CHANNELS,
    _build_client,
    build_record,
    generate_narrative,
)

random.seed(43)  # different from main run to avoid collision

OUTPUT = Path(__file__).parent / "generated_dataset.jsonl"
SIU_COUNT = 12

# Spread across coverage lines (Auto often has fraud); mix of severities
SIU_PLAN = [
    ("Commercial Auto",     "AUTO-04", "C2"),
    ("Commercial Auto",     "AUTO-02", "C2"),
    ("Commercial Auto",     "AUTO-04", "C3"),
    ("Commercial Auto",     "AUTO-07", "C2"),
    ("Commercial Property", "FIR-02",  "C1"),  # arson red flag
    ("Commercial Property", "FIR-02",  "C2"),
    ("Commercial Property", "WTR-07",  "C3"),
    ("Commercial Property", "PROP-08", "C2"),
    ("General Liability",   "LIA-11",  "C2"),  # staged slip-and-fall
    ("General Liability",   "LIA-11",  "C3"),
    ("General Liability",   "LIA-22",  "C2"),
    ("General Liability",   "LIA-09",  "C2"),
]

scenarios = []
for cov, code, sev in SIU_PLAN:
    scenarios.append({
        "coverage_line": cov,
        "cause_code": code,
        "severity": sev,
        "channel": random.choice(CHANNELS),
        "domain_signals": ["fraud_indicator"],
    })

client = _build_client()
print(f"Appending {len(scenarios)} SIU records to {OUTPUT}")

with open(OUTPUT, "a") as f:
    for i, scen in enumerate(scenarios, 1):
        try:
            narrative = generate_narrative(client, scen, dry_run=False)
            record = build_record(scen, narrative)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  [{i}/{len(scenarios)}] {scen['coverage_line'][:15]:15s} {scen['cause_code']:7s} {scen['severity']} -> SIU")
            time.sleep(0.35)
        except Exception as e:
            print(f"  ERROR at {i}: {e}")

print("Done.")
