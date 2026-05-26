# Data Card — FNOL Triage Synthetic Dataset

## Overview

| Field | Value |
|---|---|
| **Dataset name** | fnol-triage-synthetic-v1 |
| **Version** | 1.0 |
| **Records** | 550 |
| **Format** | JSONL (one JSON object per line) |
| **Generated** | Claude claude-sonnet-4-20250514 via Anthropic API |
| **Purpose** | Fine-tuning instruction dataset for commercial lines FNOL triage |
| **License** | Internal use only — not for redistribution |

---

## Schema

```json
{
  "fnol_id": "FNOL-A3F2C1D8",
  "submission_channel": "broker_email | portal | fax_to_email",
  "raw_narrative": "string (150–400 words)",
  "coverage_line": "Commercial Property | General Liability | Commercial Auto | Multi-Line",
  "cause_code": "string (e.g. WTR-07, CAT-01, LIA-11)",
  "severity_tier": "C1 | C2 | C3",
  "routing": "LLU | LMU | STP | SIU",
  "domain_signals": ["list of signal strings"]
}
```

---

## Class Distributions

### Severity Tier

| Tier | Count | % | Notes |
|---|---|---|---|
| C1 (Critical) | ~28 | ~5% | Reflects real-world distribution — large loss events are rare |
| C2 (Elevated) | ~138 | ~25% | Litigation, subrogation, coverage disputes |
| C3 (Standard) | ~384 | ~70% | Routine losses, clear coverage |

### Coverage Line

| Line | Count | % |
|---|---|---|
| Commercial Property | ~200 | ~36% |
| General Liability | ~160 | ~29% |
| Commercial Auto | ~140 | ~26% |
| Multi-Line | ~50 | ~9% |

### Routing Decision

| Routing | Count | Notes |
|---|---|---|
| STP | ~384 | All C3 unless fraud indicators present |
| LMU | ~138 | All C2 |
| LLU | ~28 | All C1 |
| SIU | <10 | Fraud indicators — intentionally rare |

### Submission Channel

| Channel | Count | % |
|---|---|---|
| broker_email | ~220 | ~40% |
| portal | ~180 | ~33% |
| fax_to_email | ~150 | ~27% |

---

## Special Domain Signal Quotas

These were mandated by the spec and enforced hard in the generator:

| Signal | Count | Min Required | Notes |
|---|---|---|---|
| `reservation_of_rights` | 32 | 30 | Skewed toward C1/C2, GL and CP lines |
| `subrogation_indicator` | 27 | 25 | Third-party negligence, recovery potential flagged |
| `cat_event` | 42 | 40 | All reference Hurricane Ida; CAT-01 cause code |
| `multi_line` (named_additional_insured_confusion) | 22 | 20 | GL + CP overlap, subcontractor scenarios |
| `occurrence_claims_made_ambiguity` | ~15 | — | Retroactive date ambiguity in C2 scenarios |

---

## Generation Methodology

1. **Scenario planning** — `build_scenario_list()` constructs the full list of 550 scenario specs before any API call, enforcing distribution targets and special quotas deterministically (seed=42).

2. **Targeted prompting** — Each scenario maps to a tailored prompt that specifies coverage line, cause code, severity, channel, and any required domain signals. The system prompt conditions the model to write as an authentic insurance professional.

3. **Channel differentiation** — Three distinct writing personas:
   - `broker_email`: professional, references policy numbers and ISO form citations
   - `portal`: structured, risk manager voice, sometimes terse
   - `fax_to_email`: OCR artifacts simulated (spacing errors, character substitutions)

4. **Post-generation validation** — `main()` prints distribution stats at completion. Manual review of ~50 random samples was performed.

---

## Known Limitations

1. **CAT clustering only covers Hurricane Ida** — The simulation component introduces "Hurricane Marlene" as an out-of-distribution test. The model will not have seen this storm name in training, which is intentional and used to stress-test generalization.

2. **C1 sample size is small (~28 records)** — Reflects realistic distribution but creates class imbalance challenges. The evaluation pipeline uses stratified metrics and the cost matrix applies asymmetric penalties to compensate.

3. **SIU labels are rare** — Fraud indicators were not explicitly targeted in generation. A future version should add 15–20 explicit SIU-routed examples with synthetic fraud signals (inconsistent dates, prior claims references, suspicious injury descriptions).

4. **No real PII** — All policy numbers, insured names, and claimant details are fictional. When real FNOL data enters the pipeline, a PII redaction step (presidio or equivalent) must run before any data reaches training infrastructure.

5. **Narrative length variance** — Fax-converted narratives average 30% shorter than broker emails due to OCR artifact simulation. This may affect tokenization statistics.

6. **Single-label classification assumption** — Multi-Line records receive one primary cause code. Real FNOL submissions sometimes have legitimately ambiguous multi-code assignments that this schema does not capture.

---

## Recommended Train/Val/Test Split

Given the C1 class imbalance, a **stratified split** on `severity_tier` is required:

| Split | % | Approx Records | Notes |
|---|---|---|---|
| Train | 70% | 385 | Used for fine-tuning |
| Validation | 15% | 82 | Early stopping signal |
| Test | 15% | 83 | Held-out evaluation only |

Do **not** use random splits without stratification — a random 15% test set could contain zero C1 examples.

---

## PII Architecture Note

Although this dataset is entirely synthetic, the pipeline is designed to receive real FNOL data in production. The following architectural controls are required before real data ingestion:

- **Presidio** (Microsoft) for PII detection and redaction: names, addresses, policy numbers, SSNs, phone numbers
- Redacted fields replaced with typed placeholders: `[INSURED_NAME]`, `[POLICY_NUMBER]`
- No raw FNOL text stored after training — only adapter weights
- Training infrastructure must be air-gapped from external networks when processing real insured data
