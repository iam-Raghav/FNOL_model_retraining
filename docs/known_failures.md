# Known Failure Modes

Honest list of things this system does not handle well.
Written for the engineer who inherits this six months from now.

---

## 1. Novel Named CAT Events (HIGH RISK)

**What fails:** The model was trained on Hurricane Ida as the sole named storm event. When presented with a novel storm name ("Hurricane Marlene", "Typhoon Keiko"), the model may produce lower-confidence predictions or incorrectly cause-code the claim.

**Why it happens:** Named entity recognition. The model may have partially learned "Ida → CAT-01" rather than "storm context → CAT-01." Distribution shift on proper nouns is a well-documented LLM failure mode.

**Mitigation:** (1) Add diverse storm names to training data. (2) Implement a deterministic CAT override: if active CAT event declared for the loss date/geography, force LLU routing regardless of model output. (3) CAT clustering detector for burst submission windows.

---

## 2. Occurrence vs. Claims-Made Policy Trigger (MEDIUM-HIGH RISK)

**What fails:** When a claim is reported in a different policy period than when the loss occurred, and the insured has switched between occurrence-based and claims-made policies, the model frequently cannot determine which policy responds.

**Why it happens:** This requires cross-referencing two temporal frames (loss date, claim date) against two policy structures — the training narratives present this ambiguity but the model doesn't reliably detect the retroactive date signal as a coverage-trigger indicator rather than just a date reference.

**Mitigation:** Add a specific `occurrence_claims_made_ambiguity` domain signal with recall-focused training examples. Current precision is acceptable; recall needs improvement.

---

## 3. Multi-Party Additional Insured Scenarios (MEDIUM RISK)

**What fails:** When a FNOL involves a named insured, additional insured, and certificate holder all potentially having claims under different coverage grants, the model frequently assigns the wrong coverage line or misroutes.

**Why it happens:** Three-party insurance relationships are structurally complex. The model sees "additional insured" and routes to GL when the actual loss may be a CP matter for the property owner. The multi-line classification task is genuinely ambiguous even for experienced examiners.

**Mitigation:** More multi-line training examples. Consider adding a specific confidence penalty for Multi-Line classifications — always flag for human review.

---

## 4. Low-Frequency Cause Codes (MEDIUM RISK)

**What fails:** Cause codes with fewer than 10 training examples (e.g., ADV-03 — advertising injury, EQUIP-05 — equipment breakdown) produce near-zero F1. The model defaults to higher-frequency codes (LIA-11, WTR-07) when uncertain.

**Why it happens:** Class imbalance. 550 training records with 12+ cause codes means some codes have very few examples. LoRA training on a classification task with this distribution will overfit to majority classes.

**Mitigation:** (1) Oversample rare cause codes in data generation. (2) Use class-weighted loss during training. (3) For cause code specifically, consider a retrieval-augmented approach rather than pure generation.

---

## 5. OCR-Degraded Fax Narratives (LOW-MEDIUM RISK)

**What fails:** Heavily OCR-degraded narratives (>10% character error rate) produce incorrect classifications. The model handles mild degradation but fails on severe garbling.

**Why it happens:** Training data simulates light OCR errors but not severe degradation. Real fax-to-email pipelines vary widely in quality.

**Mitigation:** (1) Add a pre-processing step: run incoming fax narratives through a spelling correction pass before inference. (2) Flag fax submissions with >5% suspected OCR errors (measured by non-dictionary-word rate) for human review before model inference.

---

## 6. (Bonus) Confident Wrong Answers on C1 Edge Cases

**What fails:** The model occasionally produces high-confidence (>0.80) incorrect routing on C1 claims that have unusual narrative structure (e.g., a terse 2-sentence fax reporting a fatality without explicit large-loss language).

**Why it happens:** The confidence proxy (mean token probability) reflects fluency of the generated JSON, not correctness of the classification. A confidently wrong answer bypasses the human review flag.

**Mitigation:** This is the core argument for proper confidence calibration (Platt scaling). Until calibration is implemented, add a rule: any claim with `severity_tier = C1` always sets `human_review_required = true`, regardless of confidence score. (This is already in the current API code.)
