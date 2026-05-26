"""
api.py — FNOL Triage Inference API

FastAPI service wrapping the fine-tuned Mistral-7B + LoRA adapter.
Exposes a /triage endpoint that accepts raw FNOL narratives and returns
structured triage decisions with confidence scores.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

    # Or for demo (auto-reload):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "../training/adapters/final")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Routing → severity tier reverse map (for confidence inference)
ROUTING_LABELS = ["LLU", "LMU", "STP", "SIU"]
COVERAGE_LABELS = ["Commercial Property", "General Liability", "Commercial Auto", "Multi-Line"]
SEVERITY_LABELS = ["C1", "C2", "C3"]

DOMAIN_SIGNALS = [
    "reservation_of_rights",
    "subrogation_indicator",
    "cat_event",
    "occurrence_claims_made_ambiguity",
    "named_additional_insured_confusion",
]

# Global model state (loaded once on startup)
_model = None
_tokenizer = None
_base_model_only = None  # for split-view comparison


# ── Lifespan (load model on startup) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _base_model_only
    logger.info(f"Loading tokenizer from {ADAPTER_PATH}")
    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info(f"Loading base model {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Keep base model reference for split-view
    _base_model_only = base

    logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}")
    _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    _model.eval()
    logger.info("Model ready.")
    yield
    # Cleanup
    del _model, _tokenizer, _base_model_only


app = FastAPI(
    title="FNOL Triage API",
    description="Commercial lines FNOL triage using fine-tuned Mistral-7B",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class TriageRequest(BaseModel):
    narrative: str
    submission_channel: str = "portal"  # broker_email | portal | fax_to_email
    compare_base_model: bool = False    # if True, also run base model for split-view


class DimensionResult(BaseModel):
    value: str
    confidence: str  # "high" | "medium" | "low" (claims examiner-friendly)
    confidence_score: float  # raw score for logging; not shown in UI


class TriageResponse(BaseModel):
    fnol_id: str
    coverage_line: DimensionResult
    cause_code: DimensionResult
    severity_tier: DimensionResult
    routing: DimensionResult
    domain_signals_detected: list[str]
    human_review_required: bool
    human_review_reasons: list[str]
    evidence_highlights: list[dict]   # [{span, signal, explanation}]
    processing_time_ms: int
    base_model_output: Optional[dict] = None  # populated if compare_base_model=True


# ── Inference helpers ─────────────────────────────────────────────────────────
# Llama 3.1 chat format. Mirrors training/finetune.py:INFERENCE_PROMPT — keep
# them in sync. Ends at the assistant header so the model generates the JSON.
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert commercial lines claims examiner. Triage the FNOL submission and respond with a single valid JSON object — no prose, no markdown fences.<|eot_id|><|start_header_id|>user<|end_header_id|>

Submission channel: {channel}

FNOL narrative:
{narrative}

Respond with a JSON object containing:
- coverage_line: "Commercial Property" | "General Liability" | "Commercial Auto" | "Multi-Line"
- cause_code: ISO cause code (e.g. "WTR-07", "CAT-01", "LIA-11")
- severity_tier: "C1" | "C2" | "C3"
- routing: "LLU" | "LMU" | "STP" | "SIU"
- domain_signals: list of detected signals (reservation_of_rights, subrogation_indicator, cat_event, occurrence_claims_made_ambiguity, named_additional_insured_confusion)<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def run_model_inference(model, tokenizer, narrative: str, channel: str) -> tuple[dict, float]:
    """
    Run inference and return (parsed_result, mean_confidence).
    Confidence is approximated from output token probabilities.
    """
    prompt = PROMPT_TEMPLATE.format(channel=channel, narrative=narrative)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute mean token probability as proxy for confidence
    if outputs.scores:
        token_probs = []
        for score, token_id in zip(outputs.scores, generated_ids):
            prob = torch.softmax(score, dim=-1)[0][token_id].item()
            token_probs.append(prob)
        mean_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.5
    else:
        mean_confidence = 0.5

    # Parse JSON
    result = {}
    try:
        match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"Failed to parse model output: {generated_text[:200]}")

    return result, mean_confidence


def bucket_confidence(score: float) -> str:
    """Convert raw probability score to examiner-friendly label."""
    if score >= 0.80:
        return "high"
    elif score >= CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def extract_evidence_highlights(narrative: str, signals: list[str]) -> list[dict]:
    """
    Find text spans in the narrative that triggered domain signal detection.
    Uses keyword matching as a lightweight span extractor.
    In production, this would use the model's attention weights.
    """
    SIGNAL_PATTERNS = {
        "reservation_of_rights": {
            "patterns": [
                "reservation of rights", "reserve all rights", "without prejudice",
                "reserve our rights", "ror letter", "rights and defenses",
            ],
            "explanation": "Reservation of Rights — insurer may contest coverage. Routes to LMU.",
            "color": "#FF7043",
        },
        "subrogation_indicator": {
            "patterns": [
                "subrogation", "third party", "third-party", "negligence of",
                "recovery potential", "pursue recovery", "liable party",
                "contractor negligence", "defective product",
            ],
            "explanation": "Subrogation indicator — recovery from liable third party possible.",
            "color": "#7B1FA2",
        },
        "cat_event": {
            "patterns": [
                "hurricane", "tornado", "named storm", "cat event", "cat-01",
                "catastrophe", "widespread damage", "state of emergency",
            ],
            "explanation": "CAT event linkage — part of aggregate catastrophe event. Routes to LLU.",
            "color": "#1565C0",
        },
        "occurrence_claims_made_ambiguity": {
            "patterns": [
                "retroactive date", "claims-made", "claims made", "prior acts",
                "occurrence policy", "policy period", "prior policy",
            ],
            "explanation": "Occurrence/claims-made ambiguity — coverage trigger unclear.",
            "color": "#2E7D32",
        },
        "named_additional_insured_confusion": {
            "patterns": [
                "additional insured", "certificate holder", "cert holder",
                "named insured", "additional named", "ai endorsement",
            ],
            "explanation": "Named/additional insured — coverage scope unclear for each party.",
            "color": "#E65100",
        },
    }

    highlights = []
    narrative_lower = narrative.lower()

    for signal in signals:
        if signal not in SIGNAL_PATTERNS:
            continue
        info = SIGNAL_PATTERNS[signal]
        for pattern in info["patterns"]:
            idx = narrative_lower.find(pattern.lower())
            if idx != -1:
                span = narrative[idx:idx + len(pattern)]
                highlights.append({
                    "span": span,
                    "start": idx,
                    "end": idx + len(pattern),
                    "signal": signal,
                    "explanation": info["explanation"],
                    "color": info["color"],
                })
                break  # one match per signal is enough for UI highlighting

    return highlights


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.narrative.strip():
        raise HTTPException(status_code=400, detail="Narrative cannot be empty")

    start_ms = time.time() * 1000

    # Run fine-tuned model
    result, mean_confidence = run_model_inference(
        _model, _tokenizer, request.narrative, request.submission_channel
    )

    # Build per-dimension results
    # Use mean_confidence as a proxy for all dimensions.
    # A production system would compute per-dimension confidence separately.
    def make_dim(key: str, fallback: str) -> DimensionResult:
        val = result.get(key, fallback)
        return DimensionResult(
            value=val,
            confidence=bucket_confidence(mean_confidence),
            confidence_score=round(mean_confidence, 3),
        )

    coverage = make_dim("coverage_line", "UNKNOWN")
    cause = make_dim("cause_code", "UNKNOWN")
    severity = make_dim("severity_tier", "C3")
    routing = make_dim("routing", "STP")

    # Domain signals
    detected_signals = [s for s in result.get("domain_signals", []) if s in DOMAIN_SIGNALS]

    # Human review flag
    review_reasons = []
    if mean_confidence < CONFIDENCE_THRESHOLD:
        review_reasons.append(f"Low model confidence ({mean_confidence:.0%})")
    if severity.value == "C1":
        review_reasons.append("C1 severity — large loss requires senior examiner")
    if "reservation_of_rights" in detected_signals:
        review_reasons.append("Reservation of rights detected — coverage dispute possible")
    if routing.value == "SIU":
        review_reasons.append("Fraud indicators detected — SIU referral requires verification")

    human_review_required = len(review_reasons) > 0

    # Evidence highlights
    highlights = extract_evidence_highlights(request.narrative, detected_signals)

    # Base model comparison (optional)
    base_output = None
    if request.compare_base_model and _base_model_only is not None:
        # Temporarily disable adapter
        _model.disable_adapter_layers()
        base_result, base_conf = run_model_inference(
            _model, _tokenizer, request.narrative, request.submission_channel
        )
        _model.enable_adapter_layers()
        base_output = {**base_result, "confidence": round(base_conf, 3)}

    elapsed_ms = int(time.time() * 1000 - start_ms)

    import uuid
    return TriageResponse(
        fnol_id=f"FNOL-{uuid.uuid4().hex[:8].upper()}",
        coverage_line=coverage,
        cause_code=cause,
        severity_tier=severity,
        routing=routing,
        domain_signals_detected=detected_signals,
        human_review_required=human_review_required,
        human_review_reasons=review_reasons,
        evidence_highlights=highlights,
        processing_time_ms=elapsed_ms,
        base_model_output=base_output,
    )


@app.get("/sample-fnols")
async def sample_fnols():
    """Return pre-loaded test cases for the demo UI."""
    return SAMPLE_FNOLS


# ── Pre-loaded demo test cases ─────────────────────────────────────────────────
SAMPLE_FNOLS = [
    {
        "id": "reservation_of_rights",
        "label": "Reservation of Rights (C2)",
        "channel": "broker_email",
        "narrative": (
            "RE: Claim # TBD — Policy CPP-4421-GL — ABC Construction Corp\n\n"
            "Writing to advise of a general liability claim submitted by claimant Rosa Delgado "
            "alleging slip and fall on insured's premises 11/14. Claimant is represented by "
            "Harmon & Associates. Demand letter received 12/02 for $285,000 — medical specials "
            "$42,800, lost wages $18,200, general damages balance.\n\n"
            "Coverage note: Policy is claims-made. Retroactive date is 06/01. Incident date "
            "predates retroactive date by 19 days. We are issuing a reservation of rights letter "
            "to the insured reserving all rights and defenses under the policy pending coverage "
            "determination. Please assign to LMU for coverage counsel review.\n\n"
            "ISO form CG 00 02 04 13. SIR applies at $25,000."
        ),
    },
    {
        "id": "subrogation",
        "label": "Subrogation Indicator (C2)",
        "channel": "portal",
        "narrative": (
            "Insured: Greenfield Industrial Park LLC. Date of loss: 10/28. "
            "Loss type: Water damage — burst pipe, third floor mechanical room.\n\n"
            "Estimated damage: $312,000 to equipment and inventory. Business interruption "
            "exposure approximately $85,000 (14-day shutdown).\n\n"
            "Root cause: Plumbing contractor Apex Mechanical performed system work on 10/25 "
            "under maintenance contract. Post-incident inspection confirmed improper fitting "
            "installation by contractor tech. Insured has documentation of contractor visit "
            "and intends to pursue recovery. Contractor carries GL policy — broker has cert "
            "on file. Strong subrogation potential. Request subrogation flag and preserve "
            "all contractor records."
        ),
    },
    {
        "id": "cat_event",
        "label": "CAT Event Cluster (C1)",
        "channel": "broker_email",
        "narrative": (
            "URGENT — CAT Event FNOL\n"
            "Hurricane Ida — Landfall 08/29 — Southeast portfolio\n\n"
            "Submitting on behalf of Bayou Coast Seafood Processing Inc., policy CP 8847-002. "
            "Total structural loss — processing facility roof collapsed under wind load, "
            "Category 4. Building TIV $4.2M. Equipment TIV $1.8M. BI exposure significant — "
            "peak processing season, estimate 60-day shutdown minimum.\n\n"
            "Site access restricted — county has not cleared entry. We have engaged "
            "Haag Engineering for independent cause-and-origin. Reserve recommendation: "
            "$3.5M building + $1.4M equipment + $800K BI minimum.\n\n"
            "This is the fourth CAT submission from our Gulf Coast book in the past 48 hours "
            "related to Ida. Please flag for CAT unit coordination."
        ),
    },
    {
        "id": "multi_line",
        "label": "Multi-Line Ambiguity",
        "channel": "fax_to_email",
        "narrative": (
            "FAX TRANSMISSION 11/07 14:32\n"
            "TO: CLAIMS INTAKE\nFROM: PREMIER RISK MGMT\n\n"
            "RE: Meridian Contrac1ing Group — Vehicle Accident / Premises Liability\n\n"
            "On 11/05 a vehicle operated by Meridian employee John Vargas struck a pedestrian "
            "in the parking lot of client Westbrook Corp. Cl aimant sustained fractured hip "
            "— hospitalized. Vargas was on Westbrook premises for scheduled service delivery.\n\n"
            "Meridian carries Commercial Auto (CA 00 01) and GL (CG 00 01). Westbrook is "
            "additional insured under Meridian GL cert. Potential exposure under both lines. "
            "Westbrook may also have own GL exposure as premises owner.\n\n"
            "Please advise on coverage coordination. Claimant unrepresented at this time."
        ),
    },
    {
        "id": "clean_c3",
        "label": "Standard C3 Claim",
        "channel": "portal",
        "narrative": (
            "Insured: Maple Street Bakery LLC\n"
            "Date of loss: 11/12\n"
            "Coverage: Commercial Property — CPP-3301\n\n"
            "Loss description: Vandalism to front display window overnight. "
            "Glass broken, no entry gained, no merchandise taken. "
            "Estimate from Citywide Glass: $2,400 replacement. "
            "Police report filed, case number MPD-2024-88442.\n\n"
            "Deductible: $1,000. Net claim: approximately $1,400.\n"
            "Photos attached. No injuries. Business open throughout."
        ),
    },
    {
        "id": "claims_made",
        "label": "Occurrence vs Claims-Made Ambiguity",
        "channel": "broker_email",
        "narrative": (
            "RE: Late-reported GL claim — policy year question\n\n"
            "Insured: Hartwell Professional Services. Claimant alleging professional "
            "negligence in consulting services rendered during Q3 of prior year. "
            "Claim letter received 11/01 this year.\n\n"
            "Current policy: Claims-made, effective 01/01 this year, retroactive date 01/01 "
            "three years prior. Prior carrier: Meridian Specialty, claims-made, expired 12/31 "
            "last year.\n\n"
            "Question: Services rendered during prior policy period. Claim made during current "
            "policy period. Prior carrier is disputing trigger arguing claim should fall under "
            "current carrier. Need coverage counsel opinion on which policy responds. "
            "No settlement authority extended until coverage determination complete."
        ),
    },
]
