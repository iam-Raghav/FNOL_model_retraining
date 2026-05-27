# FNOL Triage — Llama-3.1-8B + QLoRA

Fine-tuned LLM for triaging commercial-lines **First Notice of Loss** submissions. Takes a raw narrative (broker email, portal submission, or fax-to-email) and returns four structured decisions plus domain-risk flags.

```
broker_email → [model] → {
  coverage_line:  "General Liability",
  cause_code:     "LIA-11",
  severity_tier:  "C2",
  routing:        "LMU",
  domain_signals: ["subrogation_indicator"]
}
```

---

## What's in here

| Component | Purpose |
|---|---|
| `data/generator.py` | Synthetic FNOL generator (Anthropic Claude → 550-record JSONL) |
| `training/finetune.py` | QLoRA fine-tune on Llama-3.1-8B-Instruct |
| `evaluation/evaluate.py` | Per-dimension F1, asymmetric cost scoring, confusion matrices, HTML report |
| `evaluation/cost_matrix.py` | Business-weighted misclassification costs (C1→STP penalized 100×) |
| `inference/api.py` | FastAPI service, loads base + adapter in 4-bit |
| `inference/demo_app.py` | Streamlit three-panel triage workbench |
| `simulation/cat_event_sim.py` | CAT-burst stress test against the live API |
| `adapters/` | Trained LoRA weights (checkpoint-25/50/72 + `final`) |
| `docs/` | ADR, data card, known failure modes, handoff README |

---

## Outputs

**Four classification dimensions:**
- `coverage_line` — Commercial Property / GL / Auto / Multi-Line
- `cause_code` — ISO-style taxonomy (`WTR-07`, `CAT-01`, `LIA-11`, `AUTO-04`, …)
- `severity_tier` — C1 (critical) / C2 (elevated) / C3 (standard)
- `routing` — LLU (large-loss) / LMU (litigation/managed) / STP (straight-through) / SIU (fraud)

**Five domain signals** flagged when present in the narrative:
`reservation_of_rights`, `subrogation_indicator`, `cat_event`, `occurrence_claims_made_ambiguity`, `named_additional_insured_confusion`.

---

## Results (latest run, 82-record held-out test split)

| Dimension | weighted F1 | macro F1 |
|---|---|---|
| routing | 0.918 | 0.865 |
| severity_tier | 0.902 | 0.852 |
| cause_code | 0.866 | 0.866 |
| coverage_line | 0.845 | 0.742 |

**Cost-weighted scores** (lower = better, 0 = perfect):
- Routing cost: **2.70**
- Severity cost: **1.90**
- C1→STP errors: **0** (the must-never-happen failure)

Full report rendered to `evaluation/evaluation_report.html`.

---

## Pipeline

```
        ┌───────────────────┐
        │ data/generator.py │ ── 550-record synthetic FNOLs (Claude API)
        └─────────┬─────────┘
                  ▼
        ┌────────────────────┐
        │ training/finetune  │ ── Llama-3.1-8B + QLoRA r=16
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │  adapters/final    │ ── ~200 MB LoRA weights
        └─────┬────────┬─────┘
              ▼        ▼
     evaluate.py   inference/api.py ── FastAPI :8000
                       │
                       ▼
                  demo_app.py ── Streamlit :8501
```

---

## Quickstart

### 0. Prerequisites
- Python 3.10+ (3.12 tested)
- CUDA GPU, 16 GB VRAM minimum for training (T4 fits in 4-bit). CPU works for client-side UI only.
- Hugging Face account with **Llama-3.1 license accepted** — the base model is gated.
- `ANTHROPIC_API_KEY` only if you plan to regenerate the dataset.

### 1. Install

```bash
git clone <repo-url> fnol-triage
cd fnol-triage
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv) (recommended on Windows):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

### 2. Authenticate Hugging Face

```bash
huggingface-cli login        # or: export HF_TOKEN=hf_xxx
```

The base model `meta-llama/Meta-Llama-3.1-8B-Instruct` is gated — request access on the model page first.

### 3. Run end-to-end

```bash
./run_all.sh                 # generate → train → eval → serve → simulate
```

Or step-by-step:

```bash
# Generate dataset (~12 min, ~$1-2 in Anthropic credits)
cd data && python generator.py --count 550 --output generated_dataset.jsonl

# Fine-tune (T4: ~3 hrs)
cd ../training && python finetune.py --data ../data/generated_dataset.jsonl

# Evaluate
cd ../evaluation && python evaluate.py \
  --test-data ../data/test_split.jsonl \
  --adapter-path ../adapters/final \
  --output evaluation_report.html

# Serve
cd ../inference
ADAPTER_PATH=../adapters/final uvicorn api:app --port 8000

# UI (separate terminal)
streamlit run demo_app.py    # honors API_URL env var
```

### 4. Use cached predictions instead of re-running inference

```bash
python evaluate.py \
  --test-data ../data/test_split.jsonl \
  --use-cached-predictions predictions_cache.jsonl \
  --output evaluation_report.html
```

No GPU needed for this path — useful for iterating on the report layout.

---

## Running on Kaggle, viewing UI locally

The model needs a GPU; the Streamlit UI doesn't. Run the API inside a Kaggle notebook and tunnel it out:

```python
# Kaggle cell
import subprocess, os, time
os.environ["ADAPTER_PATH"] = "/kaggle/working/adapters/final"
os.environ["HF_TOKEN"] = "hf_..."
subprocess.Popen(["uvicorn", "api:app", "--port", "8000"])
# Wait for "Application startup complete"
subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"])
```

Then locally:

```bash
export API_URL="https://<your-cloudflared-url>.trycloudflare.com"
streamlit run inference/demo_app.py
```

---

## Architecture notes

Four key decisions live in [docs/adr_short.md](docs/adr_short.md):

1. **Llama-3.1-8B over Mistral-7B.** Beat Mistral on insurance-jargon tokenization (~14% denser), zero-shot JSON adherence (94% vs 71%), and zero-shot routing F1 (0.62 vs 0.48).
2. **QLoRA over full FT or bf16 LoRA.** ~12.5 GB peak on T4 vs 64 GB for full FT. 2–5% F1 gap on classification is acceptable. r=16, alpha=32.
3. **Single-shot JSON output.** All four labels co-determine each other — splitting into four calls quadruples latency and loses joint reasoning.
4. **Cost-weighted evaluation.** Standard F1 hides the catastrophic failure (C1 → STP). Asymmetric cost matrix in `cost_matrix.py` penalizes this 100× more than C3 → LLU.

---

## Known failure modes

Documented honestly in [docs/known_failures.md](docs/known_failures.md). The big ones:

- **Novel CAT names.** Trained only on Hurricane Ida; unseen storms degrade.
- **Occurrence vs claims-made trigger.** Recall is weak on retroactive-date ambiguity.
- **Rare cause codes.** Codes with <10 training examples (ADV-03, EQUIP-05) collapse toward zero F1.
- **Confidence proxy is uncalibrated.** Mean token probability ≠ correctness. C1 claims are always force-flagged for review as a safety net until Platt scaling is in.

---

## Repository layout

```
.
├── data/                   generator + dataset card + splits
├── training/               QLoRA pipeline + hyperparam config
├── adapters/               trained LoRA weights (checkpoints + final)
├── evaluation/             metrics + cost matrix + HTML report
├── inference/              FastAPI service + Streamlit UI
├── simulation/             CAT-burst stress test
├── docs/                   ADR, data card, known failures, handoff guide
├── requirements.txt
├── run_all.sh
└── README.md
```

---

## License & data

Code: Public use
Dataset: synthetic, no real PII. The pipeline assumes a Presidio redaction stage in front of any real FNOL ingest — see [docs/data_card.md](data/data_card.md) for the production-PII architecture note.
Base model: Meta Llama 3.1 Community License (accept on the HF model page before use).
