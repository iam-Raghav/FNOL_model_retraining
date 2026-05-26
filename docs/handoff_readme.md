# FNOL Triage Fine-Tuning System — Handoff README

**Author:** FDE, Intellect Design Arena  
**Tested on:** Kaggle Notebook (T4 16GB), Ubuntu 22.04, Python 3.10

---

## What This System Does

Fine-tuned Llama-3.1-8B-Instruct that triages commercial lines FNOL (First Notice of Loss) submissions across four dimensions:

- **Coverage line** — Commercial Property / GL / Auto / Multi-Line
- **Cause code** — ISO-based taxonomy (WTR-07, CAT-01, LIA-11, etc.)
- **Severity tier** — C1 (critical), C2 (elevated), C3 (standard)
- **Routing decision** — LLU / LMU / STP / SIU

Plus detection of five domain-specific signals: reservation of rights, subrogation indicators, CAT event linkage, occurrence/claims-made ambiguity, named/additional insured confusion.

---

## Prerequisites

- Python 3.10+
- CUDA GPU with 16GB VRAM for training (T4 minimum)
- Anthropic API key for dataset generation (set `ANTHROPIC_API_KEY` env var)
- **Hugging Face account with Llama 3.1 license accepted** — the base model is gated. See Step 0 below.
- ~17GB disk space (Llama-3.1-8B base weights ~16GB on first download, ~200MB for adapters)

---

## Step 0 — Environment Setup

```bash
git clone <your-repo-url> fnol-triage-finetune
cd fnol-triage-finetune

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
```

**Accept the Llama 3.1 license and authenticate Hugging Face (one-time):**

1. Sign in / create an account at https://huggingface.co
2. Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct and click **"Agree and access repository"**. Meta usually approves automatically within a few minutes; you'll receive an email when access is granted.
3. Create a read-scope token at https://huggingface.co/settings/tokens (Type: **Read**)
4. Log in from the machine that will run training/inference:

```bash
huggingface-cli login
# Paste the token when prompted. Stored at ~/.cache/huggingface/token
```

Verify access:
```bash
huggingface-cli whoami                                # should print your HF username
python -c "from huggingface_hub import HfApi; HfApi().model_info('meta-llama/Meta-Llama-3.1-8B-Instruct')"
# Should complete without 401/403. A GatedRepoError means license is not yet approved.
```

If you prefer a non-interactive setup (CI, Kaggle secret, Colab secret), export `HF_TOKEN` instead:
```bash
export HF_TOKEN="hf_..."
```
The `transformers` / `huggingface_hub` libraries pick this up automatically.

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Step 1 — Generate Synthetic Dataset

```bash
cd data
python generator.py --count 550 --output generated_dataset.jsonl
```

Expected output: `data/generated_dataset.jsonl` (~550 records)
Time: ~12 minutes (API rate limited to ~3 req/sec)
Cost: ~$1–2 in Anthropic API credits

Verify distribution:
```bash
python -c "
import json
from collections import Counter
records = [json.loads(l) for l in open('generated_dataset.jsonl')]
print('Total:', len(records))
print('Severity:', Counter(r['severity_tier'] for r in records))
print('Routing:', Counter(r['routing'] for r in records))
"
```

---

## Step 2 — Fine-Tune

**On Kaggle (recommended):**
1. Create new Kaggle Notebook
2. Upload `training/finetune.py` and `data/generated_dataset.jsonl`
3. Enable GPU accelerator (T4 or P100)
4. Run:

```python
# In Kaggle notebook cell:
!pip install -r requirements.txt
!python finetune.py --data /kaggle/input/fnol-data/generated_dataset.jsonl
```

**Locally (if you have 16GB VRAM):**
```bash
cd training
python finetune.py --data ../data/generated_dataset.jsonl
```

Expected output:
- `training/adapters/final/` — LoRA adapter weights (~200MB)
- `training/adapters/learning_curves.png` — train/val loss plot
- `data/test_split.jsonl` — held-out test set (auto-saved during training)

Training time: ~2.5–3.5 hours on T4 (Llama-3.1-8B is ~15% slower per step than Mistral-7B at the same batch size; the per-epoch business-metrics eval adds ~3-4 min/epoch on top)

---

## Step 3 — Evaluate

```bash
cd evaluation
python evaluate.py \
  --test-data ../data/test_split.jsonl \
  --adapter-path ../training/adapters/final \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output evaluation_report.html
```

Expected output:
- `evaluation/evaluation_report.html` — full report
- `evaluation/charts/` — confusion matrix heatmaps
- `evaluation/predictions_cache.jsonl` — cached predictions (reuse with `--use-cached-predictions`)

---

## Step 4 — Run the Demo

Terminal 1 — start inference API:
```bash
cd inference
export BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export ADAPTER_PATH="../training/adapters/final"
uvicorn api:app --host 0.0.0.0 --port 8000
```

Wait for: `Model ready.` in logs.

Terminal 2 — start demo UI:
```bash
cd inference
streamlit run demo_app.py
```

Open: `http://localhost:8501`

---

## Step 5 — Run CAT Simulation

Requires inference API running (Step 4).

```bash
cd simulation
python cat_event_sim.py --api-url http://localhost:8000 --output sim_results.jsonl
```

Expected output:
- `simulation/sim_results.jsonl` — per-submission results
- `simulation/cat_incident_report.md` — auto-generated incident report

---

## One-Command Run (after environment setup)

```bash
./run_all.sh
```

This runs Steps 1–5 sequentially. Use only on a machine with a GPU — training will run on CPU otherwise (24+ hours).

---

## Repository Structure

```
fnol-triage-finetune/
├── data/
│   ├── generator.py          # Synthetic FNOL dataset generator
│   ├── generated_dataset.jsonl  # Generated dataset (after Step 1)
│   └── data_card.md          # Dataset documentation
├── training/
│   ├── finetune.py           # QLoRA fine-tuning pipeline
│   ├── training_config.yaml  # Hyperparameter reference
│   └── adapters/             # Saved adapter weights (after Step 2)
├── evaluation/
│   ├── evaluate.py           # Evaluation pipeline
│   ├── cost_matrix.py        # Asymmetric cost scoring
│   └── evaluation_report.html  # Generated report (after Step 3)
├── inference/
│   ├── api.py                # FastAPI inference service
│   └── demo_app.py           # Streamlit Triage Workbench
├── simulation/
│   ├── cat_event_sim.py      # CAT event stress test
│   └── cat_incident_report.md  # Generated incident report
├── docs/
│   ├── adr.md                # Architecture Decision Record
│   ├── known_failures.md     # Honest failure mode list
│   └── handoff_readme.md     # This file
├── requirements.txt
└── run_all.sh
```

---

## Common Issues

**`GatedRepoError` or 401/403 when loading the base model:**
Llama 3.1 is gated. Confirm (a) you've clicked "Agree and access repository" on https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct, (b) Meta has granted access (check email), and (c) `huggingface-cli whoami` resolves on the training machine. On Kaggle, add an `HF_TOKEN` secret and reference it in the notebook before importing transformers.

**`Unrecognized configuration class ... LlamaConfig` or RoPE/long-context warnings:**
Your `transformers` version predates Llama 3.1 support. Pin `transformers>=4.43` (the lockfile uses 4.44.2). Re-install: `pip install -r requirements.txt --upgrade`.

**"CUDA out of memory" during training:**
Reduce `--batch-size` to 2 and increase `gradient_accumulation_steps` to 8 in `finetune.py`. Effective batch size stays the same.

**"Cannot connect to inference API" in demo:**
Check that `uvicorn api:app --port 8000` is running in a separate terminal and `Model ready.` appeared in the logs. Model loading takes 3–5 minutes.

**Generator produces generic narratives:**
The system prompt in `generator.py` is the key lever. Try adding specific policy form references (ISO CG 00 01, CP 10 30) and broker-specific phrases to the prompt.

**Evaluation shows 0% for a cause code:**
That cause code likely has <3 examples in the test set due to class imbalance. Check `data_card.md` for per-class distribution and increase generation count for underrepresented codes.

---

## Extending to Cyber Liability

To add a new line of business:
1. Add `"Cyber Liability"` to `COVERAGE_LINES` in `generator.py`
2. Add cyber cause codes: `CYB-01` (ransomware), `CYB-02` (data breach), `CYB-03` (business interruption)
3. Add domain signals: `ransom_demand_indicator`, `notification_obligation_trigger`
4. Generate 200 cyber FNOL examples with the new templates
5. Retrain LoRA adapters on the expanded dataset (base model unchanged)
6. Update `SIGNAL_PATTERNS` in `api.py` with cyber keyword patterns
7. Update cost matrix in `cost_matrix.py` for cyber routing decisions

Time estimate: 2–3 days including data generation, 2–3 GPU hours for retraining.
