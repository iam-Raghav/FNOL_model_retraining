#!/bin/bash
# run_all.sh — Single entry point to run the full pipeline end-to-end
# Requires: GPU, ANTHROPIC_API_KEY, all dependencies installed

set -e  # exit on any error

echo "================================================"
echo "  FNOL Triage Fine-Tuning Pipeline"
echo "================================================"
echo ""

# Check prerequisites
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected'" 2>/dev/null || {
  echo "WARNING: No GPU detected. Training will run on CPU (very slow)."
  read -p "Continue anyway? [y/N] " confirm
  [ "$confirm" == "y" ] || exit 1
}

echo "[1/5] Generating synthetic FNOL dataset..."
cd data
python generator.py --count 550 --output generated_dataset.jsonl
echo "      Done. Dataset: data/generated_dataset.jsonl"
cd ..

echo ""
echo "[2/5] Fine-tuning Llama-3.1-8B with QLoRA..."
cd training
python finetune.py --data ../data/generated_dataset.jsonl
echo "      Done. Adapters: training/adapters/final/"
cd ..

echo ""
echo "[3/5] Running evaluation on test split..."
cd evaluation
python evaluate.py \
  --test-data ../data/test_split.jsonl \
  --adapter-path ../training/adapters/final \
  --output evaluation_report.html
echo "      Done. Report: evaluation/evaluation_report.html"
cd ..

echo ""
echo "[4/5] Starting inference API (background)..."
cd inference
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "      Waiting for model to load (this takes 3-5 minutes)..."
sleep 180  # wait for model load

# Health check
curl -sf http://localhost:8000/health > /dev/null || {
  echo "ERROR: API failed to start. Check logs."
  kill $API_PID 2>/dev/null
  exit 1
}
echo "      API ready at http://localhost:8000"
cd ..

echo ""
echo "[5/5] Running CAT event simulation..."
cd simulation
python cat_event_sim.py --api-url http://localhost:8000 --output sim_results.jsonl
echo "      Done. Report: simulation/cat_incident_report.md"
cd ..

echo ""
echo "================================================"
echo "  Pipeline complete!"
echo "================================================"
echo ""
echo "  Evaluation report: evaluation/evaluation_report.html"
echo "  CAT incident report: simulation/cat_incident_report.md"
echo "  Learning curves: training/adapters/learning_curves.png"
echo ""
echo "  To start the demo UI:"
echo "    cd inference && streamlit run demo_app.py"
echo ""
echo "  API is still running (PID $API_PID)."
echo "  To stop: kill $API_PID"
