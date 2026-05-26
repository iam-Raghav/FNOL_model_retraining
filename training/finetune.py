"""
FNOL Triage Fine-Tuning Pipeline
QLoRA fine-tuning of Llama-3.1-8B-Instruct on synthetic FNOL dataset.

Model choice: meta-llama/Meta-Llama-3.1-8B-Instruct
  - 128K-vocab tokenizer handles insurance jargon (ISO form codes, policy numbers,
    broker shorthand) more efficiently than Mistral's 32K vocab
  - Stronger instruction following and structured JSON adherence than Mistral-7B
    (per Meta's release evals and independent leaderboards)
  - 8K context covers all observed FNOL narratives (max in dataset: ~3K tokens)
  - Fits in 16GB VRAM with 4-bit QLoRA (~12.5GB weights + activations on T4)
  - Llama Community License: free commercial use under 700M MAU — non-binding for
    a mid-market insurer. Requires HF gated-model access acceptance + HF_TOKEN.

Fine-tuning method: QLoRA (4-bit NF4 quantization + LoRA adapters)
  - Full fine-tuning would need ~64GB VRAM (8B × 4 bytes × optimizer overhead) —
    infeasible on T4. LoRA adapters trained at 4-bit base reach ~95-98% of full FT
    quality on classification-style tasks while fitting comfortably on a T4.
  - Adapter weights are stored separately from base model — allows rollback
    and side-by-side base vs. fine-tuned comparison in demo.

Usage:
    huggingface-cli login          # one-time, accept Llama 3.1 license on HF
    python finetune.py --data ../data/generated_dataset.jsonl
    python finetune.py --data ../data/generated_dataset.jsonl --resume-from-checkpoint adapters/checkpoint-50

Environment: Kaggle Notebook (T4 16GB) or Google Colab (T4 16GB)
Estimated training time: 2.5–3.5 hours for 3 epochs on ~400 training records
(Llama-3.1-8B is ~15% slower per step than Mistral-7B at this batch size)
"""

import os
import re
import sys
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# Import cost matrices from sibling evaluation/ package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from cost_matrix import routing_cost, severity_cost  # noqa: E402

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_OUTPUT_DIR = Path("adapters")
ADAPTER_OUTPUT_DIR.mkdir(exist_ok=True)

# ── Prompt template ───────────────────────────────────────────────────────────
# Llama 3.1 chat format: <|begin_of_text|> + per-turn header blocks + <|eot_id|>.
# Hardcoded here (rather than using tokenizer.apply_chat_template) so the exact
# token sequence is identical across training, evaluation, and inference modules.
# INFERENCE_PROMPT ends at the assistant header so the model generates the JSON;
# PROMPT_TEMPLATE appends the labeled JSON + <|eot_id|> for SFT training.
INFERENCE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

PROMPT_TEMPLATE = INFERENCE_PROMPT + "{output}<|eot_id|>"

OUTPUT_TEMPLATE = """{{"coverage_line": "{coverage_line}", "cause_code": "{cause_code}", "severity_tier": "{severity_tier}", "routing": "{routing}", "domain_signals": {domain_signals}}}"""


def format_record(record: dict) -> str:
    """Format a dataset record into the instruction-tuning prompt."""
    output = OUTPUT_TEMPLATE.format(
        coverage_line=record["coverage_line"],
        cause_code=record["cause_code"],
        severity_tier=record["severity_tier"],
        routing=record["routing"],
        domain_signals=json.dumps(record["domain_signals"]),
    )
    return PROMPT_TEMPLATE.format(
        channel=record["submission_channel"],
        narrative=record["raw_narrative"],
        output=output,
    )


def load_dataset_from_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset and validate required fields."""
    records = []
    required_fields = {"fnol_id", "submission_channel", "raw_narrative",
                       "coverage_line", "cause_code", "severity_tier", "routing", "domain_signals"}
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line.strip())
            missing = required_fields - set(rec.keys())
            if missing:
                raise ValueError(f"Record {i} missing fields: {missing}")
            records.append(rec)
    return records


def stratified_split(records: list[dict], val_ratio=0.15, test_ratio=0.15):
    """
    Stratified split on severity_tier to prevent C1 from disappearing
    in small splits. C1 is ~5% of data — random splits would give zero
    C1 examples in val/test ~30% of the time on this dataset size.
    """
    by_severity = {"C1": [], "C2": [], "C3": []}
    for r in records:
        by_severity[r["severity_tier"]].append(r)

    train, val, test = [], [], []
    for sev, recs in by_severity.items():
        random.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.extend(recs[:n_test])
        val.extend(recs[n_test:n_test + n_val])
        train.extend(recs[n_test + n_val:])

    print(f"\nDataset split (stratified on severity_tier):")
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        sev_dist = Counter(r["severity_tier"] for r in split)
        print(f"  {split_name}: {len(split)} records | {dict(sev_dist)}")

    return train, val, test


def build_qlora_model(base_model: str):
    """
    Load Mistral-7B in 4-bit NF4 quantization and attach LoRA adapters.

    QLoRA config choices:
    - 4-bit NF4 quantization: better than INT4 for weights that are
      normally distributed (which transformer weights are)
    - double_quant=True: quantize the quantization constants too — saves
      ~0.5GB extra VRAM at minimal quality cost
    - compute_dtype=bfloat16: faster than float16 on modern GPUs, more
      numerically stable for the forward pass
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NF4 preferred over INT4 for LLM weights
        bnb_4bit_double_quant=True,        # saves ~0.5GB extra VRAM
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 > float16 for stability
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama 3.1 ships without a default pad token
    tokenizer.padding_side = "right"           # right padding for causal LM training

    # LoRA config choices:
    # r=16: rank 16 balances adaptation capacity vs. parameter count.
    #   r=8 is common but we have 5 domain signals to learn; r=16 gives headroom.
    # lora_alpha=32: alpha/r = 2 is the standard scaling factor for stable training
    # target_modules: q_proj + v_proj is the canonical LoRA target for attention.
    #   Adding k_proj and o_proj increases trainable params ~2x but helps on
    #   structured output tasks like ours.
    # dropout=0.05: light regularization; our dataset is small (385 train records)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


class BusinessMetricsCallback(TrainerCallback):
    """
    Runs at the end of each epoch — generates predictions on the val split,
    parses the JSON outputs, and computes business-meaningful metrics
    (routing/severity/coverage weighted F1 + routing/severity asymmetric cost).

    Logged to `state.log_history` so plot_learning_curves can pick them up.
    Early stopping is unaffected — it remains driven by `eval_loss` from the
    step-based trainer eval.
    """

    # Keys with the `eval_biz_` prefix to avoid clashing with the trainer's
    # built-in `eval_loss` log entries.
    METRIC_KEYS = [
        "eval_biz_routing_f1",
        "eval_biz_severity_f1",
        "eval_biz_coverage_f1",
        "eval_biz_routing_cost",
        "eval_biz_severity_cost",
    ]

    def __init__(self, val_records, tokenizer, max_new_tokens: int = 200):
        self.val_records = val_records
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _predict_one(self, model, record: dict) -> dict:
        prompt = INFERENCE_PROMPT.format(
            channel=record["submission_channel"],
            narrative=record["raw_narrative"],
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        match = re.search(r'\{.*?\}', gen, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        was_training = model.training
        model.eval()
        print(f"\n[epoch {state.epoch:.1f}] running per-epoch business eval on "
              f"{len(self.val_records)} val records...")
        preds = [self._predict_one(model, r) for r in self.val_records]

        def f1(dim):
            y_true = [r[dim] for r in self.val_records]
            y_pred = [p.get(dim, "UNKNOWN") for p in preds]
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        metrics = {
            "epoch": state.epoch,
            "step": state.global_step,
            "eval_biz_routing_f1": f1("routing"),
            "eval_biz_severity_f1": f1("severity_tier"),
            "eval_biz_coverage_f1": f1("coverage_line"),
            "eval_biz_routing_cost": float(routing_cost(
                [r["routing"] for r in self.val_records],
                [p.get("routing", "") for p in preds],
            )),
            "eval_biz_severity_cost": float(severity_cost(
                [r["severity_tier"] for r in self.val_records],
                [p.get("severity_tier", "") for p in preds],
            )),
        }
        state.log_history.append(metrics)
        print(f"  routing_f1={metrics['eval_biz_routing_f1']:.3f}  "
              f"severity_f1={metrics['eval_biz_severity_f1']:.3f}  "
              f"coverage_f1={metrics['eval_biz_coverage_f1']:.3f}  "
              f"routing_cost={metrics['eval_biz_routing_cost']:.2f}  "
              f"severity_cost={metrics['eval_biz_severity_cost']:.2f}")

        if was_training:
            model.train()


def plot_learning_curves(log_history: list[dict], output_dir: Path):
    """Produce two plots: (1) train/val loss curves, (2) per-epoch business metrics."""
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            val_steps.append(entry["step"])
            val_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label="Training Loss", color="#2196F3", alpha=0.8)
    plt.plot(val_steps, val_loss, label="Validation Loss", color="#F44336", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("FNOL Triage Fine-Tuning — Learning Curves (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / "learning_curves.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {loss_path}")

    # ── Per-epoch business metrics curve ─────────────────────────────────────
    biz_entries = [e for e in log_history if "eval_biz_routing_f1" in e]
    if not biz_entries:
        print("No business metrics logged — skipping business-metrics plot.")
        return

    epochs = [e["epoch"] for e in biz_entries]
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, [e["eval_biz_routing_f1"]  for e in biz_entries], "-o", label="Routing F1",  color="#2e7d32")
    ax1.plot(epochs, [e["eval_biz_severity_f1"] for e in biz_entries], "-s", label="Severity F1", color="#1565c0")
    ax1.plot(epochs, [e["eval_biz_coverage_f1"] for e in biz_entries], "-^", label="Coverage F1", color="#6a1b9a")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted F1")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [e["eval_biz_routing_cost"]  for e in biz_entries], "--o", label="Routing Cost",  color="#c62828", alpha=0.7)
    ax2.plot(epochs, [e["eval_biz_severity_cost"] for e in biz_entries], "--s", label="Severity Cost", color="#ef6c00", alpha=0.7)
    ax2.set_ylabel("Asymmetric Cost (lower is better)")
    ax2.legend(loc="upper right")

    plt.title("FNOL Triage Fine-Tuning — Per-Epoch Business Metrics")
    plt.tight_layout()
    biz_path = output_dir / "learning_curves_business.png"
    plt.savefig(biz_path, dpi=150)
    plt.close()
    print(f"Business metrics curve saved: {biz_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/generated_dataset.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)  # T4 16GB can handle bs=4
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}")
    records = load_dataset_from_jsonl(args.data)
    print(f"Loaded {len(records)} records")

    train_records, val_records, test_records = stratified_split(records)

    # Save test split separately — evaluation script needs it
    test_path = Path(args.data).parent / "test_split.jsonl"
    with open(test_path, "w") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")
    print(f"Test split saved to {test_path}")

    # Format as instruction-tuning prompts
    train_texts = [format_record(r) for r in train_records]
    val_texts = [format_record(r) for r in val_records]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    print(f"\nLoading base model: {BASE_MODEL}")
    model, tokenizer = build_qlora_model(BASE_MODEL)

    # Training arguments:
    # lr=2e-4: standard for LoRA fine-tuning on instruction datasets
    # warmup_ratio=0.05: 5% warmup prevents large early gradient steps on small dataset
    # weight_decay=0.01: light L2 regularization
    # gradient_accumulation_steps=4: effective batch size = 4*4 = 16, matches common LoRA recipes
    # fp16=True: mixed precision for speed (bfloat16 not universally supported on T4)
    # eval_strategy="steps", eval_steps=25: frequent eval given small dataset
    # save_strategy="steps": save checkpoints for early stopping
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,      # effective batch 16 — standard for LoRA
        learning_rate=2e-4,                 # LoRA standard; full fine-tune uses 1e-5
        warmup_ratio=0.05,                  # prevents gradient spike on small dataset
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        load_best_model_at_end=True,        # needed for early stopping
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",                   # disable W&B; use local logs only
        seed=SEED,
        dataloader_num_workers=0,           # Kaggle environment: 0 prevents fork issues
        remove_unused_columns=False,
    )

    # Early stopping: patience=3 means stop if val loss doesn't improve for 3 eval steps.
    # With eval_steps=25 and ~96 train steps/epoch, patience=3 ~= 75 steps ~= 0.75 epochs.
    # Conservative: we don't want to overtrain on 385 examples.
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # Per-epoch business-metrics eval — generation-based F1 + asymmetric cost
    # on val_records. Doesn't influence training (no early-stopping hook); purely
    # for the second learning-curve plot. ~3-4 min added per epoch on a T4.
    business_eval = BusinessMetricsCallback(val_records=val_records, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        callbacks=[early_stopping, business_eval],
    )

    print("\nStarting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    print("\nSaving adapter weights...")
    # Save ONLY adapter weights — base model is not modified and should
    # stay separate so we can load base vs. fine-tuned side by side in demo
    final_adapter_path = ADAPTER_OUTPUT_DIR / "final"
    trainer.model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    print(f"Adapter weights saved to {final_adapter_path}")

    # Learning curves
    plot_learning_curves(trainer.state.log_history, ADAPTER_OUTPUT_DIR)

    print("\nTraining complete.")
    print(f"Best validation loss: {trainer.state.best_metric:.4f}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
