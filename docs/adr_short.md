# ADR — FNOL Triage Fine-Tuning

## Base model: Llama-3.1-8B-Instruct

We ran a 50-record bake-off against Mistral-7B-v0.2, Qwen2.5-7B, Phi-3.5-mini, and Nemo-12B before committing. Llama won on three things that actually mattered for this workload: it tokenized insurance jargon (`CG 00 01`, policy numbers, `UIM`, `SIR`) ~14% denser than Mistral, which buys us context budget; it produced parseable JSON 94% of the time zero-shot vs Mistral's 71%; and its zero-shot routing F1 was 0.62 vs Mistral's 0.48, suggesting better priors going into fine-tuning. Qwen2.5 came close and has the cleaner Apache-2.0 license — if the Meta license ever becomes a procurement headache, that's the swap. Nemo-12B was tempting for its native function-calling but doesn't fit T4 at 4-bit with batch=4. None of these numbers come from a stratified split, so treat the ranking as directional, not gospel.

## QLoRA over LoRA / full FT

Full fine-tune wants ~64GB VRAM. Bf16 LoRA wants ~32GB. QLoRA fits in ~12.5GB on T4 with headroom for batch=4, which is the only configuration we had access to. Literature and our own small runs put the F1 gap vs full FT at 2–5% for classification-style tasks — acceptable given the budget. A separate adapter file was a hard requirement anyway: the split-view demo needs to swap base vs adapter, and rollback in production means swapping a 200MB file instead of redeploying the base. We settled on rank=16, alpha=32. r=8 felt thin given five domain signals interacting with four output dimensions, and the ~50MB extra VRAM for r=16 is essentially free.

## Single-shot JSON output

The four labels co-determine each other — a slip-and-fall fixes coverage, severity, and routing in one move. Splitting into four sequential calls would quadruple latency and break that joint reasoning. JSON parses cleanly; we score malformed output as wrong in `evaluate.py` rather than papering over it. The risk we accepted: in production, schema drift or a fluky generation could push parse rate below the demo's 99%. If that happens, the answer is constrained decoding (outlines/jsonformer), not multi-call.

## Confidence threshold = 0.65 (least confident decision)

Mean token probability below 0.65 flags the claim for human review. I'll be honest — the threshold is gut feel from a few dozen hand-checked outputs, not calibration. Mean token prob is also a proxy: the model can be confidently wrong, especially on terse fax narratives where fluent JSON masks a bad classification. The plan is Platt scaling on 200–300 labeled FNOLs, then anchor the cutoff to a real accuracy-vs-confidence curve. Until then, C1 claims always force human review as a hard rule on top of the threshold.
