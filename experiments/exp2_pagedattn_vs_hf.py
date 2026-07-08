#!/usr/bin/env python3
"""Experiment 2: PagedAttention vs HuggingFace — Single Request Latency

What:  Same model (INT4-AWQ), same prompts — HF generate() vs vLLM engine. Batch=1.
Why:   Isolates the serving engine effect. PagedAttention may be SLOWER at batch=1
       (engine overhead) — that's a valid finding. PA wins at scale, not single-user.

Note:  FP16 was attempted for vLLM but failed:
       Model weights (2.89 GiB) + vLLM overhead (~0.54 GiB) = 3.43 GiB > 3.12 GiB budget.
       Available KV cache: -0.31 GiB. FP16 serving on 4GB VRAM is infeasible.
       INT4 (0.9 GiB) is used for both backends — same model, different serving engine.

⚠️  Run each backend in a SEPARATE process (two commands) to avoid CUDA context residual:
    python3 exp2_pagedattn_vs_hf.py --backend hf   --save
    python3 exp2_pagedattn_vs_hf.py --backend vllm --save  (auto-plots at the end)
"""

import torch, gc, os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

GEN_TOKENS = 256

def run_hf():
    """Run all prompts through HuggingFace generate() with INT4-AWQ model."""
    print("\n── HuggingFace (INT4-AWQ) ───────────────────────────────")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INT4, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_INT4, cache_dir=MODEL_DIR,
        torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    # Warmup — first call has CUDA kernel compilation overhead, discard it
    warm = tokenizer(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=16, min_new_tokens=16, do_sample=False)

    results = []
    for i, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        reset_vram()
        with torch.no_grad():
            with Timer() as t:
                # min_new_tokens forces exactly GEN_TOKENS output — no early EOS skew
                out = model.generate(**inputs, max_new_tokens=GEN_TOKENS,
                                     min_new_tokens=GEN_TOKENS, do_sample=False)

        gen = out.shape[1] - input_len
        results.append({
            "backend": "HF", "prompt": i + 1,
            "tokens": gen, "tok_per_s": round(gen / (t.ms / 1000), 1),
            "latency_ms": round(t.ms, 1), "vram_mb": round(peak_vram_mb(), 0),
        })
        print(f"  [{i+1}] {gen} tok in {t.ms:.0f}ms = {results[-1]['tok_per_s']} tok/s")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results

def run_vllm():
    """Run all prompts through vLLM (PagedAttention) with INT4-AWQ model."""
    print("\n── vLLM / PagedAttention (INT4-AWQ) ─────────────────────")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_INT4,
        download_dir=MODEL_DIR,
        dtype="float16",
        enforce_eager=True,
        # Max free VRAM on this GPU is always ~3.22 GiB (display driver permanently uses 0.78 GiB).
        # 0.78 × 4.0 = 3.12 GiB budget. INT4 model = ~0.9 GiB → KV cache pool ≈ 2.2 GiB.
        # 2.2 GiB supports ~1000 KV blocks (16 tokens each) = plenty for batch experiments.
        gpu_memory_utilization=0.78,
        max_model_len=1024,
    )
    sampling = SamplingParams(max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS, temperature=0)

    # Warmup
    llm.generate([PROMPTS[0]], SamplingParams(max_tokens=16, min_tokens=16, temperature=0))

    results = []
    for i, prompt in enumerate(PROMPTS):
        reset_vram()
        with Timer() as t:
            outputs = llm.generate([prompt], sampling)

        gen = len(outputs[0].outputs[0].token_ids)
        results.append({
            "backend": "vLLM", "prompt": i + 1,
            "tokens": gen, "tok_per_s": round(gen / (t.ms / 1000), 1),
            "latency_ms": round(t.ms, 1),
            # current_vram_mb() reads from CUDA driver — captures vLLM's raw pool allocation
            # peak_vram_mb() would return 0 because vLLM bypasses PyTorch's allocator
            "vram_mb": round(current_vram_mb(), 0),
        })
        print(f"  [{i+1}] {gen} tok in {t.ms:.0f}ms = {results[-1]['tok_per_s']} tok/s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results

def plot_from_csvs():
    """Read the two saved CSVs and generate the comparison chart.
    Run this after both backends have completed and saved their CSVs.
    """
    hf_path   = os.path.join(RESULTS, "exp2_hf_results.csv")
    vllm_path = os.path.join(RESULTS, "exp2_vllm_results.csv")

    if not os.path.exists(hf_path) or not os.path.exists(vllm_path):
        print("ERROR: Run --backend hf and --backend vllm first.")
        return

    hf     = pd.read_csv(hf_path).to_dict("records")
    vllm_r = pd.read_csv(vllm_path).to_dict("records")

    setup_plot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    prompts = [f"P{r['prompt']}" for r in hf]
    x, w = range(len(prompts)), 0.35

    # Throughput chart
    ax1.bar([i - w/2 for i in x], [r["tok_per_s"] for r in hf],   w, label="HF",   color=COLORS["hf"])
    ax1.bar([i + w/2 for i in x], [r["tok_per_s"] for r in vllm_r], w, label="vLLM", color=COLORS["vllm"])
    ax1.set_xlabel("Prompt"); ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_title("Throughput: HF vs vLLM (batch=1)")
    ax1.set_xticks(x); ax1.set_xticklabels(prompts); ax1.legend()

    # Latency chart
    ax2.bar([i - w/2 for i in x], [r["latency_ms"] for r in hf],   w, label="HF",   color=COLORS["hf"])
    ax2.bar([i + w/2 for i in x], [r["latency_ms"] for r in vllm_r], w, label="vLLM", color=COLORS["vllm"])
    ax2.set_xlabel("Prompt"); ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency: HF vs vLLM (batch=1)")
    ax2.set_xticks(x); ax2.set_xticklabels(prompts); ax2.legend()

    save_plot("exp2_comparison.png")

def main():
    p = base_args("Exp2: PagedAttention vs HF")
    p.add_argument("--backend", choices=["hf", "vllm"], default=None,
                   help="Which backend to run. Run each in a separate fresh container.")
    p.add_argument("--plot", action="store_true",
                   help="Generate chart from saved CSVs. Run after both backends finish.")
    args = p.parse_args()

    if args.plot:
        plot_from_csvs()
        return

    if args.backend is None:
        print("Specify --backend hf or --backend vllm  (see docstring for order)")
        return

    if args.backend == "hf":
        results = run_hf()
        print("\n=== Experiment 2: HF Results ===")
        print_table(results)
        if args.save:
            save_csv(results, "exp2_hf_results.csv")

    elif args.backend == "vllm":
        results = run_vllm()
        print("\n=== Experiment 2: vLLM Results ===")
        print_table(results)
        if args.save:
            save_csv(results, "exp2_vllm_results.csv")
            # Auto-plot if HF results already exist — no third command needed
            if os.path.exists(os.path.join(RESULTS, "exp2_hf_results.csv")):
                print("\nHF results found — generating comparison plot...")
                plot_from_csvs()

if __name__ == "__main__":
    main()
