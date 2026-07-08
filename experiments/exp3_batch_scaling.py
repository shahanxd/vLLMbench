#!/usr/bin/env python3
"""Experiment 3: Batch Scaling — Where PagedAttention Actually Wins

What:  Increase concurrent requests (1→2→4→8→16). Measure throughput.
Why:   PagedAttention eliminates KV cache fragmentation → fits more requests in VRAM.
       HF will OOM early on 4GB. That OOM point IS the finding.
Input: Same prompt duplicated N times per batch size.
Output: Total throughput (tok/s), peak VRAM at each batch size, for both backends.

Run:   python3 exp3_batch_scaling.py --save
"""

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

GEN_TOKENS   = 128
BATCH_SIZES  = [1, 2, 4, 8]  # conservative for 4GB — extend if memory allows
TEST_PROMPT  = PROMPTS[0]     # use one prompt, duplicated per batch

def run_hf_batch(batch_size, model, tokenizer):
    """Run batch_size copies of the prompt through HF. Returns result dict or None on OOM."""
    prompts = [TEST_PROMPT] * batch_size
    try:
        # Tokenize as a batch (pads shorter sequences)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        reset_vram()
        with torch.no_grad():
            with Timer() as t:
                # min_new_tokens forces exact length — prevents EOS skewing batch comparison
                out = model.generate(**inputs, max_new_tokens=GEN_TOKENS,
                                     min_new_tokens=GEN_TOKENS, do_sample=False)

        # Total output tokens across all requests in the batch
        input_len = inputs["input_ids"].shape[1]
        total_tokens = sum(out.shape[1] - input_len for _ in range(batch_size))
        return {
            "backend": "HF", "batch": batch_size,
            "total_tokens": total_tokens,
            "tok_per_s": round(total_tokens / (t.ms / 1000), 1),
            "latency_ms": round(t.ms, 1),
            "vram_mb": round(peak_vram_mb(), 0),
        }
    except torch.cuda.OutOfMemoryError:
        # OOM is a finding, not a failure — record it
        torch.cuda.empty_cache()
        print(f"  [HF] batch={batch_size} → OOM ❌")
        return {
            "backend": "HF", "batch": batch_size,
            "total_tokens": 0, "tok_per_s": 0,
            "latency_ms": -1, "vram_mb": -1,
        }

def run_vllm_batch(batch_size, llm):
    """Run batch_size prompts through vLLM. Returns result dict."""
    from vllm import SamplingParams
    prompts = [TEST_PROMPT] * batch_size
    sampling = SamplingParams(max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS, temperature=0)

    reset_vram()
    with Timer() as t:
        outputs = llm.generate(prompts, sampling)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "backend": "vLLM", "batch": batch_size,
        "total_tokens": total_tokens,
        "tok_per_s": round(total_tokens / (t.ms / 1000), 1),
        "latency_ms": round(t.ms, 1),
        # current_vram_mb() reads total GPU usage from CUDA driver, capturing vLLM's
        # subprocess KV cache pool. peak_vram_mb() would return ~8 MB (PyTorch delta only).
        "vram_mb": round(current_vram_mb(), 0),
    }

def plot_scaling(results):
    """Line chart: throughput vs batch size, one line per backend."""
    setup_plot()
    fig, ax = plt.subplots()

    for backend, color in [("HF", COLORS["hf"]), ("vLLM", COLORS["vllm"])]:
        rows = [r for r in results if r["backend"] == backend and r["tok_per_s"] > 0]
        if rows:
            batches = [r["batch"] for r in rows]
            tps = [r["tok_per_s"] for r in rows]
            ax.plot(batches, tps, "o-", color=color, label=backend, linewidth=2, markersize=8)

    # Mark OOM points
    oom = [r for r in results if r["latency_ms"] == -1]
    for r in oom:
        ax.axvline(x=r["batch"], color=COLORS["hf"], linestyle="--", alpha=0.5)
        ax.annotate(f"HF OOM", (r["batch"], ax.get_ylim()[1] * 0.9),
                    fontsize=10, color="red", ha="center")

    ax.set_xlabel("Batch Size (concurrent requests)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Batch Scaling: HF vs vLLM (PagedAttention)")
    ax.legend()
    save_plot("exp3_throughput.png")

def main():
    args = base_args("Exp3: Batch Scaling").parse_args()
    results = []

    # ── Phase 1: HuggingFace ──────────────────────────────────────
    print("\n── HuggingFace ──────────────────────────────────────────")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INT4, cache_dir=MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_INT4, cache_dir=MODEL_DIR,
        torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    # Warmup
    warm = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=16, min_new_tokens=16, do_sample=False)

    for bs in BATCH_SIZES:
        r = run_hf_batch(bs, model, tokenizer)
        results.append(r)
        if r["tok_per_s"] > 0:
            print(f"  batch={bs}: {r['tok_per_s']} tok/s, VRAM={r['vram_mb']}MB")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: vLLM ─────────────────────────────────────────────
    print("\n── vLLM (PagedAttention) ────────────────────────────────")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_INT4, download_dir=MODEL_DIR,
        dtype="float16", enforce_eager=True,
        # Display driver uses 0.78 GiB permanently → max 3.22 GiB free → cap at 0.78.
        # INT4 model (~0.9 GiB) → KV cache ≈ 2.2 GiB → supports large batch sizes.
        gpu_memory_utilization=0.78, max_model_len=1024,
    )

    # Warmup
    llm.generate([TEST_PROMPT], SamplingParams(max_tokens=16, min_tokens=16, temperature=0))

    for bs in BATCH_SIZES:
        r = run_vllm_batch(bs, llm)
        results.append(r)
        print(f"  batch={bs}: {r['tok_per_s']} tok/s, VRAM={r['vram_mb']}MB")

    # ── Results ───────────────────────────────────────────────────
    print("\n=== Experiment 3 Results ===")
    print_table(results)

    if args.save:
        save_csv(results, "exp3_results.csv")
        plot_scaling(results)

if __name__ == "__main__":
    main()
