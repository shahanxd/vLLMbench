#!/usr/bin/env python3
"""Experiment 1: HuggingFace Baseline (The Control)

What:  Standard HuggingFace model.generate() — no PagedAttention, no batching tricks.
Why:   Every other experiment is compared to these numbers.
Input: 5 fixed prompts, generate 128 tokens each, single request at a time.
Output: TTFT, TPOT, total latency, peak VRAM per prompt.

Run:   python3 exp1_hf_baseline.py --save
"""

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

GEN_TOKENS = 128  # tokens to generate per prompt

def main():
    args = base_args("Exp1: HF Baseline").parse_args()

    # ── Load model ─────────────────────────────────────────────────
    print("Loading Qwen2-1.5B (FP16) with HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FP16, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_FP16, cache_dir=MODEL_DIR,
        torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    # ── Warmup (first run has CUDA kernel compilation overhead) ────
    print("Warmup run (discarded)...")
    warm_in = tokenizer(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**warm_in, max_new_tokens=16, min_new_tokens=16, do_sample=False)

    # ── Run each prompt ────────────────────────────────────────────
    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:60]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Measure TTFT: time to produce the first token
            # (= prefill time + one decode step, approximately)
            reset_vram()
            with Timer() as t1:
                model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
            ttft = t1.ms

            # Measure full generation
            reset_vram()
            with Timer() as t_full:
                # min_new_tokens forces exactly GEN_TOKENS — prevents early EOS skewing timing
                out = model.generate(**inputs, max_new_tokens=GEN_TOKENS,
                                     min_new_tokens=GEN_TOKENS, do_sample=False)
            total = t_full.ms

        gen_count = out.shape[1] - input_len
        # TPOT = decode time / decode tokens. Decode time ≈ total - TTFT.
        tpot = (total - ttft) / max(gen_count - 1, 1)
        vram = peak_vram_mb()

        results.append({
            "prompt":     i + 1,
            "in_tokens":  input_len,
            "out_tokens": gen_count,
            "ttft_ms":    round(ttft, 1),
            "tpot_ms":    round(tpot, 1),
            "total_ms":   round(total, 1),
            "vram_mb":    round(vram, 0),
        })
        print(f"  TTFT={ttft:.0f}ms  TPOT={tpot:.1f}ms/tok  Total={total:.0f}ms  VRAM={vram:.0f}MB")

    # ── Results ────────────────────────────────────────────────────
    print("\n=== Experiment 1 Results: HF Baseline ===")
    print_table(results)

    if args.save:
        save_csv(results, "exp1_results.csv")

if __name__ == "__main__":
    main()
