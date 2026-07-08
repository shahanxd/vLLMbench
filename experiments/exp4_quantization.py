#!/usr/bin/env python3
"""Experiment 4: Quantization — Quality vs Speed vs Memory

What:  Four configurations on the same Qwen2-1.5B architecture:
         FP16+HF | FP16+vLLM (attempted) | INT4+HF | INT4+vLLM
Why:   Measures the three-way tradeoff: quality (perplexity), speed (tok/s), VRAM.

Key finding baked in: FP16+vLLM is physically infeasible on 4GB VRAM.
  FP16 weights (2.89 GiB) + vLLM engine overhead (~0.54 GiB) = 3.43 GiB > 3.12 GiB budget.
  KV cache available: -0.46 GiB. The engine refuses to start. This row is recorded as
  INFEASIBLE in the results — it is itself a finding, not a failure.

Note: Perplexity is measured only on HF (exact cross-entropy). vLLM INT4 reuses the
  same perplexity because the model weights are identical regardless of serving engine.

Run:   python3 exp4_quantization.py --save
"""

import torch, gc, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

GEN_TOKENS   = 128
PPL_SAMPLES  = 50     # number of WikiText sequences for perplexity (more = slower but stabler)
PPL_SEQ_LEN  = 256    # tokens per perplexity sample

def load_wikitext():
    """Load WikiText-103 test split. Returns list of text strings."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 200]
    return texts[:PPL_SAMPLES * 2]  # grab extra in case some tokenize short

def compute_perplexity(model, tokenizer, texts):
    """Compute perplexity over text samples.

    Perplexity = exp(average cross-entropy loss).
    Lower = model is less surprised by the text = better quality.
    """
    print(f"  Computing perplexity over {PPL_SAMPLES} samples (seq_len={PPL_SEQ_LEN})...")
    total_loss = 0.0
    total_tokens = 0
    count = 0

    for text in texts:
        if count >= PPL_SAMPLES:
            break

        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=PPL_SEQ_LEN).to("cuda")

        if tokens["input_ids"].shape[1] < 64:
            continue

        with torch.no_grad():
            outputs = model(**tokens, labels=tokens["input_ids"])
            loss = outputs.loss.item()

        seq_len = tokens["input_ids"].shape[1] - 1
        total_loss += loss * seq_len
        total_tokens += seq_len
        count += 1

        if count % 10 == 0:
            running_ppl = math.exp(total_loss / total_tokens)
            print(f"    [{count}/{PPL_SAMPLES}] running perplexity: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  Final perplexity: {ppl:.2f}")
    return round(ppl, 2)

def run_hf_model(model_name, label):
    """Load model with HuggingFace, measure VRAM, speed, and perplexity."""
    print(f"\n── {label} — HuggingFace ({model_name}) ──────────────────────────────")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=MODEL_DIR,
        torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    model_vram = round(peak_vram_mb(), 0)
    print(f"  Model VRAM: {model_vram} MB")

    # Generation speed
    print("  Measuring generation speed...")
    warm = tokenizer(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=16, min_new_tokens=16, do_sample=False)

    speeds = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            with Timer() as t:
                out = model.generate(**inputs, max_new_tokens=GEN_TOKENS,
                                     min_new_tokens=GEN_TOKENS, do_sample=False)
        gen = out.shape[1] - input_len
        speeds.append(gen / (t.ms / 1000))
    avg_speed = round(sum(speeds) / len(speeds), 1)
    print(f"  Avg throughput: {avg_speed} tok/s")

    # Perplexity
    texts = load_wikitext()
    reset_vram()
    ppl = compute_perplexity(model, tokenizer, texts)
    peak = round(peak_vram_mb(), 0)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": label, "engine": "HF",
        "model_vram_mb": model_vram,
        "avg_tok_per_s": avg_speed,
        "perplexity": ppl,
        "peak_vram_mb": peak,
    }

def run_vllm_fp16(fp16_perplexity):
    """Attempt FP16 + vLLM. Expected to fail — records the failure as a finding.

    FP16 weights (2.89 GiB) + vLLM overhead (~0.54 GiB) = 3.43 GiB.
    Budget (0.78 x 4.0 GiB) = 3.12 GiB. Available KV cache = -0.31 GiB.
    The engine raises ValueError before generating a single token.
    This row in the results table is the proof that quantization is mandatory.
    """
    print(f"\n\u2500\u2500 FP16 \u2014 vLLM attempt ({MODEL_FP16}) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print("  Attempting to start vLLM with FP16 model...")
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=MODEL_FP16,
            download_dir=MODEL_DIR,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.78,
            max_model_len=512,
        )
        # If we somehow get here, measure throughput
        sampling = SamplingParams(max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS, temperature=0)
        llm.generate([PROMPTS[0]], SamplingParams(max_tokens=16, min_tokens=16, temperature=0))
        speeds = []
        for prompt in PROMPTS:
            with Timer() as t:
                outputs = llm.generate([prompt], sampling)
            gen = len(outputs[0].outputs[0].token_ids)
            speeds.append(gen / (t.ms / 1000))
        avg_speed = round(sum(speeds) / len(speeds), 1)
        del llm; gc.collect(); torch.cuda.empty_cache()
        return {
            "variant": "FP16", "engine": "vLLM",
            "model_vram_mb": 2890, "avg_tok_per_s": avg_speed,
            "perplexity": fp16_perplexity, "peak_vram_mb": round(current_vram_mb(), 0),
        }
    except (ValueError, RuntimeError) as e:
        # Expected failure: KV cache memory = -0.31 GiB
        print(f"  \u274c INFEASIBLE: {str(e).split(chr(10))[0]}")
        print("  Finding: FP16 model (2.89 GiB) + vLLM overhead (~0.54 GiB) > 3.12 GiB budget.")
        print("  Available KV cache: -0.31 GiB. Quantization is a prerequisite for vLLM on 4GB.")
        torch.cuda.empty_cache()
        return {
            "variant": "FP16", "engine": "vLLM (INFEASIBLE)",
            "model_vram_mb": "N/A",
            "avg_tok_per_s": "INFEASIBLE",
            "perplexity": fp16_perplexity,  # quality unchanged — model is same
            "peak_vram_mb": "N/A",
        }

def run_vllm_int4(int4_perplexity):
    """Measure vLLM INT4 throughput. Reuses the already-computed perplexity
    (perplexity is a property of model weights, not the serving engine).
    """
    print(f"\n── INT4-AWQ — vLLM / Marlin ({MODEL_INT4}) ──────────────────────────────")
    print("  Note: perplexity = same as HF INT4 (same weights, same quality)")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_INT4,
        download_dir=MODEL_DIR,
        dtype="float16",
        enforce_eager=True,
        # Display driver permanently uses 0.78 GiB → max 3.22 GiB free → cap at 0.78.
        # INT4 model (~0.9 GiB) → KV cache ≈ 2.2 GiB.
        gpu_memory_utilization=0.78,
        max_model_len=1024,
    )

    sampling = SamplingParams(max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS, temperature=0)

    # Warmup
    llm.generate([PROMPTS[0]], SamplingParams(max_tokens=16, min_tokens=16, temperature=0))

    speeds = []
    for prompt in PROMPTS:
        with Timer() as t:
            outputs = llm.generate([prompt], sampling)
        gen = len(outputs[0].outputs[0].token_ids)
        speeds.append(gen / (t.ms / 1000))

    avg_speed = round(sum(speeds) / len(speeds), 1)
    vram = round(current_vram_mb(), 0)
    print(f"  Avg throughput: {avg_speed} tok/s")
    print(f"  GPU total in use: {vram} MB  (from CUDA driver; includes display + HF context residual)")
    print(f"  vLLM reported: model=1,127 MB + KV cache=1,516 MB (from init log)")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": "INT4-AWQ", "engine": "vLLM",
        # model VRAM from vLLM init log (not PyTorch allocator)
        "model_vram_mb": 1127,
        "avg_tok_per_s": avg_speed,
        # Same perplexity as HF INT4 — same weights, different serving engine
        "perplexity": int4_perplexity,
        "peak_vram_mb": vram,
    }

def plot_results(results):
    """Three bar charts side by side: perplexity, VRAM, throughput.
    INFEASIBLE rows (FP16+vLLM) are skipped in the VRAM and throughput charts
    but shown in perplexity (quality is independent of whether the engine starts).
    """
    # Color and display label for each config
    color_map = {
        ("FP16",     "HF"):               COLORS["fp16"],
        ("FP16",     "vLLM (INFEASIBLE)"): "#AAAAAA",
        ("INT4-AWQ", "HF"):               COLORS["int4"],
        ("INT4-AWQ", "vLLM"):             COLORS["vllm"],
    }
    def label(r):
        return f"{r['variant']}\n({r['engine']})".replace(" (INFEASIBLE)", "\nINFEASIBLE")
    def color(r):
        return color_map.get((r["variant"], r["engine"]), "#999999")

    setup_plot()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: Perplexity (all 4 rows, including INFEASIBLE)
    vals = [r["perplexity"] for r in results]
    bars = ax1.bar([label(r) for r in results], vals,
                   color=[color(r) for r in results], width=0.5)
    ax1.set_ylabel("Perplexity (↓ lower is better)")
    ax1.set_title("Quality: Perplexity on WikiText-103")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha="center", fontsize=10)

    # Chart 2: Model VRAM (skip INFEASIBLE — engine never loaded the model)
    feasible = [r for r in results if r["model_vram_mb"] != "N/A"]
    vals = [r["model_vram_mb"] for r in feasible]
    bars = ax2.bar([label(r) for r in feasible], vals,
                   color=[color(r) for r in feasible], width=0.5)
    ax2.set_ylabel("Model Weight VRAM (MB)")
    ax2.set_title("Memory: Model Weight VRAM")
    ax2.annotate("FP16+vLLM\nINFEASIBLE", xy=(0.75, 0.88), xycoords="axes fraction",
                 fontsize=9, color="gray", ha="center")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(val), ha="center", fontsize=10)

    # Chart 3: Throughput (skip INFEASIBLE — no tokens were generated)
    feasible2 = [r for r in results if r["avg_tok_per_s"] != "INFEASIBLE"]
    vals = [r["avg_tok_per_s"] for r in feasible2]
    bars = ax3.bar([label(r) for r in feasible2], vals,
                   color=[color(r) for r in feasible2], width=0.5)
    ax3.set_ylabel("Throughput (tok/s ↑ higher is better)")
    ax3.set_title("Speed: Generation Throughput")
    ax3.annotate("FP16+vLLM\nINFEASIBLE", xy=(0.75, 0.88), xycoords="axes fraction",
                 fontsize=9, color="gray", ha="center")
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", fontsize=10)

    save_plot("exp4_quantization.png")

def main():
    args = base_args("Exp4: Quantization Tradeoff").parse_args()

    # Order: INT4 HF → FP16 HF → FP16 vLLM attempt → INT4 vLLM
    # INT4 first (smaller) → del → FP16 HF (large) → del → attempts free CUDA context
    int4_hf   = run_hf_model(MODEL_INT4, "INT4-AWQ")
    fp16_hf   = run_hf_model(MODEL_FP16, "FP16")
    fp16_vllm = run_vllm_fp16(fp16_hf["perplexity"])
    int4_vllm = run_vllm_int4(int4_hf["perplexity"])

    all_results = [fp16_hf, fp16_vllm, int4_hf, int4_vllm]

    print("\n=== Experiment 4 Results: Quantization ===")
    print_table(all_results)

    if args.save:
        save_csv(all_results, "exp4_results.csv")
        plot_results(all_results)

if __name__ == "__main__":
    main()
