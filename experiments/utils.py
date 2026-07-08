"""Shared utilities for all experiments.
Timing, VRAM measurement, results saving, plotting setup.
"""

import time, os, csv, warnings, torch, argparse
import matplotlib
matplotlib.use('Agg')  # No display inside Docker — saves PNGs only
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Suppress repetitive transformers warnings about temperature/top_p/top_k
# when do_sample=False. These fire on every generate() call and add no value.
warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")

# ── Constants ──────────────────────────────────────────────────────────

MODEL_FP16 = "Qwen/Qwen2-1.5B-Instruct"
MODEL_INT4 = "Qwen/Qwen2-1.5B-Instruct-AWQ"
MODEL_DIR  = "/models"                       # mounted volume inside container
RESULTS    = "/workspace/results"            # also mounted — visible from Windows

# Fixed prompts — same across all experiments for reproducibility
PROMPTS = [
    "Explain how a CPU processes instructions step by step.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in simple terms.",
    "List five important principles of software engineering.",
]

# ── Timing ─────────────────────────────────────────────────────────────

class Timer:
    """Context manager that times a block of code.
    Uses cuda synchronize for accurate GPU timing.

    Usage:
        with Timer() as t:
            do_stuff()
        print(t.ms)
    """
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.start) * 1000

# ── VRAM helpers ───────────────────────────────────────────────────────

def reset_vram():
    """Reset peak VRAM tracking. Call before each measurement."""
    torch.cuda.reset_peak_memory_stats()

def peak_vram_mb():
    """Return peak VRAM used (MB) since last reset_vram() call.
    Uses PyTorch's allocator stats — accurate for HF generate() calls.
    Returns 0 for vLLM since vLLM uses raw CUDA allocation, not PyTorch's allocator.
    Use current_vram_mb() for vLLM instead.
    """
    return torch.cuda.max_memory_allocated() / 1024**2

def current_vram_mb():
    """Return TOTAL GPU VRAM currently in use (MB), read from the CUDA driver.
    Captures ALL allocations: PyTorch tensors + vLLM's KV cache pool + display driver.
    Use this for vLLM measurements where peak_vram_mb() returns 0.
    """
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2

# ── Results I/O ────────────────────────────────────────────────────────

def save_csv(rows, filename):
    """Save list-of-dicts to CSV in the results directory."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")

def print_table(rows):
    """Pretty-print a list-of-dicts as an aligned table."""
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print()

# ── Plot helpers ───────────────────────────────────────────────────────

# Clean, publication-ready style
COLORS = {"hf": "#4A90D9", "vllm": "#E55934", "fp16": "#4A90D9", "int4": "#2ECC71"}

def setup_plot():
    """Apply clean plot styling. Call once per plot."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

def save_plot(filename):
    """Save current figure to results dir as PNG."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

# ── CLI ────────────────────────────────────────────────────────────────

def base_args(description):
    """Return an argparse parser with the --save flag."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--save", action="store_true",
                   help="Save CSV results and PNG plots to /workspace/results/")
    return p
