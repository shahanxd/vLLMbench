# GPU Memory Optimization: LLM Inference Benchmarking

Comparative performance study of **HuggingFace** vs **vLLM (PagedAttention)** inference engines on constrained 4GB VRAM hardware (RTX 3050).

## Research Question

Can PagedAttention + quantization-aware serving overcome the memory and throughput limitations of standard HuggingFace inference on consumer-grade GPUs?

## Key Results

| Config | Engine | Throughput | VRAM (model) | Perplexity |
|---|---|---|---|---|
| FP16 | HuggingFace | 31 tok/s | 2,954 MB | 16.55 |
| FP16 | vLLM | **INFEASIBLE** | — | 16.55 |
| INT4-AWQ | HuggingFace | 20 tok/s | 1,544 MB | 18.41 |
| INT4-AWQ | vLLM | **59 tok/s** | 1,127 MB | 18.41 |

At batch=8, vLLM achieves **460 tok/s** vs HuggingFace's 156 tok/s — a **2.9× advantage**.

## Hardware

- GPU: NVIDIA RTX 3050 Laptop (4GB VRAM, 3.22 GiB usable after display driver)
- CPU: Intel Core i7
- RAM: 16GB
- OS: Windows 11 + WSL2 (Docker container)

## Model

- **FP16:** `Qwen/Qwen2-1.5B-Instruct` (HuggingFace Hub)
- **INT4:** `Qwen/Qwen2-1.5B-Instruct-AWQ` (HuggingFace Hub, pre-quantized AWQ)

## Experiments

| Script | What it measures |
|---|---|
| `exp1_hf_baseline.py` | HF FP16 baseline: TTFT, TPOT, throughput, VRAM |
| `exp2_pagedattn_vs_hf.py` | HF vs vLLM at batch=1 using INT4 (isolates engine effect) |
| `exp3_batch_scaling.py` | Throughput scaling at batch=1/2/4/8 for both engines |
| `exp4_quantization.py` | FP16 vs INT4 tradeoff: quality (perplexity), speed, VRAM |

## Reproduction

### Requirements
- Docker with NVIDIA GPU support (`--gpus all`)
- Model cache at `/models` inside container (or set `MODEL_DIR` in `utils.py`)

### Setup

```bash
# Pull a vLLM-ready container (includes PyTorch + CUDA)
docker run --gpus all --rm -it \
  -v "$(pwd)/experiments:/workspace" \
  -v "/path/to/model/cache:/models" \
  vllm/vllm-openai:latest bash

# Inside container
pip install autoawq python-docx
```

### Running experiments

> ⚠️ **VRAM constraint:** Run Exp 2 backends in separate processes to avoid CUDA context residual crashing the second engine.

```bash
# Exp 1 — HF FP16 baseline
python3 exp1_hf_baseline.py --save

# Exp 2 — run backends separately
python3 exp2_pagedattn_vs_hf.py --backend hf   --save
python3 exp2_pagedattn_vs_hf.py --backend vllm --save   # auto-generates comparison plot

# Exp 3 — batch scaling
python3 exp3_batch_scaling.py --save

# Exp 4 — quantization tradeoff
python3 exp4_quantization.py --save
```

Results are saved to `shared/results/` as CSVs + PNG plots.

## Software Versions

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.7.0+cu128 |
| transformers | 4.51.3 |
| vLLM | 0.9.1 |
| autoawq | 0.2.9 |
| CUDA | 12.8 |

## VRAM Measurement Note

- **HuggingFace rows:** `torch.cuda.max_memory_allocated()` — accurate peak PyTorch allocation.
- **vLLM rows:** Cited from engine initialization logs (authoritative). `torch.cuda.mem_get_info()` from the parent process is unreliable for vLLM's subprocess allocations under WSL2.

## Repository Structure

```
GPU MemOpt/
├── .gitignore
├── .gitattributes
├── README.md
└── experiments/
    ├── utils.py                    # Shared timing, VRAM, plotting utilities
    ├── exp1_hf_baseline.py
    ├── exp2_pagedattn_vs_hf.py
    ├── exp3_batch_scaling.py
    ├── exp4_quantization.py
    └── results/
        ├── exp1_results.csv
        ├── exp2_hf_results.csv
        ├── exp2_vllm_results.csv
        ├── exp2_comparison.png
        ├── exp3_results.csv
        ├── exp3_throughput.png
        ├── exp4_results.csv
        └── exp4_quantization.png
```
