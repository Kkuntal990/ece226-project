# ECE 226 Project: Post-Training Quantization of LLMs

A comparative study of post-training quantization (PTQ) techniques applied to **Qwen2-1.5B**, evaluating the trade-offs between model compression, inference efficiency, and task accuracy across different quantization scopes.

## Project Overview

Large Language Models are expensive to deploy due to their memory and compute requirements. This project benchmarks three popular PTQ methods — **SmoothQuant**, **GPTQ**, and **AWQ** — each applied at three granularity levels (full model, attention-only, MLP-only) to understand which layers are most sensitive to quantization and which technique offers the best accuracy-compression trade-off.

### Quantization Techniques

| Technique | Precision | Description |
|-----------|-----------|-------------|
| **SmoothQuant** | W8A8 | Migrates quantization difficulty from activations to weights using a mathematically equivalent per-channel scaling transform. Uses calibration-driven smoothing (alpha=0.8). |
| **GPTQ** | W4 (group_size=128) | Second-order weight quantization that uses approximate Hessian information to minimize layer-wise reconstruction error. |
| **AWQ** | W4A16 (group_size=128) | Activation-Aware Weight Quantization that protects salient weight channels identified via activation magnitudes. |

### Quantization Scopes

Each technique is tested with three variants:
- **Full Quant** — All linear layers (attention + MLP) are quantized
- **Attention-Only** — Only self-attention projections (Q, K, V, O) are quantized
- **MLP-Only** — Only MLP projections (gate, up, down) are quantized

### Evaluation Benchmarks

- **Perplexity** — WikiText-2 test set (sliding window, stride=1024, max_length=2048)
- **ARC-Challenge** — 500 samples from AI2 ARC-Challenge (multiple-choice science reasoning)
- **GSM8K** — 300 samples, 8-shot chain-of-thought (math reasoning; GPTQ and AWQ notebooks only)

## Notebooks

| Notebook | Technique | Scope | Benchmarks |
|----------|-----------|-------|------------|
| `full_quant_smoothquant_ppl_arc.ipynb` | SmoothQuant W8A8 | Full model | Perplexity, ARC |
| `attn_only_smoothquant_ppl_arc (1).ipynb` | SmoothQuant W8A8 | Attention-only | Perplexity, ARC |
| `mlp_only_smoothquant_ppl_arc.ipynb` | SmoothQuant W8A8 | MLP-only | Perplexity, ARC |
| `gptq_full_quant_gsm8k_arc_only (1).ipynb` | GPTQ W4 g128 | Full model | Perplexity, GSM8K, ARC |
| `gptq_attn_only_quant_ppl_gsm8k_arc.ipynb` | GPTQ W4 g128 | Attention-only | Perplexity, GSM8K, ARC |
| `gptq_mlp_only_quant_ppl_gsm8k_arc.ipynb` | GPTQ W4 g128 | MLP-only | Perplexity, GSM8K, ARC |
| `02_awq_notebook.ipynb` | AWQ W4A16 g128 | All three + baseline | Perplexity, GSM8K, ARC, layer diagnostics |

## How to Run

### Requirements

- Google Colab with a GPU runtime (tested on NVIDIA L4 / T4)
- Python 3.10+
- No local installation required — all dependencies are installed within each notebook

### Steps

1. Upload the desired notebook to [Google Colab](https://colab.research.google.com/)
2. Select a **GPU runtime** (Runtime > Change runtime type > T4 or L4)
3. Run cells sequentially:
   - **Cell 1**: Installs dependencies (`transformers`, `datasets`, `accelerate`, `gptqmodel` for GPTQ, `llm-compressor` for AWQ)
   - **Cell 2**: Loads the base model, runs calibration (512 samples from WikiText-2 train), and quantizes
   - **Cell 3+**: Evaluates perplexity, ARC-Challenge, and GSM8K (if applicable)
   - **Final cell**: Prints summary metrics and saves result artifacts

### Notes

- For GPTQ notebooks, restart the runtime once after the `pip install gptqmodel` cell, then re-run from the next cell
- Each notebook is fully self-contained — no shared code files or external dependencies between notebooks
- Calibration data: 512 samples from WikiText-2 train split, max_length=2048, seed=42

## Key Results

### SmoothQuant (W8A8)

| Variant | Perplexity | ARC Accuracy | Artifact Size (GB) |
|---------|-----------|--------------|-------------------|
| Full Quant | 8.456 | 64.6% | 1.67 |
| Attn-Only | 8.395 | 65.6% | 2.75 |
| MLP-Only | 8.456 | 64.0% | 1.82 |

### GPTQ (W4, group_size=128)

| Variant | Perplexity | ARC Accuracy | Model Size (GB) | Bits/Param |
|---------|-----------|--------------|-----------------|-----------|
| Full Quant | 8.895 | 61.8% | 1.07 | 5.56 |
| Attn-Only | — | 64.8% | 2.66 | 13.84 |
| MLP-Only | 8.577 | 64.8% | 1.28 | 6.66 |

### AWQ (W4A16, group_size=128)

| Variant | Perplexity | ARC Accuracy | Peak VRAM (GB) |
|---------|-----------|--------------|----------------|
| Baseline (FP16) | 12.336 | 65.8% | 5.59 |
| Full Quant | 13.017 | 62.6% | 6.38 |
| Attn-Only | 12.545 | 64.0% | 5.71 |
| MLP-Only | 12.682 | 63.4% | 6.28 |

### Key Takeaways

1. **Attention layers are more robust to quantization than MLP layers.** Across all three techniques, attention-only quantization consistently preserves accuracy better than MLP-only or full quantization.
2. **SmoothQuant (W8A8) delivers the best accuracy retention** with perplexity nearly identical to FP16, at the cost of less compression (8-bit vs 4-bit).
3. **GPTQ provides the best compression ratio** with W4 quantization bringing the full model down to ~1 GB while maintaining reasonable accuracy (61.8% ARC).
4. **MLP layers dominate model size** — quantizing only MLP layers achieves nearly as much compression as full quantization, since MLP parameters make up ~57% of total weights in Qwen2-1.5B.
5. **AWQ layer diagnostics confirm** attention-only quantization has the highest cosine similarity (0.997) and lowest KL divergence (0.024) relative to the baseline, indicating minimal distribution shift.

## Base Model

- **Model**: [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)
- **Architecture**: 28 transformer layers, hidden_size=1536, intermediate_size=8960
- **Parameters**: ~1.5B
- **Original precision**: BFloat16
