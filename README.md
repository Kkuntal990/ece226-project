# Post-Training Quantization of LLMs

A comparative study of post-training quantization (PTQ) techniques applied to **Qwen2-1.5B**, evaluating trade-offs between model compression, inference efficiency, and task accuracy across different quantization methods and scopes.

## Project Overview

Large Language Models are expensive to deploy due to their memory and compute requirements. This project benchmarks four PTQ methods — **RTN**, **GPTQ**, **SmoothQuant**, and **AWQ** — each applied at three granularity levels (full model, attention-only, MLP-only) to understand which layers are most sensitive to quantization and how different techniques navigate the accuracy–compression–speed trade-off.

### Methods

| Method | Precision | Approach |
|--------|-----------|----------|
| **RTN** | W8A16 | Round-to-nearest via bitsandbytes INT8. No calibration data required. |
| **GPTQ** | W4 (group_size=128) | Second-order weight quantization using approximate Hessian information to minimize layer-wise reconstruction error. |
| **SmoothQuant** | W8A8 | Migrates quantization difficulty from activations to weights via per-channel scaling (alpha=0.8). Quantizes both weights and activations. |
| **AWQ** | W4A16 (group_size=128) | Activation-aware weight quantization that protects salient weight channels identified via activation magnitudes. |

### Quantization Scopes

Each method is evaluated with three scopes:
- **Full** — All linear layers (attention + MLP) are quantized
- **Attention-Only** — Only self-attention projections (Q, K, V, O)
- **MLP-Only** — Only MLP projections (gate, up, down)

### Evaluation

- **ARC-Challenge** — 500 samples, 0-shot multiple-choice reasoning
- **WikiText-2 Perplexity** — Sliding window evaluation (max_length=512, stride=256)
- **GSM8K** — 8-shot chain-of-thought math reasoning (RTN: 300 samples, AWQ: 50 samples)
- **Deployment metrics** — Throughput (tok/s), latency (ms/tok), peak VRAM, model size

### Base Model

- **Model**: [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)
- **Architecture**: 28 transformer layers, hidden_size=1536, intermediate_size=8960, ~1.5B parameters
- **Original precision**: BFloat16 (2.88 GB)

## Repository Structure

```
├── notebooks/
│   ├── rtn_eval.ipynb                  # RTN (W8A16) — all scopes + baseline
│   ├── awq_eval.ipynb                  # AWQ (W4A16) — all scopes + baseline
│   ├── gptq_full_eval.ipynb            # GPTQ (W4) — full model
│   ├── gptq_attn_eval.ipynb            # GPTQ (W4) — attention-only
│   ├── gptq_mlp_eval.ipynb             # GPTQ (W4) — MLP-only
│   ├── smoothquant_full_eval.ipynb     # SmoothQuant (W8A8) — full model
│   ├── smoothquant_attn_eval.ipynb     # SmoothQuant (W8A8) — attention-only
│   ├── smoothquant_mlp_eval.ipynb      # SmoothQuant (W8A8) — MLP-only
│   └── results_analysis.ipynb          # Cross-method analysis and visualization
├── results/
│   ├── baseline/                       # FP16 baseline evaluation
│   ├── rtn-eval/                       # RTN results + layer diagnostics
│   ├── awq-eval/                       # AWQ results + logit diagnostics
│   ├── full_quant_results_only/        # GPTQ full results
│   ├── attn_only_quant_results_only/   # GPTQ attn-only results
│   ├── mlp_only_quant_results_only/    # GPTQ MLP-only results
│   ├── smoothquant-eval/               # SmoothQuant results (all scopes)
│   └── analysis/                       # Generated figures and tables
├── scripts/
│   ├── generate_analysis.py            # Generates all analysis figures
│   ├── plot_literature_figures.py      # Literature-informed deployment plots
│   └── plot_pareto_multidataset.py     # Multi-benchmark Pareto plots
```

## How to Run

### Requirements

- Google Colab with a GPU runtime (tested on NVIDIA L4 / T4)
- Python 3.10+
- No local installation required — all dependencies are installed within each notebook

### Steps

1. Upload the desired notebook to [Google Colab](https://colab.research.google.com/)
2. Select a **GPU runtime** (Runtime > Change runtime type > T4 or L4)
3. Run cells sequentially — each notebook handles dependency installation, model loading, quantization, and evaluation
4. Results are saved as JSON/CSV artifacts in the notebook's output directory

### Notes

- Each notebook is fully self-contained with no shared dependencies between them
- For GPTQ notebooks, restart the runtime after the `pip install gptqmodel` cell, then re-run from the next cell
- The `results_analysis.ipynb` notebook reads from `results/` and regenerates all figures

## Key Results

### Cross-Method Comparison

| Method | Scope | Precision | ARC (%) | Size (GB) | VRAM (GB) | Throughput (tok/s) |
|--------|-------|-----------|:-------:|:---------:|:---------:|:-----------------:|
| Baseline (FP16) | Full | FP16 | 65.8 | 2.88 | 5.81 | 27.2 |
| RTN | Full | W8A16 | 65.4 | 1.72 | 1.72 | 13.2 |
| RTN | MLP-Only | W8A16 | 66.4 | 1.82 | 1.82 | 23.2 |
| SmoothQuant | Attn-Only | W8A8 | 65.6 | 2.75 | 2.77 | 14.0 |
| GPTQ | MLP-Only | W4 | 64.8 | 1.28 | 5.71 | 19.0 |
| AWQ | MLP-Only | W4A16 | 63.4 | 1.12 | 6.49 | 12.3 |

### Key Findings

1. **Accuracy is broadly preserved across all methods.** ARC-Challenge scores span only 4.6 pp (61.8%–66.4%) despite compression ratios up to 2.7×. With 500 evaluation samples (95% CI ≈ ±4 pp), differences between 8-bit methods and the baseline are not statistically significant, while the gap for 4-bit full-model quantization is meaningful.

2. **MLP layers are the dominant source of quantization error.** RTN logit diagnostics show that Full and MLP-Only quantization produce nearly identical output degradation (KL divergence ~0.008), while Attn-Only preserves 98.6% top-1 agreement. This confirms that MLP layers — which hold ~70% of parameters in Qwen2's SwiGLU architecture — are harder to quantize.

3. **No method achieves inference speed-up over the FP16 baseline.** Despite significant model compression, all quantized methods exhibit lower throughput than FP16 due to dequantization overhead and lack of optimized quantized kernels in the evaluation setup.

4. **RTN and SmoothQuant achieve the best VRAM savings** (1.7–2.8 GB vs 5.8 GB baseline), while GPTQ and AWQ show higher VRAM than expected due to runtime dequantization buffers.

5. **Perplexity comparisons require matching evaluation configs.** RTN/AWQ use max_length=512, stride=256 while GPTQ/SmoothQuant use max_length=2048, stride=1024 — making cross-method PPL values not directly comparable.

## Contributors

- **Kuntal Kokate**
- **Aidan**
- **Manasvin**
