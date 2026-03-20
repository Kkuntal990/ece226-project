#!/usr/bin/env python3
"""
ECE 226 — Post-Training Quantization Comparative Analysis
Generates cross-method comparison tables, plots, and per-method deep-dive figures.

Model: Qwen2-1.5B
Methods: RTN (W8A16), GPTQ (W4), SmoothQuant (W8A8), AWQ (W4A16)
Variants: Full, Attention-only, MLP-only
"""

import json, math, os, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── paths ───────────────────────────────────────────────────────────────────
ROOT   = Path("/tmp/ece226_results")
OUTDIR = Path("/tmp/ece226_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"RTN (W8A16)": "#4C72B0", "GPTQ (W4)": "#DD8452",
           "SmoothQuant (W8A8)": "#55A868", "AWQ (W4A16)": "#C44E52",
           "Baseline (FP16)": "#8C8C8C"}
VARIANT_ORDER = ["full_quant", "attn_only", "mlp_only"]
VARIANT_LABELS = {
    "full_quant": "Full", "attn_only": "Attn-Only", "mlp_only": "MLP-Only",
    "attn_only_quant": "Attn-Only", "mlp_only_quant": "MLP-Only",
}

# ── helpers ─────────────────────────────────────────────────────────────────
def load_json(p):
    with open(p) as f: return json.load(f)

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ═════════════════════════════════════════════════════════════════════════════
# 1.  ASSEMBLE UNIFIED DATAFRAME
# ═════════════════════════════════════════════════════════════════════════════
records = []

# Helper to normalize variant names
def norm_variant(v):
    return {"attn_only_quant": "attn_only", "mlp_only_quant": "mlp_only"}.get(v, v)

# ── Baseline ────────────────────────────────────────────────────────────────
bl = load_jsonl(ROOT / "baseline/baseline-eval.jsonl")[0]
records.append({
    "method": "Baseline (FP16)", "variant": "full_quant",
    "precision": "FP16", "bits_per_param": 16.0,
    "ppl": bl.get("ppl"), "arc": bl.get("arc_challenge"),
    "gsm8k": bl.get("gsm8k"),
    "model_size_gb": bl.get("model_size", 2.88),
    "throughput_tps": bl.get("tok_s"),
    "latency_ms": bl.get("ms_per_token"),
    "peak_vram_gb": bl.get("peak_vram_gb"),
})

# ── RTN (W8A16) ────────────────────────────────────────────────────────────
for row in load_jsonl(ROOT / "rtn-eval/rtn_eval.jsonl"):
    records.append({
        "method": "RTN (W8A16)", "variant": norm_variant(row["variant"]),
        "precision": "W8A16",
        "bits_per_param": row.get("bytes_per_param_actual", 0) * 4,
        "ppl": row.get("ppl"), "arc": row.get("arc_challenge"),
        "gsm8k": row.get("gsm8k"),
        "model_size_gb": row.get("peak_vram_gb"),
        "throughput_tps": row.get("tok_s"),
        "latency_ms": row.get("ms_per_token"),
        "peak_vram_gb": row.get("peak_vram_gb"),
    })

# ── AWQ (W4A16) ────────────────────────────────────────────────────────────
for row in load_jsonl(ROOT / "awq-eval/awq_eval.jsonl"):
    if row.get("method") == "baseline":
        continue  # already added
    records.append({
        "method": "AWQ (W4A16)", "variant": norm_variant(row.get("variant", "full_quant")),
        "precision": "W4A16",
        "bits_per_param": row.get("bytes_per_param_actual", 0) * 8,
        "ppl": row.get("ppl"), "arc": row.get("arc_challenge"),
        "gsm8k": row.get("gsm8k"),
        "model_size_gb": row.get("model_size"),
        "throughput_tps": row.get("tok_s"),
        "latency_ms": row.get("ms_per_token"),
        "peak_vram_gb": row.get("peak_vram_gb"),
    })

# ── GPTQ (W4) ──────────────────────────────────────────────────────────────
gptq_dirs = {
    "full_quant": ROOT / "full_quant_results_only",
    "attn_only":  ROOT / "attn_only_quant_results_only",
    "mlp_only":   ROOT / "mlp_only_quant_results_only",
}
for var, d in gptq_dirs.items():
    summary = load_json(d / "summary.json") if (d / "summary.json").exists() else {}
    arc = load_json(d / "arc_challenge_metrics.json") if (d / "arc_challenge_metrics.json").exists() else {}
    ppl_data = load_json(d / "perplexity_metrics.json") if (d / "perplexity_metrics.json").exists() else {}
    manifest = load_json(d / "quantization_manifest.json") if (d / "quantization_manifest.json").exists() else {}

    ppl_val = summary.get("perplexity") or ppl_data.get("perplexity")
    arc_val = summary.get("arc_accuracy") or (arc.get("accuracy") if arc else None)
    size_gb = manifest.get("model_size_gb")
    bits = manifest.get("bits_per_param")

    records.append({
        "method": "GPTQ (W4)", "variant": var,
        "precision": "W4",
        "bits_per_param": bits,
        "ppl": ppl_val, "arc": arc_val, "gsm8k": None,
        "model_size_gb": size_gb,
        "throughput_tps": None, "latency_ms": None,
        "peak_vram_gb": arc.get("peak_vram_gb"),
    })

# ── SmoothQuant (W8A8) ─────────────────────────────────────────────────────
sq_dirs = {
    "full_quant": ROOT / "smoothquant-eval/smoothquant_full_quant_results_only/full_quant",
    "attn_only":  ROOT / "smoothquant-eval/smoothquant_attn_only_results_only/attn_only",
    "mlp_only":   ROOT / "smoothquant-eval/smoothquant_mlp_only_results_only/mlp_only",
}
for var, d in sq_dirs.items():
    ppl_data = load_json(d / "perplexity_metrics.json") if (d / "perplexity_metrics.json").exists() else {}
    arc_data = load_json(d / "arc_challenge_metrics.json") if (d / "arc_challenge_metrics.json").exists() else {}
    state    = load_json(d / "results_state.json") if (d / "results_state.json").exists() else {}
    size_gb  = state.get("artifact", {}).get("size_gb")

    records.append({
        "method": "SmoothQuant (W8A8)", "variant": var,
        "precision": "W8A8",
        "bits_per_param": 8.0,
        "ppl": ppl_data.get("perplexity"), "arc": arc_data.get("accuracy"),
        "gsm8k": None,
        "model_size_gb": size_gb,
        "throughput_tps": None, "latency_ms": None,
        "peak_vram_gb": arc_data.get("peak_vram_gb") or ppl_data.get("peak_vram_gb"),
    })

df = pd.DataFrame(records)

# Overwrite model sizes and deployment metrics from the curated comprehensive CSV
comp_csv = OUTDIR / "table_comprehensive.csv"
if comp_csv.exists():
    comp = pd.read_csv(comp_csv)
    SCOPE_MAP = {"Full": "full_quant", "Attn-Only": "attn_only", "MLP-Only": "mlp_only"}
    COL_MAP = {"Size(GB)": "model_size_gb", "VRAM(GB)": "peak_vram_gb",
               "Throughput": "throughput_tps", "Latency(ms)": "latency_ms"}
    for _, cr in comp.iterrows():
        var = SCOPE_MAP.get(cr["Scope"], cr["Scope"])
        mask = (df["method"] == cr["Method"]) & (df["variant"] == var)
        if mask.any():
            for csv_col, df_col in COL_MAP.items():
                if csv_col in cr and not pd.isna(cr[csv_col]):
                    df.loc[mask, df_col] = cr[csv_col]

# Convert arc from fraction to percentage
df["arc_pct"] = df["arc"].apply(lambda x: x * 100 if x is not None and x <= 1 else x)
df["gsm8k_pct"] = df["gsm8k"].apply(lambda x: x * 100 if x is not None and x <= 1 else x)
df["variant_label"] = df["variant"].map(VARIANT_LABELS)

print("=" * 80)
print("ASSEMBLED DATA")
print("=" * 80)
print(df[["method", "variant", "ppl", "arc_pct", "gsm8k_pct", "model_size_gb"]].to_string(index=False))
print()

# ═════════════════════════════════════════════════════════════════════════════
# 2.  CROSS-METHOD COMPARISON TABLE (LaTeX-ready)
# ═════════════════════════════════════════════════════════════════════════════
table_cols = ["method", "variant_label", "precision", "ppl", "arc_pct", "gsm8k_pct", "model_size_gb", "peak_vram_gb"]
table_df = df[table_cols].copy()
table_df.columns = ["Method", "Variant", "Precision", "PPL (↓)", "ARC-C (%↑)", "GSM8K (%↑)", "Size (GB)", "VRAM (GB)"]
table_df = table_df.sort_values(["Method", "Variant"])

print("=" * 80)
print("TABLE 1: Cross-Method Comparison")
print("=" * 80)
print(table_df.to_string(index=False, float_format="%.2f", na_rep="—"))
table_df.to_csv(OUTDIR / "table1_cross_method_comparison.csv", index=False, float_format="%.4f")
print(f"\nSaved to {OUTDIR / 'table1_cross_method_comparison.csv'}")

# ═════════════════════════════════════════════════════════════════════════════
# 3.  PLOT: ARC-Challenge Accuracy (common metric across ALL methods)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
plot_df = df[df["method"] != "Baseline (FP16)"].copy()
plot_df["variant_label"] = pd.Categorical(plot_df["variant_label"], categories=["Full", "Attn-Only", "MLP-Only"])

bars = sns.barplot(data=plot_df, x="variant_label", y="arc_pct", hue="method",
                   palette=PALETTE, ax=ax, edgecolor="black", linewidth=0.5)

# Baseline reference line
bl_arc = df.loc[df["method"] == "Baseline (FP16)", "arc_pct"].values[0]
ax.axhline(bl_arc, color=PALETTE["Baseline (FP16)"], ls="--", lw=2, label=f"Baseline FP16 ({bl_arc:.1f}%)")

ax.set_ylabel("ARC-Challenge Accuracy (%)")
ax.set_xlabel("Quantization Scope")
ax.set_title("ARC-Challenge Accuracy: Quantization Method × Scope\n(Qwen2-1.5B, 500 samples, 0-shot)", fontsize=13)
ax.set_ylim(58, 70)
ax.legend(title="Method", loc="lower right", fontsize=9)

# Add value labels on bars
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTDIR / "fig1_arc_accuracy_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig1_arc_accuracy_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"Saved fig1_arc_accuracy_comparison")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  PLOT: Perplexity Comparison (note: different eval configs per method)
# ═════════════════════════════════════════════════════════════════════════════
ppl_df = df.dropna(subset=["ppl"]).copy()

fig, ax = plt.subplots(figsize=(12, 6))
ppl_plot = ppl_df[ppl_df["method"] != "Baseline (FP16)"]
ppl_plot["variant_label"] = pd.Categorical(ppl_plot["variant_label"], categories=["Full", "Attn-Only", "MLP-Only"])

sns.barplot(data=ppl_plot, x="variant_label", y="ppl", hue="method",
            palette=PALETTE, ax=ax, edgecolor="black", linewidth=0.5)

bl_ppl = df.loc[df["method"] == "Baseline (FP16)", "ppl"].values[0]
ax.axhline(bl_ppl, color=PALETTE["Baseline (FP16)"], ls="--", lw=2, label=f"Baseline FP16 ({bl_ppl:.2f})")

ax.set_ylabel("Perplexity (WikiText-2, ↓ is better)")
ax.set_xlabel("Quantization Scope")
ax.set_title("WikiText-2 Perplexity by Method × Scope\n(Qwen2-1.5B — note: eval configs vary across methods)", fontsize=13)
ax.legend(title="Method", loc="upper right", fontsize=9)

for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTDIR / "fig2_perplexity_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig2_perplexity_comparison.pdf", bbox_inches="tight")
plt.close()
print("Saved fig2_perplexity_comparison")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  PLOT: Compression–Accuracy Pareto Front
# ═════════════════════════════════════════════════════════════════════════════
pareto_df = df.dropna(subset=["model_size_gb", "arc_pct"]).copy()

fig, ax = plt.subplots(figsize=(10, 7))
for method, grp in pareto_df.groupby("method"):
    ax.scatter(grp["model_size_gb"], grp["arc_pct"], s=120,
               color=PALETTE.get(method, "#333"), label=method, edgecolors="black", zorder=5)
    for _, r in grp.iterrows():
        ax.annotate(VARIANT_LABELS.get(r["variant"], r["variant"]),
                    (r["model_size_gb"], r["arc_pct"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)

ax.set_xlabel("Model Size (GB) — ← more compressed")
ax.set_ylabel("ARC-Challenge Accuracy (%)")
ax.set_title("Compression vs. Accuracy Trade-off\n(Pareto front: upper-left is ideal)", fontsize=13)
ax.legend(title="Method", fontsize=9)
ax.invert_xaxis()
plt.tight_layout()
fig.savefig(OUTDIR / "fig3_pareto_compression_accuracy.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig3_pareto_compression_accuracy.pdf", bbox_inches="tight")
plt.close()
print("Saved fig3_pareto_compression_accuracy")

# ═════════════════════════════════════════════════════════════════════════════
# 6.  PLOT: Variant Sensitivity — delta from Full for each method
# ═════════════════════════════════════════════════════════════════════════════
methods_for_delta = ["RTN (W8A16)", "GPTQ (W4)", "SmoothQuant (W8A8)", "AWQ (W4A16)"]
delta_records = []
for m in methods_for_delta:
    mdf = df[df["method"] == m]
    full_arc = mdf.loc[mdf["variant"] == "full_quant", "arc_pct"].values
    if len(full_arc) == 0:
        continue
    full_arc = full_arc[0]
    for _, r in mdf.iterrows():
        if r["variant"] != "full_quant" and r["arc_pct"] is not None and full_arc is not None:
            delta_records.append({
                "method": m,
                "variant_label": VARIANT_LABELS.get(r["variant"], r["variant"]),
                "delta_arc": r["arc_pct"] - full_arc,
            })

delta_df = pd.DataFrame(delta_records)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=delta_df, x="method", y="delta_arc", hue="variant_label",
            palette=["#5B9BD5", "#ED7D31"], ax=ax, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Δ ARC Accuracy vs. Full Quant (pp)")
ax.set_xlabel("")
ax.set_title("Selective Quantization Gain Over Full Quantization\n(positive = better than full-model quant)", fontsize=13)
ax.legend(title="Scope")

for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if abs(h) > 0.01:
            va = "bottom" if h >= 0 else "top"
            offset = 0.1 if h >= 0 else -0.1
            ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:+.1f}",
                    ha="center", va=va, fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTDIR / "fig4_variant_sensitivity.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig4_variant_sensitivity.pdf", bbox_inches="tight")
plt.close()
print("Saved fig4_variant_sensitivity")

# ═════════════════════════════════════════════════════════════════════════════
# 7.  PLOT: GSM8K Math Reasoning (RTN + AWQ + Baseline)
# ═════════════════════════════════════════════════════════════════════════════
gsm_df = df.dropna(subset=["gsm8k_pct"]).copy()
gsm_df["variant_label"] = pd.Categorical(gsm_df["variant_label"], categories=["Full", "Attn-Only", "MLP-Only"])

fig, ax = plt.subplots(figsize=(10, 5))
gsm_plot = gsm_df[gsm_df["method"] != "Baseline (FP16)"]
sns.barplot(data=gsm_plot, x="variant_label", y="gsm8k_pct", hue="method",
            palette=PALETTE, ax=ax, edgecolor="black", linewidth=0.5)

bl_gsm = gsm_df.loc[gsm_df["method"] == "Baseline (FP16)", "gsm8k_pct"].values
if len(bl_gsm) > 0:
    ax.axhline(bl_gsm[0], color=PALETTE["Baseline (FP16)"], ls="--", lw=2,
               label=f"Baseline FP16 ({bl_gsm[0]:.0f}%)")

ax.set_ylabel("GSM8K Accuracy (%, 8-shot CoT)")
ax.set_xlabel("Quantization Scope")
ax.set_title("GSM8K Math Reasoning: RTN vs AWQ\n(8-shot chain-of-thought, Qwen2-1.5B)", fontsize=13)
ax.set_ylim(50, 65)
ax.legend(title="Method", fontsize=9)

for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTDIR / "fig5_gsm8k_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig5_gsm8k_comparison.pdf", bbox_inches="tight")
plt.close()
print("Saved fig5_gsm8k_comparison")

# ═════════════════════════════════════════════════════════════════════════════
# 8.  DEEP DIVE: RTN Logit-Level Diagnostics
# ═════════════════════════════════════════════════════════════════════════════
diag = load_json(ROOT / "rtn-eval/rtn_logit_diagnostics.json")
diag_records = []
for entry in diag:
    diag_records.append({
        "variant": VARIANT_LABELS.get(entry["variant"], entry["variant"]),
        "KL Divergence": entry["kl_div"],
        "Cosine Similarity": entry["cosine_sim"],
        "Top-1 Agreement (%)": entry["top1_agreement"] * 100,
        "Top-5 Agreement (%)": entry["top5_agreement"] * 100,
    })
diag_df = pd.DataFrame(diag_records)
print("\n" + "=" * 80)
print("TABLE 2: RTN Logit Diagnostics (output fidelity vs FP16)")
print("=" * 80)
print(diag_df.to_string(index=False, float_format="%.4f"))
diag_df.to_csv(OUTDIR / "table2_rtn_logit_diagnostics.csv", index=False)

# Radar chart for RTN diagnostics
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
categories = ["1 − KL Div\n(×100)", "Cosine\nSimilarity", "Top-1\nAgreement", "Top-5\nAgreement"]
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

colors = ["#4C72B0", "#DD8452", "#55A868"]
for i, (_, row) in enumerate(diag_df.iterrows()):
    vals = [
        (1 - row["KL Divergence"]) * 100,
        row["Cosine Similarity"] * 100,
        row["Top-1 Agreement (%)"],
        row["Top-5 Agreement (%)"],
    ]
    vals += vals[:1]
    ax = axes[i]
    ax.plot(angles, vals, "o-", color=colors[i], linewidth=2)
    ax.fill(angles, vals, alpha=0.15, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(94, 100)
    ax.set_title(row["variant"], fontsize=12, fontweight="bold", pad=15)

fig.suptitle("RTN (W8A16) — Output Fidelity Radar\n(closer to 100 = closer to FP16 baseline)", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUTDIR / "fig6_rtn_logit_radar.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig6_rtn_logit_radar.pdf", bbox_inches="tight")
plt.close()
print("Saved fig6_rtn_logit_radar")

# ═════════════════════════════════════════════════════════════════════════════
# 9.  DEEP DIVE: RTN Layer-wise MSE Heatmap
# ═════════════════════════════════════════════════════════════════════════════
layer_df = pd.read_parquet(ROOT / "rtn-eval/rtn_layer_stats.parquet")
print(f"\nLayer stats columns: {list(layer_df.columns)}")
print(f"Layer stats shape: {layer_df.shape}")
print(layer_df.head(3))

# Normalize variant names in layer_df
layer_df["variant_norm"] = layer_df["variant"].map(lambda v: {"attn_only_quant": "attn_only", "mlp_only_quant": "mlp_only"}.get(v, v))

# Pivot for heatmap: layers × variants (aggregate MSE per layer)
if "variant" in layer_df.columns and "layer_idx" in layer_df.columns and "mse" in layer_df.columns:
    # Aggregate MSE per layer (mean across module families)
    heatmap_data = layer_df.pivot_table(index="layer_idx", columns="variant_norm", values="mse", aggfunc="mean")
    heatmap_data.columns = [VARIANT_LABELS.get(c, c) for c in heatmap_data.columns]
    heatmap_data = heatmap_data.sort_index()

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, annot=False, fmt=".4f",
                cbar_kws={"label": "Mean MSE (quantized vs FP16)"})
    ax.set_ylabel("Transformer Layer")
    ax.set_xlabel("Quantization Scope")
    ax.set_title("RTN (W8A16) — Per-Layer Mean MSE Heatmap\n(higher = more quantization error)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig7_rtn_layer_mse_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig7_rtn_layer_mse_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig7_rtn_layer_mse_heatmap")

    # Line plot of layer MSE
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in heatmap_data.columns:
        ax.plot(heatmap_data.index, heatmap_data[col], marker="o", markersize=3, label=col)
    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Mean MSE")
    ax.set_title("RTN (W8A16) — Layer-wise Quantization Error\n(spikes indicate sensitive layers — consistent with Dettmers et al., 2022)", fontsize=13)
    ax.legend(title="Scope")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig8_rtn_layer_mse_line.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig8_rtn_layer_mse_line.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig8_rtn_layer_mse_line")

    # Attn vs MLP module family MSE comparison
    if "module_family" in layer_df.columns:
        fam_df = layer_df.pivot_table(index="layer_idx", columns="module_family", values="mse", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in fam_df.columns:
            ax.plot(fam_df.index, fam_df[col], marker=".", markersize=4, label=col, alpha=0.8)
        ax.set_xlabel("Transformer Layer Index")
        ax.set_ylabel("Mean MSE")
        ax.set_title("RTN — Attention vs MLP Module MSE by Layer\n(identifies which component is harder to quantize per layer)", fontsize=13)
        ax.legend(title="Module Family")
        plt.tight_layout()
        fig.savefig(OUTDIR / "fig8b_rtn_attn_vs_mlp_mse.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUTDIR / "fig8b_rtn_attn_vs_mlp_mse.pdf", bbox_inches="tight")
        plt.close()
        print("Saved fig8b_rtn_attn_vs_mlp_mse")

# Activation outlier analysis (columns: p99, act_max)
if "p99" in layer_df.columns and "act_max" in layer_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # p99 vs max ratio → outlier severity
    layer_df["outlier_ratio"] = layer_df["act_max"] / layer_df["p99"].clip(lower=1e-8)

    # Aggregate per layer per variant
    act_agg = layer_df.groupby(["variant_norm", "layer_idx"]).agg(
        p99_mean=("p99", "mean"), outlier_max=("outlier_ratio", "max")
    ).reset_index()

    for var, grp in act_agg.groupby("variant_norm"):
        lbl = VARIANT_LABELS.get(var, var)
        axes[0].plot(grp["layer_idx"], grp["p99_mean"], marker=".", label=f"{lbl}")
        axes[1].plot(grp["layer_idx"], grp["outlier_max"], marker=".", label=lbl)

    axes[0].set_title("Activation p99 Magnitude by Layer")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean Activation p99")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Outlier Ratio (max / p99) by Layer\n(high = extreme outliers — motivates SmoothQuant)")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("max / p99")
    axes[1].legend(fontsize=8)

    fig.suptitle("RTN — Activation Distribution Analysis\n(Dettmers et al. 2022: outlier features drive quantization error)", fontsize=13, y=1.05)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig9_rtn_activation_outliers.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig9_rtn_activation_outliers.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig9_rtn_activation_outliers")

# ═════════════════════════════════════════════════════════════════════════════
# 10. DEEP DIVE: AWQ Throughput & Latency Analysis
# ═════════════════════════════════════════════════════════════════════════════
awq_df = df[df["method"].isin(["AWQ (W4A16)", "Baseline (FP16)"])].dropna(subset=["throughput_tps"]).copy()
if len(awq_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Throughput
    awq_df_sorted = awq_df.sort_values("throughput_tps", ascending=True)
    labels = awq_df_sorted.apply(lambda r: f"{r['method']}\n{VARIANT_LABELS.get(r['variant'], r['variant'])}", axis=1)
    colors = [PALETTE.get(r["method"], "#333") for _, r in awq_df_sorted.iterrows()]
    axes[0].barh(labels, awq_df_sorted["throughput_tps"], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Throughput (tok/s)")
    axes[0].set_title("Inference Throughput")
    for i, v in enumerate(awq_df_sorted["throughput_tps"]):
        axes[0].text(v + 1, i, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    # Model size vs throughput scatter
    for method, grp in awq_df.groupby("method"):
        axes[1].scatter(grp["model_size_gb"], grp["throughput_tps"], s=100,
                       color=PALETTE.get(method, "#333"), label=method, edgecolors="black", zorder=5)
        for _, r in grp.iterrows():
            axes[1].annotate(VARIANT_LABELS.get(r["variant"], r["variant"]),
                            (r["model_size_gb"], r["throughput_tps"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[1].set_xlabel("Model Size (GB)")
    axes[1].set_ylabel("Throughput (tok/s)")
    axes[1].set_title("Size vs. Throughput")
    axes[1].legend(fontsize=9)

    fig.suptitle("AWQ (W4A16) — Deployment Efficiency Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig10_awq_throughput.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig10_awq_throughput.pdf", bbox_inches="tight")
    plt.close()
    print("Saved fig10_awq_throughput")

# ═════════════════════════════════════════════════════════════════════════════
# 11. DEEP DIVE: SmoothQuant — W8A8 Quality Preservation Heatmap
# ═════════════════════════════════════════════════════════════════════════════
sq_records = []
for var in ["full_quant", "attn_only", "mlp_only"]:
    row = df[(df["method"] == "SmoothQuant (W8A8)") & (df["variant"] == var)]
    if len(row) > 0:
        r = row.iloc[0]
        sq_records.append({
            "Variant": VARIANT_LABELS[var],
            "PPL": r["ppl"],
            "ARC (%)": r["arc_pct"],
            "Size (GB)": r["model_size_gb"],
            "VRAM (GB)": r["peak_vram_gb"],
        })
sq_summary = pd.DataFrame(sq_records)
print("\n" + "=" * 80)
print("TABLE 3: SmoothQuant (W8A8) Summary")
print("=" * 80)
print(sq_summary.to_string(index=False, float_format="%.3f"))
sq_summary.to_csv(OUTDIR / "table3_smoothquant_summary.csv", index=False)

# ═════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY: Method Recommendation Heatmap
# ═════════════════════════════════════════════════════════════════════════════
# Normalize metrics to 0-1 scale for a heatmap-style comparison
full_df = df[df["variant"] == "full_quant"].copy()
full_df = full_df[full_df["method"] != "Baseline (FP16)"]

metrics_for_heatmap = []
for _, r in full_df.iterrows():
    metrics_for_heatmap.append({
        "Method": r["method"],
        "ARC Accuracy": r["arc_pct"] if r["arc_pct"] else 0,
        "Perplexity\n(inverted)": -r["ppl"] if r["ppl"] else 0,  # negative so higher is better
        "Model Size\n(inverted)": -(r["model_size_gb"] if r["model_size_gb"] else 0),
    })
hm_df = pd.DataFrame(metrics_for_heatmap).set_index("Method")

# Min-max normalize each column
hm_norm = (hm_df - hm_df.min()) / (hm_df.max() - hm_df.min() + 1e-8)

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(hm_norm, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            vmin=0, vmax=1, linewidths=1, cbar_kws={"label": "Normalized Score (1 = best)"})
ax.set_title("Full-Model Quantization — Normalized Method Comparison\n(1.0 = best in category)", fontsize=13)
plt.tight_layout()
fig.savefig(OUTDIR / "fig11_method_heatmap.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig11_method_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Saved fig11_method_heatmap")

# ═════════════════════════════════════════════════════════════════════════════
# 13. COMPREHENSIVE SUMMARY TABLE — All results
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL COMPREHENSIVE TABLE")
print("=" * 80)
final = df[["method", "variant_label", "precision", "ppl", "arc_pct", "gsm8k_pct",
            "model_size_gb", "peak_vram_gb", "throughput_tps", "latency_ms"]].copy()
final.columns = ["Method", "Scope", "Precision", "PPL↓", "ARC%↑", "GSM8K%↑",
                  "Size(GB)", "VRAM(GB)", "Throughput", "Latency(ms)"]
print(final.to_string(index=False, float_format="%.2f", na_rep="—"))
final.to_csv(OUTDIR / "table_comprehensive.csv", index=False, float_format="%.4f")

print("\n\nAll outputs saved to:", OUTDIR)
print("Files generated:")
for f in sorted(OUTDIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
