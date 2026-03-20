#!/usr/bin/env python3
"""
Literature-informed plots for PTQ comparative analysis.
Based on conventions from GPTQ, AWQ, SmoothQuant, ParetoQ, and survey papers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

OUTDIR = Path("/tmp/ece226_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)

df = pd.read_csv("/tmp/ece226_results/analysis/table_comprehensive.csv")

METHOD_COLORS = {
    "RTN (W8A16)": "#4C72B0", "GPTQ (W4)": "#DD8452",
    "SmoothQuant (W8A8)": "#55A868", "AWQ (W4A16)": "#C44E52",
    "Baseline (FP16)": "#8C8C8C",
}
SCOPE_MARKERS = {"Full": "o", "Attn-Only": "s", "MLP-Only": "D"}
SCOPE_ORDER = ["Full", "Attn-Only", "MLP-Only"]
METHODS_ONLY = [m for m in METHOD_COLORS if m != "Baseline (FP16)"]

quant_df = df[df["Method"] != "Baseline (FP16)"].copy()
quant_df["Scope"] = pd.Categorical(quant_df["Scope"], categories=SCOPE_ORDER)
bl = df[df["Method"] == "Baseline (FP16)"].iloc[0]

# ═══════════════════════════════════════════════════════════════════════════
# FIG 14: Bubble Chart — Size vs ARC, bubble = throughput
# (Literature: ParetoQ, Microscaling Formats paper)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 8))

for _, r in df.iterrows():
    if pd.isna(r["Size(GB)"]) or pd.isna(r["ARC%↑"]): continue
    thr = r["Throughput"] if not pd.isna(r["Throughput"]) else 10
    bubble_size = (thr / df["Throughput"].max()) * 800 + 40
    ax.scatter(r["Size(GB)"], r["ARC%↑"], s=bubble_size, alpha=0.7,
               color=METHOD_COLORS.get(r["Method"], "#333"),
               marker=SCOPE_MARKERS.get(r["Scope"], "o"),
               edgecolors="black", linewidth=0.8, zorder=5)
    ax.annotate(f'{r["Scope"][:4]}\n{thr:.0f} t/s',
                (r["Size(GB)"], r["ARC%↑"]),
                textcoords="offset points", xytext=(10, -5), fontsize=7,
                color="#444")

# Legend: methods
method_handles = [mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                   markersize=10, markeredgecolor="black", label=m)
                  for m, c in METHOD_COLORS.items()]
scope_handles = [mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                  markersize=10, markeredgecolor="black", label=s)
                 for s, mk in SCOPE_MARKERS.items()]
# Bubble size legend
for thr_val, label in [(20, "20 tok/s"), (90, "90 tok/s"), (189, "189 tok/s")]:
    sz = (thr_val / df["Throughput"].max()) * 800 + 40
    scope_handles.append(mlines.Line2D([], [], color="lightgray", marker="o",
                         linestyle="None", markersize=np.sqrt(sz)/2.5,
                         markeredgecolor="black", label=label))

ax.legend(handles=method_handles + scope_handles, loc="lower right",
          fontsize=8, ncol=2, title="Method / Scope / Throughput")
ax.set_xlabel("Model Size (GB) — ← more compressed", fontsize=12)
ax.set_ylabel("ARC-Challenge Accuracy (%)", fontsize=12)
ax.set_title("Compression–Accuracy–Throughput Trade-off\n"
             "(bubble size ∝ throughput; upper-left is ideal)", fontsize=14)
ax.invert_xaxis()
plt.tight_layout()
fig.savefig(OUTDIR / "fig14_bubble_pareto.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig14_bubble_pareto.pdf", bbox_inches="tight")
plt.close()
print("Saved fig14_bubble_pareto")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 15: Throughput Grouped Bar (all methods)
# (Literature: AWQ Figure 5, Jarvislabs guide)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=quant_df, x="Scope", y="Throughput", hue="Method",
            palette=METHOD_COLORS, ax=ax, edgecolor="black", linewidth=0.5,
            order=SCOPE_ORDER)
ax.axhline(bl["Throughput"], color=METHOD_COLORS["Baseline (FP16)"],
           ls="--", lw=2, label=f'Baseline FP16 ({bl["Throughput"]:.1f} tok/s)')
ax.set_ylabel("Throughput (tok/s, ↑ better)")
ax.set_xlabel("Quantization Scope")
ax.set_title("Inference Throughput by Method × Scope\n"
             "(AWQ achieves 3.3× baseline; SmoothQuant incurs overhead)", fontsize=13)
ax.legend(title="Method", fontsize=9, loc="upper right")
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR / "fig15_throughput_all.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig15_throughput_all.pdf", bbox_inches="tight")
plt.close()
print("Saved fig15_throughput_all")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 16: Latency Grouped Bar (all methods)
# (Literature: SmoothQuant Figure 8, Table 8)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=quant_df, x="Scope", y="Latency(ms)", hue="Method",
            palette=METHOD_COLORS, ax=ax, edgecolor="black", linewidth=0.5,
            order=SCOPE_ORDER)
ax.axhline(bl["Latency(ms)"], color=METHOD_COLORS["Baseline (FP16)"],
           ls="--", lw=2, label=f'Baseline FP16 ({bl["Latency(ms)"]:.1f} ms)')
ax.set_ylabel("Latency (ms/token, ↓ better)")
ax.set_xlabel("Quantization Scope")
ax.set_title("Per-Token Latency by Method × Scope\n"
             "(AWQ lowest latency; RTN/SmoothQuant higher due to mixed-precision overhead)", fontsize=13)
ax.legend(title="Method", fontsize=9, loc="upper right")
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR / "fig16_latency_all.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig16_latency_all.pdf", bbox_inches="tight")
plt.close()
print("Saved fig16_latency_all")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 17: VRAM Grouped Bar (all methods)
# (Literature: SmoothQuant Figure 8, Red Hat evaluation)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=quant_df, x="Scope", y="VRAM(GB)", hue="Method",
            palette=METHOD_COLORS, ax=ax, edgecolor="black", linewidth=0.5,
            order=SCOPE_ORDER)
ax.axhline(bl["VRAM(GB)"], color=METHOD_COLORS["Baseline (FP16)"],
           ls="--", lw=2, label=f'Baseline FP16 ({bl["VRAM(GB)"]:.1f} GB)')
ax.set_ylabel("Peak VRAM (GB, ↓ better)")
ax.set_xlabel("Quantization Scope")
ax.set_title("Peak VRAM Usage by Method × Scope\n"
             "(RTN/SmoothQuant achieve lowest VRAM; GPTQ Attn-Only anomalously high)", fontsize=13)
ax.legend(title="Method", fontsize=9, loc="upper right")
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR / "fig17_vram_all.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig17_vram_all.pdf", bbox_inches="tight")
plt.close()
print("Saved fig17_vram_all")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 18: Slope/Bump Chart — ARC Accuracy Across Scopes
# (Literature: ACL Findings 2024 cross-condition plots)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for metric, ylabel, ax, invert in [
    ("ARC%↑", "ARC-Challenge (%)", axes[0], False),
    ("PPL↓", "Perplexity (WikiText-2)", axes[1], True),
]:
    for method in METHODS_ONLY:
        mdf = quant_df[quant_df["Method"] == method].sort_values("Scope")
        vals = mdf[metric].values
        scopes = mdf["Scope"].values
        ax.plot(scopes, vals, "o-", color=METHOD_COLORS[method],
                linewidth=2.5, markersize=10, label=method, markeredgecolor="black")
        # Annotate values
        for s, v in zip(scopes, vals):
            if not pd.isna(v):
                offset = 0.3 if not invert else -0.15
                ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points",
                           xytext=(0, 10), fontsize=8, ha="center", fontweight="bold",
                           color=METHOD_COLORS[method])

    # Baseline reference
    bl_val = bl[metric]
    if not pd.isna(bl_val):
        ax.axhline(bl_val, color=METHOD_COLORS["Baseline (FP16)"],
                   ls="--", lw=1.5, alpha=0.7, label=f"Baseline ({bl_val:.1f})")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Quantization Scope")
    ax.legend(fontsize=8, loc="best")
    if invert:
        ax.invert_yaxis()

axes[0].set_title("ARC-Challenge Accuracy by Scope\n(flat = robust to scope selection)", fontsize=13)
axes[1].set_title("Perplexity by Scope\n(flat = robust to scope selection)", fontsize=13)

fig.suptitle("Scope Sensitivity Analysis — How Much Does Selective Quantization Help?\n"
             "(steeper slope = greater sensitivity to which layers are quantized)",
             fontsize=14, y=1.03)
plt.tight_layout()
fig.savefig(OUTDIR / "fig18_slope_sensitivity.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig18_slope_sensitivity.pdf", bbox_inches="tight")
plt.close()
print("Saved fig18_slope_sensitivity")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 19: Full Multi-Metric Heatmap — Methods × Scopes
# (Literature: Jarvislabs "combined benchmark summary heatmap", Microscaling)
# ═══════════════════════════════════════════════════════════════════════════
metrics = {
    "ARC-C (%)": ("ARC%↑", True),        # higher is better
    "PPL": ("PPL↓", False),               # lower is better
    "Throughput\n(tok/s)": ("Throughput", True),
    "Latency\n(ms/tok)": ("Latency(ms)", False),
    "VRAM (GB)": ("VRAM(GB)", False),     # lower is better
    "Size (GB)": ("Size(GB)", False),     # lower is better
}

# Build normalized heatmap data
rows = []
for _, r in quant_df.iterrows():
    label = f"{r['Method']}\n{r['Scope']}"
    row = {"Config": label}
    for display_name, (col, higher_better) in metrics.items():
        val = r[col]
        if pd.isna(val):
            row[display_name] = np.nan
            continue
        col_vals = quant_df[col].dropna()
        mn, mx = col_vals.min(), col_vals.max()
        norm = (val - mn) / (mx - mn + 1e-8)
        if not higher_better:
            norm = 1 - norm  # invert so 1 = best
        row[display_name] = norm
    rows.append(row)

hm = pd.DataFrame(rows).set_index("Config")

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            vmin=0, vmax=1, linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Normalized Score (1 = best)", "shrink": 0.6})
ax.set_title("Multi-Metric Quantization Scorecard\n"
             "(all metrics normalized: 1.0 = best across all configs)", fontsize=14)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(OUTDIR / "fig19_full_heatmap.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig19_full_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Saved fig19_full_heatmap")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 20: Compression Ratio Bar Chart (ordered)
# (Literature: GPTQ Table 6, AWQ Table 4)
# ═══════════════════════════════════════════════════════════════════════════
quant_df_sorted = quant_df.dropna(subset=["Size(GB)"]).copy()
quant_df_sorted["Compression Ratio"] = bl["Size(GB)"] / quant_df_sorted["Size(GB)"]
quant_df_sorted["Label"] = quant_df_sorted["Method"].astype(str) + "\n" + quant_df_sorted["Scope"].astype(str)
quant_df_sorted = quant_df_sorted.sort_values("Compression Ratio", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = [METHOD_COLORS.get(m, "#333") for m in quant_df_sorted["Method"]]
bars = ax.barh(quant_df_sorted["Label"], quant_df_sorted["Compression Ratio"],
               color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5, label="1× (no compression)")
for i, (cr, sz) in enumerate(zip(quant_df_sorted["Compression Ratio"],
                                   quant_df_sorted["Size(GB)"])):
    ax.text(cr + 0.02, i, f"{cr:.2f}× ({sz:.2f} GB)",
            va="center", fontsize=8, fontweight="bold")

ax.set_xlabel("Compression Ratio (baseline 2.88 GB / model size)", fontsize=11)
ax.set_title("Model Compression Ratio by Configuration\n"
             "(higher = more compressed; AWQ MLP-Only achieves 2.57×)", fontsize=13)
plt.tight_layout()
fig.savefig(OUTDIR / "fig20_compression_ratio.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig20_compression_ratio.pdf", bbox_inches="tight")
plt.close()
print("Saved fig20_compression_ratio")

print("\nAll new figures saved to:", OUTDIR)
for f in sorted(OUTDIR.glob("fig1[4-9]*.png")) | sorted(OUTDIR.glob("fig20*.png")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
