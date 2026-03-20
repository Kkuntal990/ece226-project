#!/usr/bin/env python3
"""Compression vs Accuracy trade-off across multiple benchmarks (ARC, PPL, Throughput)."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

OUTDIR = Path("/tmp/ece226_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Load updated CSV ──
df = pd.read_csv("/tmp/ece226_results/analysis/table_comprehensive.csv")

# Color by method
METHOD_COLORS = {
    "RTN (W8A16)": "#4C72B0",
    "GPTQ (W4)": "#DD8452",
    "SmoothQuant (W8A8)": "#55A868",
    "AWQ (W4A16)": "#C44E52",
    "Baseline (FP16)": "#8C8C8C",
}

# Shape by scope
SCOPE_MARKERS = {
    "Full": "o",       # circle
    "Attn-Only": "s",  # square
    "MLP-Only": "D",   # diamond
}

# ═══════════════════════════════════════════════════════════════════════════
# Plot: Model Size vs multiple metrics (ARC, PPL inverted, Throughput)
# Each metric gets a different marker edge style; scopes get shapes; methods get colors
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# ── Panel 1: Size vs ARC-Challenge Accuracy ──
ax = axes[0]
for _, r in df.iterrows():
    if pd.isna(r["ARC%↑"]) or pd.isna(r["Size(GB)"]): continue
    ax.scatter(r["Size(GB)"], r["ARC%↑"], s=140, zorder=5,
               color=METHOD_COLORS.get(r["Method"], "#333"),
               marker=SCOPE_MARKERS.get(r["Scope"], "o"),
               edgecolors="black", linewidth=0.8)
    ax.annotate(r["Scope"][:4], (r["Size(GB)"], r["ARC%↑"]),
                textcoords="offset points", xytext=(7, 5), fontsize=7)

ax.set_xlabel("Model Size (GB)")
ax.set_ylabel("ARC-Challenge Accuracy (%)")
ax.set_title("Size vs. ARC-Challenge (↑)", fontsize=13)

# ── Panel 2: Size vs Perplexity (inverted — lower PPL is better, so flip y) ──
ax = axes[1]
for _, r in df.iterrows():
    if pd.isna(r["PPL↓"]) or pd.isna(r["Size(GB)"]): continue
    ax.scatter(r["Size(GB)"], r["PPL↓"], s=140, zorder=5,
               color=METHOD_COLORS.get(r["Method"], "#333"),
               marker=SCOPE_MARKERS.get(r["Scope"], "o"),
               edgecolors="black", linewidth=0.8)
    ax.annotate(r["Scope"][:4], (r["Size(GB)"], r["PPL↓"]),
                textcoords="offset points", xytext=(7, 5), fontsize=7)

ax.set_xlabel("Model Size (GB)")
ax.set_ylabel("Perplexity (WikiText-2)")
ax.set_title("Size vs. Perplexity (↓)", fontsize=13)
ax.invert_yaxis()  # lower PPL = better = higher on chart

# ── Panel 3: Size vs Throughput ──
ax = axes[2]
for _, r in df.iterrows():
    if pd.isna(r["Throughput"]) or pd.isna(r["Size(GB)"]): continue
    ax.scatter(r["Size(GB)"], r["Throughput"], s=140, zorder=5,
               color=METHOD_COLORS.get(r["Method"], "#333"),
               marker=SCOPE_MARKERS.get(r["Scope"], "o"),
               edgecolors="black", linewidth=0.8)
    ax.annotate(r["Scope"][:4], (r["Size(GB)"], r["Throughput"]),
                textcoords="offset points", xytext=(7, 5), fontsize=7)

ax.set_xlabel("Model Size (GB)")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Size vs. Throughput (↑)", fontsize=13)

# ── Shared legend ──
method_handles = [mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                   markersize=10, markeredgecolor="black", label=m)
                  for m, c in METHOD_COLORS.items()]
scope_handles = [mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                  markersize=10, markeredgecolor="black", label=s)
                 for s, mk in SCOPE_MARKERS.items()]

fig.legend(handles=method_handles + scope_handles,
           loc="lower center", ncol=7, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle("Compression vs. Quality Trade-off Across Benchmarks\n"
             "(Qwen2-1.5B — lower size with higher metric is ideal)",
             fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig(OUTDIR / "fig12_pareto_multidataset.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig12_pareto_multidataset.pdf", bbox_inches="tight")
plt.close()
print("Saved fig12_pareto_multidataset")

# ═══════════════════════════════════════════════════════════════════════════
# Single combined scatter: Size (x) vs all metrics overlaid with different shapes
# ═══════════════════════════════════════════════════════════════════════════
METRIC_MARKERS = {
    "ARC-Challenge (%)": "o",     # circle
    "Perplexity (inv.)": "^",     # triangle up
    "Throughput (tok/s)": "P",    # plus (filled)
}

fig, ax = plt.subplots(figsize=(12, 8))

# Normalize each metric to 0–100 for overlay
def minmax(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-8) * 100

arc_norm = minmax(df["ARC%↑"].dropna())
ppl_norm = minmax(-df["PPL↓"].dropna())  # negate so lower PPL = higher score
thr_norm = minmax(df["Throughput"].dropna())

for idx, r in df.iterrows():
    size = r["Size(GB)"]
    if pd.isna(size): continue
    method_color = METHOD_COLORS.get(r["Method"], "#333")
    scope_marker_base = SCOPE_MARKERS.get(r["Scope"], "o")

    # ARC
    if not pd.isna(r["ARC%↑"]):
        val = (r["ARC%↑"] - df["ARC%↑"].min()) / (df["ARC%↑"].max() - df["ARC%↑"].min() + 1e-8) * 100
        ax.scatter(size, val, s=130, color=method_color, marker="o",
                   edgecolors="black", linewidth=0.7, zorder=5, alpha=0.9)

    # PPL (inverted)
    if not pd.isna(r["PPL↓"]):
        val = (-r["PPL↓"] - (-df["PPL↓"].max())) / ((-df["PPL↓"].min()) - (-df["PPL↓"].max()) + 1e-8) * 100
        ax.scatter(size, val, s=130, color=method_color, marker="^",
                   edgecolors="black", linewidth=0.7, zorder=5, alpha=0.9)

    # Throughput
    if not pd.isna(r["Throughput"]):
        val = (r["Throughput"] - df["Throughput"].min()) / (df["Throughput"].max() - df["Throughput"].min() + 1e-8) * 100
        ax.scatter(size, val, s=130, color=method_color, marker="P",
                   edgecolors="black", linewidth=0.7, zorder=5, alpha=0.9)

ax.set_xlabel("Model Size (GB) — → more compressed", fontsize=12)
ax.set_ylabel("Normalized Score (0–100, ↑ better)", fontsize=12)
ax.set_title("Unified Compression vs. Quality: All Benchmarks\n"
             "(each metric normalized to 0–100; higher = better)", fontsize=14)

# Legends
metric_handles = [mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                   markersize=10, markeredgecolor="black", label=m)
                  for m, mk in METRIC_MARKERS.items()]
method_handles = [mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                   markersize=10, markeredgecolor="black", label=m)
                  for m, c in METHOD_COLORS.items()]

legend1 = ax.legend(handles=metric_handles, title="Benchmark", loc="lower left", fontsize=9)
ax.add_artist(legend1)
ax.legend(handles=method_handles, title="Method", loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig(OUTDIR / "fig13_unified_pareto.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTDIR / "fig13_unified_pareto.pdf", bbox_inches="tight")
plt.close()
print("Saved fig13_unified_pareto")
