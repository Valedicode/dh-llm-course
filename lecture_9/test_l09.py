"""Test L09 visualization: runs the core chart cells independently."""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("outputs") / "L09"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate demo data (mirroring code-8 fallback)
np.random.seed(42)
n = 20
scores = np.round(np.random.uniform(0.5, 12.0, n), 2).tolist()

demo_terms = ["刻本", "缮写", "金石学", "校勘", "传世本", "拓片", "目录学", "甲骨文"]
term_hits = [bool(x) for x in np.random.choice([0, 1], size=len(demo_terms), p=[0.5, 0.5])]

data = {
    "_metadata": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "num_samples": n},
    "bleu": {
        "individual_scores": scores,
        "mean": round(float(np.mean(scores)), 2),
    },
    "term_accuracy": {
        "accuracy_pct": round(np.mean(term_hits) * 100, 1),
        "total_hits": int(np.sum(term_hits)),
        "total_terms": len(term_hits),
    },
    "samples": [{"index": i, "bleu": scores[i]} for i in range(n)],
}

bleu = data["bleu"]
term_acc = data["term_accuracy"]
samples = data["samples"]

samples_sorted = sorted(samples, key=lambda x: x["bleu"])
bleu_scores = [s["bleu"] for s in samples_sorted]
labels = [f"#{s['index']}" for s in samples_sorted]

# ---- Chart 1: Basic BLEU line chart ----
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(bleu_scores))
ax.plot(x, bleu_scores, "o-", color="steelblue", linewidth=1.5, markersize=5)
ax.axhline(y=bleu["mean"], color="red", linestyle="--", linewidth=1, label=f'Mean = {bleu["mean"]:.2f}')
ax.set_xlabel("Sample Index"); ax.set_ylabel("BLEU Score")
ax.set_title("BLEU Scores Across Samples"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "bleu_line_basic.png", dpi=150); plt.close()
print(f"[OK] bleu_line_basic.png ({os.path.getsize(OUT_DIR / 'bleu_line_basic.png') // 1024} KB)")

# ---- Chart 2: Basic term bar chart ----
fig, ax = plt.subplots(figsize=(6, 4.5))
hits, total = term_acc["total_hits"], term_acc["total_terms"]
misses = total - hits
ax.bar(["Hit", "Miss"], [hits, misses], color=["steelblue", "lightgray"], edgecolor="white", width=0.5)
ax.set_ylabel("Term Count"); ax.set_title("Term Accuracy")
ax.set_ylim(0, total * 1.3)
for i, v in enumerate([hits, misses]):
    ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold", fontsize=12)
plt.tight_layout(); plt.savefig(OUT_DIR / "term_bar_basic.png", dpi=150); plt.close()
print(f"[OK] term_bar_basic.png ({os.path.getsize(OUT_DIR / 'term_bar_basic.png') // 1024} KB)")

# ---- Chart 3: Paper-ready BLEU line chart ----
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x, bleu_scores, "o-", color="#2B579A", linewidth=1.5, markersize=4,
        markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#2B579A",
        label="BLEU per sample", zorder=3)
ax.axhline(y=bleu["mean"], color="#D64531", linestyle="--", linewidth=1, alpha=0.8,
           label=f'Mean = {bleu["mean"]:.2f}', zorder=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5); ax.spines["bottom"].set_linewidth(0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
ax.set_xlabel("Sample (sorted by BLEU)"); ax.set_ylabel("BLEU Score")
ax.set_title("BLEU Scores Across Samples", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
plt.tight_layout(); plt.savefig(OUT_DIR / "bleu_line_paper.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"[OK] bleu_line_paper.png ({os.path.getsize(OUT_DIR / 'bleu_line_paper.png') // 1024} KB)")

# ---- Chart 4: Paper-ready term bar chart ----
fig, ax = plt.subplots(figsize=(5.5, 4))
pct = term_acc["accuracy_pct"]
bar_colors = ["#2B579A", "#D2D2D2"]
bars = ax.bar(["Hit", "Miss"], [hits, misses], color=bar_colors, edgecolor="white", width=0.5, linewidth=0.8)
for bar, v in zip(bars, [hits, misses]):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
            str(v), ha="center", fontweight="bold", fontsize=13, color="#333333")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5); ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(left=True, bottom=True, length=3, width=0.5)
ax.set_ylabel("Count"); ax.set_title(f"Term Accuracy: {pct:.1f}%", fontweight="bold")
ax.set_ylim(0, total * 1.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "term_bar_paper.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"[OK] term_bar_paper.png ({os.path.getsize(OUT_DIR / 'term_bar_paper.png') // 1024} KB)")

print("\nALL 4 CHARTS GENERATED")
