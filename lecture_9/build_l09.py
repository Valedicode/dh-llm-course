import json, os

cells = []

def md(id, source):
    cells.append({"cell_type": "markdown", "id": id, "metadata": {}, "source": source.split("\n")})

def code(id, source):
    cells.append({"cell_type": "code", "id": id, "metadata": {}, "outputs": [], "execution_count": None, "source": source.split("\n")})

# ===== Cover =====
md("md-0", """# 第9讲：MLLM评估结果可视化与论文应用
## 用折线图与柱状图呈现评估数据，适配论文学术图表规范

> **本讲定位**：可视化收尾讲——读取第8讲的评估数据，用最基础的折线图和柱状图呈现结果，并按论文图表规范优化样式。复用L08数据与代码框架，仅新增图表优化步骤。""")

# ===== 本讲定位 =====
md("md-1", """# 本讲定位

第8讲我们得到了 BLEU 分数和术语准确率——但面对一行行数字，读者很难直觉地判断模型表现。

本讲只做一件事：**把数字变成论文可用的图表**。

只聚焦两种最基础、最通用的图表：
- **折线图**：展示 BLEU 分数在不同样本上的变化趋势
- **柱状图**：对比术语准确率的命中/未命中

不展开复杂图表类型（热力图、雷达图等），确保你在论文/报告中能直接使用。""")

# ===== 本讲结构 =====
md("md-2", """# 本讲结构

## 1  环境准备
## 2  基础可视化技巧（理论）
## 3  载入 L08 评估结果
## 4  生成基础图表（折线图 + 柱状图）
## 5  可视化结果的优化与论文适配规范（理论）
## 6  微调图表样式适配论文排版（实操）
## 7  评估实验在论文中的简要呈现思路（理论）
## 快速回顾""")

# ===== 透明度说明 =====
md("md-3", """# 透明度说明

> **数据来源**：本讲直接复用第8讲保存的 `l08_evaluation_results.json`，不引入新数据。如果找不到该文件（例如跳过了L08），代码会自动生成演示数据以确保可视化流程可独立运行——但演示数据不代表真实模型表现。
>
> **工具链**：仅使用 `matplotlib`——Python 最成熟的可视化库，也是学术论文中最广泛使用的 Python 图表工具。不使用需要额外安装的 seaborn、plotly 等库。
>
> **设计原则**：本讲聚焦两种最基础的图表类型（折线图、柱状图），不展开复杂图表。所有代码复用 L08 的数据加载框架，仅新增图表生成与样式优化步骤。""")

# ===== Ch1: 环境准备 =====
md("md-4", """# 第1章  环境准备

- 只需 `matplotlib` 和 `numpy`，均为科学计算标配
- Colab 已预装这两个库，本地用户若使用 `mllm-dh` conda 环境也可直接运行

> **本地用户**：如果已在 conda 环境中运行，可注释掉 `!pip install` 行。""")

code("code-5", """# ===== 第1章：环境准备 =====
# Colab 用户：运行此单元格自动安装依赖。
# 本地 mllm-dh 环境用户：可注释掉 !pip 行。

import sys, os, subprocess
from pathlib import Path

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "numpy"], check=False)
    print("Colab 环境已就绪")
else:
    print("本地环境：请确保已安装 matplotlib, numpy")

import matplotlib.pyplot as plt
import numpy as np

# 中文字体设置（按优先级尝试）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

print(f"matplotlib {matplotlib.__version__}")
print(f"numpy {np.__version__}")

# 输出目录
LESSON_ID = "L09"
OUT_DIR = Path("outputs") / LESSON_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {OUT_DIR}")""")

# ===== Ch2: 基础可视化技巧（理论） =====
md("md-6", """# 第2章  基础可视化技巧

## 2.0 为什么是折线图 + 柱状图？

评估结果通常包括两类数据：
- **连续序列**（如逐样本 BLEU 分数）——适合用折线图展示趋势和波动
- **分类对比**（如术语命中 vs 未命中）——适合用柱状图展示数量差异

这两种图表是学术论文中使用频率最高的图表类型，也是审稿人最熟悉的表达方式。

## 2.1 折线图的适配场景

在评估场景中，折线图可以回答：
- BLEU 分数在各样本间如何变化？（波动大说明模型不稳定）
- 是否存在某些样本分数特别低？（定位问题样本）
- 整体趋势是上升还是下降？（用于对比微调前后）

> 折线图的优势在于"趋势可视化"——让读者一眼看到数据的变化模式。

## 2.2 柱状图的适配场景

柱状图更适合回答：
- 术语准确率有多高？（Hit vs Miss 一目了然）
- 不同类别之间的对比（如多个术语各自的命中情况）
- 微调前后的效果差异（并排柱状图）

> 柱状图的优势在于"数量对比"——让读者快速感知相对大小。""")

# ===== Ch3: 载入 L08 评估结果 =====
md("md-7", """# 第3章  载入 L08 评估结果

复用第8讲的评估结果数据。如果文件不存在，自动生成演示数据——确保本讲可独立运行，但演示数据不代表真实模型表现。""")

code("code-8", """# ===== 第3章：载入 L08 评估结果 =====
import json

RESULT_PATH = Path("l08_evaluation_results.json")
ALT_PATH = Path("../lecture_8/l08_evaluation_results.json")

def load_or_generate():
    for p in [RESULT_PATH, ALT_PATH]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[OK] 已载入: {p}")
            print(f"  样本数: {data['_metadata']['num_samples']}")
            return data

    print("[WARN] 未找到 l08_evaluation_results.json，使用演示数据。")
    print("  演示数据仅供可视化流程验证，不代表真实模型表现。")
    print()

    np.random.seed(42)
    n = 20
    scores = np.round(np.random.uniform(0.5, 12.0, n), 2).tolist()

    demo_terms = ["刻本", "缮写", "金石学", "校勘", "传世本", "拓片", "目录学", "甲骨文"]
    term_hits = [bool(x) for x in np.random.choice([0, 1], size=len(demo_terms), p=[0.5, 0.5])]

    return {
        "_metadata": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "num_samples": n},
        "bleu": {
            "individual_scores": scores,
            "mean": round(float(np.mean(scores)), 2),
            "median": round(float(np.median(scores)), 2),
            "bleu1": 0.312, "bleu2": 0.145, "bleu3": 0.072, "bleu4": 0.038,
            "brevity_penalty": 0.985,
        },
        "term_accuracy": {
            "accuracy_pct": round(np.mean(term_hits) * 100, 1),
            "total_hits": int(np.sum(term_hits)),
            "total_terms": len(term_hits),
            "details": [
                {"term": demo_terms[i], "hit": term_hits[i],
                 "transcript": f"演示转写文本 #{i+1}"}
                for i in range(len(demo_terms))
            ],
        },
        "samples": [
            {"index": i, "transcript": f"演示文本 #{i+1}", "bleu": scores[i]}
            for i in range(n)
        ],
    }

data = load_or_generate()
bleu = data["bleu"]
term_acc = data["term_accuracy"]
samples = data["samples"]

# 按 BLEU 从低到高排序（便于折线图展示趋势）
samples_sorted = sorted(samples, key=lambda x: x["bleu"])
bleu_scores = [s["bleu"] for s in samples_sorted]
labels = [f"#{s['index']}" for s in samples_sorted]

print(f"BLEU 均值: {bleu['mean']:.2f}")
print(f"术语准确率: {term_acc['accuracy_pct']:.1f}% ({term_acc['total_hits']}/{term_acc['total_terms']})")""")

# ===== Ch4: 生成基础图表 =====
md("md-9", """# 第4章  生成基础图表

两张最核心的图表：
1. **BLEU 折线图**——逐样本 BLEU 分数（从低到高排列），展示分数分布趋势
2. **术语准确率柱状图**——命中 vs 未命中的直观对比

先画出"能用但不够好看"的默认版本，下一章再做论文级优化。""")

code("code-10", """# ===== 第4章：生成基础图表（默认样式）=====

# --- 图1：BLEU 逐样本折线图 ---
fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(bleu_scores))
ax.plot(x, bleu_scores, "o-", color="steelblue", linewidth=1.5, markersize=5)
ax.axhline(y=bleu["mean"], color="red", linestyle="--", linewidth=1, label=f'Mean = {bleu["mean"]:.2f}')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_xlabel("Sample (sorted by BLEU)")
ax.set_ylabel("BLEU Score")
ax.set_title("BLEU Scores Across Samples")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
base_line_path = OUT_DIR / "bleu_line_basic.png"
plt.savefig(base_line_path, dpi=150)
plt.show()
print(f"[OK] 基础折线图已保存: {base_line_path}")

# --- 图2：术语准确率柱状图 ---
fig, ax = plt.subplots(figsize=(6, 4.5))

hits = term_acc["total_hits"]
total = term_acc["total_terms"]
misses = total - hits

categories = ["Hit", "Miss"]
values = [hits, misses]
colors = ["steelblue", "lightgray"]

ax.bar(categories, values, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Term Count")
ax.set_title(f"Term Accuracy")
ax.set_ylim(0, total * 1.3)

# 数值标签
for i, v in enumerate(values):
    ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold", fontsize=12)

plt.tight_layout()
base_bar_path = OUT_DIR / "term_bar_basic.png"
plt.savefig(base_bar_path, dpi=150)
plt.show()
print(f"[OK] 基础柱状图已保存: {base_bar_path}")""")

# ===== Ch5: 可视化优化与论文适配（理论） =====
md("md-11", """# 第5章  可视化结果的优化与论文适配规范

## 5.0 学术图表的三个核心原则

| 原则 | 说明 |
|------|------|
| **简洁** | 去除多余的装饰（默认背景色、粗边框），让数据本身成为焦点 |
| **清晰** | 确保字号在论文排版（通常缩小至 8-10cm 宽）后仍可读 |
| **自明** | 图例、坐标轴标签完整，读者无需翻阅正文即可理解图表内容 |

## 5.1 常见优化步骤

**从默认样式到论文样式，通常做以下调整：**

1. **去除 chartjunk**：关闭不必要的网格线、背景填充、顶部/右侧边框（spines）
2. **调整字号**：坐标轴标签用 11pt，标题用 13pt，确保缩小后仍可读
3. **统一配色**：使用灰度友好或色盲友好的调色板（如 Tableau 或 ColorBrewer）
4. **添加数据标注**：在关键数据点旁标注数值，减少读者来回对照的工作量
5. **控制图片尺寸**：论文通常为单栏（~8cm）或双栏（~16cm），提前设定 figsize
6. **选择文件格式**：论文投稿优先用 PDF 或 SVG 矢量图（无损缩放）；PNG 适合预览和 PPT，但印刷可能模糊

## 5.2 折线图的论文适配要点

- 使用实线 + 圆点标记数据点（`"o-"`），确保黑白打印后仍可区分
- 均值参考线用虚线 + 半透明，不喧宾夺主
- X 轴标签若过多（> 20），每隔 N 个标注一个，避免拥挤

## 5.3 柱状图的论文适配要点

- 柱体宽度适中（0.4-0.6），柱间距保持一致性
- 数值标签直接标注在柱顶，减少 Y 轴刻度线的认知负担
- 避免使用 3D 效果——它在学术图表中被视为误导性的视觉修辞""")

# ===== Ch6: 微调图表样式适配论文 =====
md("md-12", """# 第6章  微调图表样式适配论文排版（实操）

基于第5章的规范，对第4章的两张图表做论文级样式优化。

核心改动：
- 移除顶部和右侧边框（spines）
- 统一字号和配色
- 对折线图：添加数据点标注、优化网格
- 对柱状图：添加数值标签、调整配色为论文常用色""")

code("code-13", """# ===== 第6章：论文级图表样式优化 =====

# ---- 全局样式设置 ----
# TIP: 可先用 plt.style.use('seaborn-v0_8-colorblind') 快速获得色盲友好配色，
# 再手动覆盖关键颜色以获得更精细控制（学术图表常见流程）
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ---- 图1：论文级 BLEU 折线图 ----
fig, ax = plt.subplots(figsize=(8, 3.5))  # 论文单栏宽度的合适比例

# 主折线
ax.plot(x, bleu_scores, "o-", color="#2B579A", linewidth=1.5, markersize=4,
        markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#2B579A",
        label="BLEU per sample", zorder=3)

# 均值线
ax.axhline(y=bleu["mean"], color="#D64531", linestyle="--", linewidth=1,
           alpha=0.8, label=f'Mean = {bleu["mean"]:.2f}', zorder=2)

# 移除顶部和右侧边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

# X 轴标签优化：超过 20 个样本时隔位标注
if len(labels) > 20:
    step = len(labels) // 10
    tick_positions = x[::step]
    tick_labels = [labels[i] for i in range(0, len(labels), step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
else:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)

ax.set_xlabel("Sample (sorted by BLEU)")
ax.set_ylabel("BLEU Score")
ax.set_title("BLEU Scores Across Samples", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

plt.tight_layout()
line_path = OUT_DIR / "bleu_line_paper.png"
plt.savefig(line_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"[OK] 论文级折线图已保存: {line_path}")

# ---- 图2：论文级术语准确率柱状图 ----
fig, ax = plt.subplots(figsize=(5.5, 4))

pct = term_acc["accuracy_pct"]

# 论文常用配色：深蓝（Hit）+ 浅灰（Miss）
bar_colors = ["#2B579A", "#D2D2D2"]
bars = ax.bar(categories, values, color=bar_colors, edgecolor="white", width=0.5, linewidth=0.8)

# 柱顶数值标注
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
            str(v), ha="center", fontweight="bold", fontsize=13, color="#333333")

# 移除 spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

# Y 轴只显示刻度，不显示整条轴线
ax.tick_params(left=True, bottom=True, length=3, width=0.5)

ax.set_ylabel("Count")
ax.set_title(f"Term Accuracy: {pct:.1f}%", fontweight="bold")
ax.set_ylim(0, total * 1.3)

plt.tight_layout()
bar_path = OUT_DIR / "term_bar_paper.png"
plt.savefig(bar_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"[OK] 论文级柱状图已保存: {bar_path}")

# ---- 打印优化前后对比 ----
print()
print("=== 图表优化前后对比 ===")
print(f"{'图表':20s} {'优化前':25s} {'优化后':25s}")
print("-" * 70)
print(f"{'BLEU折线图':20s} {str(base_line_path):25s} {str(line_path):25s}")
print(f"{'术语柱状图':20s} {str(base_bar_path):25s} {str(bar_path):25s}")
print()
print("WARNING: Colab 运行时重置后输出文件会丢失。")
print("  建议：左侧文件浏览器 -> 右键 outputs/L09/ -> 下载整个文件夹。")""")

# ===== Ch7: 论文呈现思路 =====
md("md-14", """# 第7章  评估实验在论文中的简要呈现思路

## 7.0 在论文的哪个位置放评估结果？

通常放在"实验与分析"（Experiments）章节，结构为：

```
3. 实验与分析
  3.1 实验设置（模型、数据集、超参数）
  3.2 评估指标（BLEU + 术语准确率）
  3.3 实验结果与分析  ← 本讲的图表放在这里
```

## 7.1 图表在论文中的呈现方式

**一张折线图 + 一段分析文字**是最常见的组合：

> [以下为示例分析文字，请替换为你的实际数据]
> 图 X 展示了 QLoRA 微调后模型在 20 条测试样本上的 BLEU 分数分布。平均 BLEU 为 X.XX，中位数为 X.XX，表明模型在多数样本上能生成与参考答案有一定重叠的解释文本。但样本 #X 和 #X 的 BLEU 显著低于均值，分析发现这两条样本的参考答案包含多个长尾术语，模型尚未充分学习这些术语的语境使用。

**柱状图通常配合一句总结**：

> [以下为示例分析文字，请替换为你的实际数据]
> 术语准确率为 XX.X%（X/X），说明模型已能识别部分高频术语，但对复合术语和古文专有名词的召回率较低。具体薄弱术语见第5章的逐条详情。

## 7.2 两个常见误区

1. **不要只放图表不写分析**——图表是证据，分析是论证。每张图配 2-3 句分析文字。
2. **不要过度解读低 BLEU**——如前所述，中文开放式生成场景的 BLEU 天然偏低，重点是比较相对变化（微调前 vs 微调后），而非追求绝对高分。

## 7.3 从本讲到论文的最小路径

1. 运行第8讲，得到真实的 `l08_evaluation_results.json`
2. 运行本讲，生成论文级图表（`bleu_line_paper.png` + `term_bar_paper.png`）
3. 将图表插入论文的"实验结果与分析"小节
4. 为每张图写 2-3 句分析文字，聚焦"什么因素导致分数高/低"
5. 在结论部分使用术语准确率作为"业务指标"，BLEU 作为"语言学指标"，二者互补论证""")

# ===== Quick Review =====
md("md-15", """# 快速回顾（Quick Check）

**本讲完成了以下工作：**

- **基础可视化**：用折线图展示 BLEU 逐样本趋势，用柱状图对比术语命中情况
- **论文级优化**：去除装饰元素、统一字号配色、控制图表尺寸，使图表可直接用于论文排版
- **论文呈现思路**：了解评估图表在论文中的标准位置、呈现方式和分析写作要点

> 三讲连贯性：
> L03 做微调 → L08 做评估 → **L09 做可视化**（L04-L07 为多模态章节，与此流水线并行）。
> "训练 → 评估 → 可视化"的完整流水线到此闭环。本讲的产物（两张论文级图表）可直接插入你的论文或实验报告。

## 产出文件

| 文件 | 用途 |
|------|------|
| `bleu_line_basic.png` | 基础版 BLEU 折线图（优化前） |
| `term_bar_basic.png` | 基础版术语柱状图（优化前） |
| `bleu_line_paper.png` | 论文级 BLEU 折线图（优化后，推荐使用） |
| `term_bar_paper.png` | 论文级术语柱状图（优化后，推荐使用） |

## 第10讲预告

最后一讲将基于前 9 讲的全部核心技术（加载→微调→评估→可视化），搭建一个简易的"视频-文献-分析"实验原型，形成完整的数字人文 MLLM 科研闭环。""")

# ===== Build notebook =====
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs("D:/dh-llm-course/lecture_9", exist_ok=True)
out_path = "D:/dh-llm-course/lecture_9/L9_visualization.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("L9 notebook written successfully.")

for i, cell in enumerate(cells):
    ct = cell["cell_type"]
    cid = cell["id"]
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    preview = src[:80].replace("\n", " ")
    print(f"  [{i:02d}] {ct:8s} {cid:10s} | {preview}...")
