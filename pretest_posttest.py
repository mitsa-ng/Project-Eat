"""前側後測分析腳本 — 生成範例資料 + 統計分析 + 圖表輸出"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

# ── 設定中文字體 ──
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "pretest_posttest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 生成假資料（30 位學生，前測 vs 後測） ──
np.random.seed(42)
N = 30
pre = np.random.normal(58, 12, N).clip(20, 90)
improvement = np.random.normal(15, 8, N)  # 平均進步 15 分
post = (pre + improvement).clip(30, 100)

# 讓資料更像真實分數（整數）
pre = np.round(pre).astype(int)
post = np.round(post).astype(int)

# ── 2. 描述統計 ──
diff = post - pre
print("=" * 50)
print("  前側 / 後側 描述統計")
print("=" * 50)
print(f"{'':>12} {'平均':>8} {'標準差':>8} {'最小值':>8} {'最大值':>8}")
print(f"{'前測':>12} {pre.mean():>8.2f} {pre.std(ddof=1):>8.2f} {pre.min():>8} {pre.max():>8}")
print(f"{'後測':>12} {post.mean():>8.2f} {post.std(ddof=1):>8.2f} {post.min():>8} {post.max():>8}")
print(f"{'進步分數':>12} {diff.mean():>8.2f} {diff.std(ddof=1):>8.2f} {diff.min():>8} {diff.max():>8}")
print()

# ── 3. 配對 t 檢定 ──
t_stat, p_val = stats.ttest_rel(post, pre)
# Cohen's d
cohen_d = diff.mean() / diff.std(ddof=1)

print("=" * 50)
print("  配對樣本 t 檢定")
print("=" * 50)
print(f"  t({N - 1}) = {t_stat:.3f}")
print(f"  p 值     = {p_val:.6f}")
print(f"  Cohen's d = {cohen_d:.3f}")
print(f"  顯著性   = {'*** 極顯著 (p < 0.001)' if p_val < 0.001 else '** 顯著 (p < 0.01)' if p_val < 0.01 else '* 顯著 (p < 0.05)' if p_val < 0.05 else 'n.s. 不顯著'}")
print()

# ── 4. 圖表輸出 ──

# 顏色主題
C1, C2 = "#4A90D9", "#E8833A"

# --- 圖 A：成對長條圖（含誤差線）---
fig, ax = plt.subplots(figsize=(5, 5))
means = [pre.mean(), post.mean()]
sems = [pre.std(ddof=1) / np.sqrt(N), post.std(ddof=1) / np.sqrt(N)]
bars = ax.bar(["前測", "後測"], means, yerr=sems, capsize=8,
              color=[C1, C2], edgecolor="black", linewidth=1.2,
              error_kw={"linewidth": 1.5})
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("平均分數", fontsize=13)
ax.set_title("前測 vs 後測 平均分數比較", fontsize=15, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_bar_comparison.png", dpi=200)
plt.close(fig)
print(f"  ✓ 01_bar_comparison.png")

# --- 圖 B：盒形圖 ---
fig, ax = plt.subplots(figsize=(5, 5))
bp = ax.boxplot([pre, post], labels=["前測", "後測"], patch_artist=True,
                widths=0.45,
                medianprops={"color": "black", "linewidth": 2})
bp["boxes"][0].set_facecolor(C1)
bp["boxes"][1].set_facecolor(C2)
ax.set_ylabel("分數", fontsize=13)
ax.set_title("前測 vs 後測 分數分布", fontsize=15, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_boxplot.png", dpi=200)
plt.close(fig)
print(f"  ✓ 02_boxplot.png")

# --- 圖 C：個別學生變化折線圖 ---
fig, ax = plt.subplots(figsize=(8, 4.5))
for i in range(N):
    color = "#2ECC71" if diff[i] > 0 else "#E74C3C"
    lw = 0.6 if diff[i] > 0 else 0.8
    alpha = 0.5 if diff[i] > 0 else 0.7
    ax.plot(["前測", "後測"], [pre[i], post[i]], color=color, linewidth=lw, alpha=alpha)
平均值線 = ax.plot(["前測", "後測"], [pre.mean(), post.mean()],
                    color="black", linewidth=2.5, linestyle="--",
                    label=f"平均 ({pre.mean():.0f} → {post.mean():.0f})",
                    zorder=5)
ax.set_ylabel("分數", fontsize=13)
ax.set_title("個別學生前測 → 後測變化", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_individual_lines.png", dpi=200)
plt.close(fig)
print(f"  ✓ 03_individual_lines.png")

# --- 圖 D：進步分數直方圖 ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))
bins = np.arange(diff.min() - 2, diff.max() + 4, 3)
ax.hist(diff, bins=bins, color="#2ECC71", edgecolor="white", linewidth=1.2)
ax.axvline(diff.mean(), color="black", linewidth=2.5, linestyle="--",
           label=f"平均進步 {diff.mean():.1f} 分")
ax.set_xlabel("進步分數", fontsize=13)
ax.set_ylabel("人數", fontsize=13)
ax.set_title("進步分數分布", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_diff_histogram.png", dpi=200)
plt.close(fig)
print(f"  ✓ 04_diff_histogram.png")

# --- 圖 E：小提琴圖 + 蜂群圖 ---
try:
    import seaborn as sns
    df_long = []
    for v, lbl in zip([pre, post], ["前測", "後測"]):
        for x in v:
            df_long.append({"測驗": lbl, "分數": x})
    import pandas as pd
    df_long = pd.DataFrame(df_long)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.violinplot(x="測驗", y="分數", data=df_long, inner=None,
                   palette={"前測": C1, "後測": C2}, ax=ax, linewidth=1.2)
    sns.stripplot(x="測驗", y="分數", data=df_long, color="black",
                  size=4, alpha=0.5, jitter=0.2, ax=ax)
    ax.set_title("前測 vs 後測 分布（小提琴圖）", fontsize=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_violin.png", dpi=200)
    plt.close(fig)
    print(f"  ✓ 05_violin.png (seaborn)")
except ImportError:
    print(f"  - 05_violin.png 跳過（需要 seaborn）")

print()
print(f"  所有圖表已儲存至 {OUTPUT_DIR}/")
print()
