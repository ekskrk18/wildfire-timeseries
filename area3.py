import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# USER SETTINGS
# =========================================================
RANK_CSV = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\21th Wildfire\outputs_area_heatmap_exclude1_rank36\rank_36pairs_excl1.csv"
OUT_DIR = "outputs_summary_figures"
TOP_K = 10
DPI = 350

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# combo 이름에서 T / RH window 파싱 (네 naming rule에 맞게!)
# 예: T1h_RH3d, T1d_RH1h ...
# ---------------------------------------------------------
def parse_windows(combo):
    parts = combo.split("_")
    T = parts[0].replace("T", "")
    RH = parts[1].replace("RH", "")
    return T, RH


# =========================================================
# Load ranking data
# =========================================================
df = pd.read_csv(RANK_CSV)

# combo, case, rank, score, rho, tau, r2, rmse 포함되어 있음
df[["T_window", "RH_window"]] = df["combo"].apply(
    lambda x: pd.Series(parse_windows(x))
)

# =========================================================
# FIGURE 1: Temperature–RH window summary heatmap
# (rank 평균 기준 → 낮을수록 좋음)
# =========================================================
pivot_TRH = (
    df.groupby(["RH_window", "T_window"])["rank"]
      .mean()
      .reset_index()
      .pivot(index="RH_window", columns="T_window", values="rank")
)

plt.figure(figsize=(5.2, 4.4))
sns.heatmap(
    pivot_TRH,
    annot=True,
    fmt=".1f",
    cmap="viridis_r",
    cbar_kws={"label": "Mean rank (lower = better)"}
)
plt.title("Temperature–Humidity Window Performance Summary")
plt.xlabel("Temperature accumulation window")
plt.ylabel("Humidity accumulation window")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_TRH_window_meanRank.png"), dpi=DPI)
plt.close()

print("[OK] Figure 1 saved: TRH window heatmap")


# =========================================================
# FIGURE 2: Vegetation configuration rank boxplot
# =========================================================
plt.figure(figsize=(6.2, 4.5))
sns.boxplot(
    data=df,
    x="case",
    y="rank",
    showfliers=True
)

plt.gca().invert_yaxis()  # rank 1이 위로 오도록
plt.title("Rank Distribution by Vegetation Configuration")
plt.xlabel("Vegetation configuration")
plt.ylabel("Overall rank (lower = better)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_vegetation_rank_boxplot.png"), dpi=DPI)
plt.close()

print("[OK] Figure 2 saved: Vegetation rank boxplot")


# =========================================================
# FIGURE 3A: Top-K frequency heatmap (T–RH)
# =========================================================
topk = df[df["rank"] <= TOP_K]

freq_TRH = (
    topk.groupby(["RH_window", "T_window"])
        .size()
        .reset_index(name="count")
        .pivot(index="RH_window", columns="T_window", values="count")
        .fillna(0)
)

plt.figure(figsize=(5.2, 4.4))
sns.heatmap(
    freq_TRH,
    annot=True,
    fmt=".0f",
    cmap="Reds",
    cbar_kws={"label": f"Count in Top {TOP_K}"}
)
plt.title(f"Top-{TOP_K} Frequency of T–RH Window Combinations")
plt.xlabel("Temperature accumulation window")
plt.ylabel("Humidity accumulation window")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"Fig_TRH_top{TOP_K}_frequency.png"), dpi=DPI)
plt.close()

print("[OK] Figure 3A saved: Top-K TRH frequency heatmap")


# =========================================================
# FIGURE 3B: Top-K frequency by vegetation case (bar plot)
# =========================================================
freq_case = (
    topk["case"]
    .value_counts()
    .reindex(sorted(df["case"].unique()))
)

plt.figure(figsize=(5.4, 4.2))
freq_case.plot(kind="bar")

plt.title(f"Top-{TOP_K} Frequency by Vegetation Configuration")
plt.xlabel("Vegetation configuration")
plt.ylabel(f"Count in Top {TOP_K}")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"Fig_vegetation_top{TOP_K}_frequency.png"), dpi=DPI)
plt.close()

print("[OK] Figure 3B saved: Vegetation Top-K frequency barplot")

print("\n[DONE] All summary figures generated.")
print(f"Output directory: {OUT_DIR}")
