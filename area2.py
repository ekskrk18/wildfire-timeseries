import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


# =========================================================
# USER SETTINGS
# =========================================================
EVENTS_CSV = "wildfire_events.csv"

FFDI_DIR = r"outputs_ffdi_9combos\csv"
FFDI_PATTERN = "event_*_FFDI_9combos.csv"

SAT_DIR = r"outputs_gk2a_vgt\csv"
SAT_PATTERN = "event_*_VGT_point.csv"

LOOKBACK_DAYS = 7
FFDI_SUMMARY = "auc_pre7d"     # "auc_pre7d" or "max_pre7d" or "last"
SAT_SUMMARY = "mean_pre7d"     # (현재는 mean만 사용)

LOG_AREA = True
LOG_EPS = 1.0

RIDGE_ALPHA = 1.0

OUT_DIR = "outputs_area_heatmap_exclude1_rank36"
DPI = 350
os.makedirs(OUT_DIR, exist_ok=True)

# Obs-Pred plot 생성 수 (Top 5 + Bottom 5)
TOP_N = 5
BOTTOM_N = 5


# =========================================================
# Helper functions
# =========================================================
def extract_event_id(fp):
    return int(os.path.basename(fp).split("_")[1])

def clean_area(x):
    return float(str(x).replace(",", ""))

def window_df(df, tcol, t0, days):
    t0 = pd.to_datetime(t0)
    return df[(df[tcol] >= t0 - pd.Timedelta(days=days)) & (df[tcol] <= t0)]

def summarize_ffdi(df, col, t0):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    w = window_df(df, "datetime", t0, LOOKBACK_DAYS)
    s = w[col].dropna()
    if s.empty:
        return np.nan
    if FFDI_SUMMARY == "auc_pre7d":
        return float(s.sum())
    if FFDI_SUMMARY == "max_pre7d":
        return float(s.max())
    return float(s.iloc[-1])

def summarize_sat(df, col, t0):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    w = window_df(df, "datetime", t0, LOOKBACK_DAYS)
    s = w[col].dropna()
    if s.empty:
        return np.nan
    return float(s.mean())

def loocv_predict(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)

    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(Xtr, y[tr])
        preds[te[0]] = model.predict(Xte)[0]

    return preds

def metrics_from_preds(y, preds):
    return {
        "rho": float(spearmanr(preds, y)[0]),
        "tau": float(kendalltau(preds, y)[0]),
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
    }

def normalize_col(s, higher_is_better=True):
    """0~1 min-max normalize (전체 36개 풀에서 정규화)"""
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0):
        return pd.Series(np.ones_like(s), index=s.index) * 0.5
    z = (s - mn) / (mx - mn)
    return z if higher_is_better else (1 - z)

def save_obs_pred_plot(y, preds, title, outpath, m):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.scatter(y, preds)

    mn = min(float(np.min(y)), float(np.min(preds)))
    mx = max(float(np.max(y)), float(np.max(preds)))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

    txt = (
        f"ρ={m['rho']:.3f}\n"
        f"τ={m['tau']:.3f}\n"
        f"R²={m['r2']:.3f}\n"
        f"RMSE={m['rmse']:.3f}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top")

    ax.set_xlabel("Observed " + ("log(area+1)" if LOG_AREA else "area"))
    ax.set_ylabel("Predicted " + ("log(area+1)" if LOG_AREA else "area"))
    ax.set_title(title)

    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Load event metadata
# =========================================================
events = pd.read_csv(EVENTS_CSV)
events["id"] = events["id"].astype(int)
events["event_time"] = pd.to_datetime(events["event_time"])
events["area_ha"] = events["area_ha"].apply(clean_area)

event_time = dict(zip(events.id, events.event_time))
event_area = dict(zip(events.id, events.area_ha))


# =========================================================
# File discovery
# =========================================================
ffdi_files = {extract_event_id(f): f for f in glob.glob(os.path.join(FFDI_DIR, FFDI_PATTERN))}
sat_files  = {extract_event_id(f): f for f in glob.glob(os.path.join(SAT_DIR, SAT_PATTERN))}

common_ids = sorted(set(ffdi_files) & set(sat_files) & set(event_time))
if len(common_ids) < 6:
    raise RuntimeError("유효 이벤트 수 부족")

tmp = pd.read_csv(ffdi_files[common_ids[0]])
combo_cols = [c for c in tmp.columns if c.startswith("FFDI_")]
combo_names = [c.replace("FFDI_", "") for c in combo_cols]


# =========================================================
# Feature table (ALL events)
# =========================================================
rows = []
for eid in common_ids:
    t0 = event_time[eid]
    y = np.log(event_area[eid] + LOG_EPS) if LOG_AREA else event_area[eid]

    sat = pd.read_csv(sat_files[eid])
    ndvi = summarize_sat(sat, "NDVI", t0)
    evi  = summarize_sat(sat, "EVI", t0)

    ff = pd.read_csv(ffdi_files[eid])
    for col, cname in zip(combo_cols, combo_names):
        fval = summarize_ffdi(ff, col, t0)
        rows.append({
            "event_id": eid,
            "combo": cname,
            "FFDI": fval,
            "NDVI": ndvi,
            "EVI": evi,
            "y": y
        })

df_all = pd.DataFrame(rows).dropna()
df_all.to_csv(os.path.join(OUT_DIR, "feature_table_event_combo_ALL.csv"),
              index=False, encoding="utf-8-sig")


# =========================================================
# Cases
# =========================================================
cases = {
    "A: FFDI": ["FFDI"],
    "B: FFDI+NDVI": ["FFDI", "NDVI"],
    "C: FFDI+EVI": ["FFDI", "EVI"],
    "D: FFDI+NDVI+EVI": ["FFDI", "NDVI", "EVI"]
}


# =========================================================
# Step 1) Auto-detect 1 outlier event (reference = Case D + best rho combo)
# =========================================================
ref_case = "D: FFDI+NDVI+EVI"
ref_cols = cases[ref_case]

mean_rho_by_combo = {}
for cname in combo_names:
    sub = df_all[df_all.combo == cname].sort_values("event_id")
    X = sub[ref_cols].values
    y = sub["y"].values
    preds = loocv_predict(X, y)
    mean_rho_by_combo[cname] = float(spearmanr(preds, y)[0])

ref_combo = max(mean_rho_by_combo, key=mean_rho_by_combo.get)
sub_ref = df_all[df_all.combo == ref_combo].sort_values("event_id")
preds_ref = loocv_predict(sub_ref[ref_cols].values, sub_ref["y"].values)
residuals = np.abs(preds_ref - sub_ref["y"].values)

outlier_idx = int(np.argmax(residuals))
OUTLIER_EVENT_ID = int(sub_ref.iloc[outlier_idx]["event_id"])

print(f"[INFO] Reference combo={ref_combo} / case={ref_case}")
print(f"[INFO] Auto-outlier event_id = {OUTLIER_EVENT_ID}")

# exclude outlier
df = df_all[df_all.event_id != OUTLIER_EVENT_ID].copy()
df.to_csv(os.path.join(OUT_DIR, "feature_table_event_combo_excl1.csv"),
          index=False, encoding="utf-8-sig")


# =========================================================
# Step 2) Compute LOOCV metrics for ALL (combo, case) pairs = 36
# =========================================================
detail_rows = []

for combo in combo_names:
    sub = df[df.combo == combo].sort_values("event_id")
    y = sub["y"].values

    for case_name, cols in cases.items():
        X = sub[cols].values
        preds = loocv_predict(X, y)
        m = metrics_from_preds(y, preds)

        detail_rows.append({
            "combo": combo,
            "case": case_name,
            **m
        })

detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv(os.path.join(OUT_DIR, "metrics_36pairs_excl1.csv"),
                 index=False, encoding="utf-8-sig")


# =========================================================
# Step 3) Heatmaps (4x9) for each metric
# =========================================================
heatmaps = {}
for metric in ["rho", "tau", "r2", "rmse"]:
    pivot = detail_df.pivot(index="case", columns="combo", values=metric).astype(float)
    heatmaps[metric] = pivot
    pivot.to_csv(os.path.join(OUT_DIR, f"heatmap_{metric}_excl1.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 3.8), constrained_layout=True)
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    title_map = {"rho": "Spearman ρ", "tau": "Kendall τ", "r2": "R²", "rmse": "RMSE"}
    ax.set_title(f"{title_map[metric]} heatmap (9 events, outlier excluded: {OUTLIER_EVENT_ID})")

    plt.colorbar(im, ax=ax)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center")

    fig.savefig(os.path.join(OUT_DIR, f"Fig_heatmap_{metric}_excl1.png"), dpi=DPI)
    plt.close(fig)


# =========================================================
# Step 4) Rank ALL 36 (combo, case) pairs using normalized score
#   - rho/tau/r2 higher better
#   - rmse lower better
# =========================================================
rank_df = detail_df.copy()

rank_df["score"] = (
    normalize_col(rank_df["rho"], True)
    + normalize_col(rank_df["tau"], True)
    + normalize_col(rank_df["r2"], True)
    + normalize_col(rank_df["rmse"], False)
)

rank_df = rank_df.sort_values("score", ascending=False).reset_index(drop=True)
rank_df["rank"] = np.arange(1, len(rank_df) + 1)

rank_df.to_csv(os.path.join(OUT_DIR, "rank_36pairs_excl1.csv"),
               index=False, encoding="utf-8-sig")

top_pairs = rank_df.iloc[:TOP_N].copy()
bottom_pairs = rank_df.iloc[-BOTTOM_N:].copy()
selected_pairs = pd.concat([top_pairs, bottom_pairs], axis=0)

selected_pairs.to_csv(os.path.join(OUT_DIR, "TopBottom_pairs_excl1.csv"),
                      index=False, encoding="utf-8-sig")

print("[INFO] Selected pairs (Top/Bottom):")
print(selected_pairs[["rank", "combo", "case", "score", "rho", "tau", "r2", "rmse"]])


# =========================================================
# Step 5) Generate Obs-Pred plots for Top 5 + Bottom 5 pairs
# =========================================================
plot_dir = os.path.join(OUT_DIR, "obs_pred_topbottom_36pairs")
os.makedirs(plot_dir, exist_ok=True)

for _, row in selected_pairs.iterrows():
    combo = row["combo"]
    case_name = row["case"]
    rank_no = int(row["rank"])

    sub = df[df.combo == combo].sort_values("event_id")
    y = sub["y"].values
    X = sub[cases[case_name]].values

    preds = loocv_predict(X, y)
    m = metrics_from_preds(y, preds)

    title = f"Rank {rank_no}: {combo} | {case_name} | excl outlier={OUTLIER_EVENT_ID}"
    outpath = os.path.join(plot_dir, f"ObsPred_rank{rank_no:02d}_{combo}_{case_name.replace(':','')}.png")

    save_obs_pred_plot(y, preds, title, outpath, m)

print("[DONE]")
print(f"Outlier excluded event_id = {OUTLIER_EVENT_ID}")
print(f"Heatmaps + 36-pair ranking + Top/Bottom ObsPred plots saved to: {OUT_DIR}")
