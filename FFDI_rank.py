import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn이 없을 수도 있으니 안전하게 처리
try:
    import seaborn as sns
except ImportError:
    sns = None

# ===============================
# USER SETTINGS
# ===============================
DATA_DIR = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\21th Wildfire\outputs_ffdi_9combos\csv"   # event_XX_FFDI_9combos.csv가 있는 폴더
EVENT_CSV_PATTERN = "event_*_FFDI_9combos.csv"

EVENTS_CSV = "wildfire_events.csv"   # event_time 가져오기용
LOOKBACK_DAYS = 7                    # 사건 전 7일 구간에서 percentile 계산

DPI = 350
OUT_DIR = "outputs_ffdi_figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ===============================
# Helper functions
# ===============================
def extract_event_id(filepath: str) -> int:
    # event_01_FFDI_9combos.csv -> 1
    base = os.path.basename(filepath)
    parts = base.split("_")
    return int(parts[1])

def compute_event_time_percentile(df: pd.DataFrame, ffdi_col: str, t0: pd.Timestamp) -> float:
    """
    사건시점 t0 기준, [t0-7d, t0] 구간에서
    FFDI(t0)가 상위 몇 %인지(percentile; 0~1)를 계산.
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d = d.dropna(subset=["datetime"]).sort_values("datetime")

    t0 = pd.to_datetime(t0)

    window = d[(d["datetime"] >= t0 - pd.Timedelta(days=LOOKBACK_DAYS)) &
               (d["datetime"] <= t0)].copy()
    if window.empty:
        return np.nan

    # t0와 가장 가까운 시간 선택
    idx0 = (window["datetime"] - t0).abs().idxmin()
    f0 = window.loc[idx0, ffdi_col]

    # 결측/무한 제거
    vals = window[ffdi_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 5:
        return np.nan

    return float((vals <= f0).mean())

def parse_combo(combo_str: str):
    """
    combo_str 예: "T1h_RH24h"
    return: ("1h", "24h")
    """
    # 기대 포맷: T{t}_RH{rh}
    parts = combo_str.split("_")
    t = parts[0].replace("T", "")
    rh = parts[1].replace("RH", "")
    return t, rh


# ===============================
# Load event times
# ===============================
if not os.path.exists(EVENTS_CSV):
    raise FileNotFoundError(f"'{EVENTS_CSV}' 파일이 현재 폴더에 없습니다. (event_time 읽기용)")

events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig").dropna(how="all")
if "id" not in events.columns or "event_time" not in events.columns:
    raise ValueError(f"{EVENTS_CSV}에 'id', 'event_time' 컬럼이 필요합니다. 현재 컬럼: {list(events.columns)}")

events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
bad = events[events["event_time"].isna()]
if not bad.empty:
    raise ValueError(f"event_time 파싱 실패 행이 있습니다:\n{bad[['id','event_time']]}")

event_time_dict = dict(zip(events["id"].astype(int), events["event_time"]))


# ===============================
# Collect percentiles from uploaded CSVs
# ===============================
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, EVENT_CSV_PATTERN)))
if len(csv_files) == 0:
    raise FileNotFoundError(f"'{DATA_DIR}'에서 '{EVENT_CSV_PATTERN}' 파일을 찾지 못했습니다.")

records = []

for fp in csv_files:
    eid = extract_event_id(fp)
    if eid not in event_time_dict:
        print(f"[WARN] events.csv에 event_id={eid}가 없어서 스킵: {fp}")
        continue

    t0 = event_time_dict[eid]
    df = pd.read_csv(fp, encoding="utf-8-sig")

    if "datetime" not in df.columns:
        raise ValueError(f"{fp}에 'datetime' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    ffdi_cols = [c for c in df.columns if c.startswith("FFDI_")]
    if len(ffdi_cols) == 0:
        raise ValueError(f"{fp}에서 'FFDI_'로 시작하는 컬럼을 찾지 못했습니다.")

    for col in ffdi_cols:
        p = compute_event_time_percentile(df, col, t0)

        # ✅ combo 컬럼을 확실히 만들어줌 (에러의 원인 해결)
        records.append({
            "event_id": int(eid),
            "combo": col.replace("FFDI_", ""),   # 예: "T1h_RH24h"
            "percentile": p
        })

percentile_df = pd.DataFrame(records)

# ---- 디버그 출력 & 방어 코드 ----
print("[DEBUG] percentile_df columns:", list(percentile_df.columns))
print("[DEBUG] percentile_df head:\n", percentile_df.head())

assert "combo" in percentile_df.columns, "percentile_df에 combo 컬럼이 없습니다."
assert "percentile" in percentile_df.columns, "percentile_df에 percentile 컬럼이 없습니다."
assert "event_id" in percentile_df.columns, "percentile_df에 event_id 컬럼이 없습니다."

# 결측 제거(그림 안정화)
percentile_df = percentile_df.dropna(subset=["percentile"]).copy()
if percentile_df.empty:
    raise RuntimeError("percentile 계산 결과가 모두 NaN입니다. datetime 범위/이벤트 시각/데이터 기간을 확인하세요.")


# ===============================
# 1) BOXPLOT: event-time percentile by combo
# ===============================
box_out_png = os.path.join(OUT_DIR, "boxplot_event_time_percentile.png")
box_out_pdf = os.path.join(OUT_DIR, "boxplot_event_time_percentile.pdf")

plt.figure(figsize=(10.5, 5.2))

if sns is not None:
    sns.boxplot(data=percentile_df, x="combo", y="percentile")
else:
    # seaborn 없을 경우 matplotlib 대체(간단 버전)
    combos = sorted(percentile_df["combo"].unique())
    data = [percentile_df.loc[percentile_df["combo"] == c, "percentile"].values for c in combos]
    plt.boxplot(data, labels=combos)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Event-time percentile (0–1)")
plt.xlabel("FFDI combination (T window & RH window, V=1h)")
plt.title("Event-time percentile distribution by FFDI combination")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(box_out_png, dpi=DPI)
plt.savefig(box_out_pdf, dpi=DPI)
plt.close()


# ===============================
# 2) HEATMAP: mean percentile (RH x T)
# ===============================
# combo별 평균 percentile
heat = (percentile_df.groupby("combo")["percentile"]
        .mean()
        .reset_index()
        .rename(columns={"percentile": "mean_percentile"}))

heat["T"] = heat["combo"].apply(lambda s: parse_combo(s)[0])    # "1h","24h","72h"
heat["RH"] = heat["combo"].apply(lambda s: parse_combo(s)[1])

pivot = heat.pivot(index="RH", columns="T", values="mean_percentile")

# 보기 좋게 정렬
t_order = ["1h", "24h", "72h"]
rh_order = ["1h", "24h", "72h"]
pivot = pivot.reindex(index=rh_order, columns=t_order)

hm_out_png = os.path.join(OUT_DIR, "heatmap_mean_percentile.png")
hm_out_pdf = os.path.join(OUT_DIR, "heatmap_mean_percentile.pdf")

plt.figure(figsize=(6.3, 5.3))

if sns is not None:
    sns.heatmap(pivot, annot=True, fmt=".3f", cbar_kws={"label": "Mean event-time percentile"})
else:
    # seaborn 없을 경우 matplotlib 대체
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mean event-time percentile")
    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.3f}", ha="center", va="center")

plt.xlabel("Temperature window (T)")
plt.ylabel("Humidity window (RH)")
plt.title("Mean event-time percentile heatmap")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.tight_layout()
plt.savefig(hm_out_png, dpi=DPI)
plt.savefig(hm_out_pdf, dpi=DPI)
plt.close()


# ===============================
# 3) RANK STABILITY: Top 3 combos
# ===============================
rank_df = percentile_df.copy()
# event별 percentile 높은 순(=1이 best)
rank_df["rank"] = rank_df.groupby("event_id")["percentile"].rank(ascending=False, method="min")

# combo별 평균 rank로 Top3 선정
mean_rank = rank_df.groupby("combo")["rank"].mean().sort_values()
top3 = mean_rank.head(3).index.tolist()

rs_out_png = os.path.join(OUT_DIR, "rank_stability_top3.png")
rs_out_pdf = os.path.join(OUT_DIR, "rank_stability_top3.pdf")

plt.figure(figsize=(8.5, 5.2))
for combo in top3:
    sub = rank_df[rank_df["combo"] == combo].sort_values("event_id")
    plt.plot(sub["event_id"], sub["rank"], marker="o", linewidth=1.6, label=combo)

plt.gca().invert_yaxis()
plt.xlabel("Event ID")
plt.ylabel("Rank (1 = best)")
plt.title("Rank stability plot (Top 3 combinations)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(rs_out_png, dpi=DPI)
plt.savefig(rs_out_pdf, dpi=DPI)
plt.close()


# ===============================
# Save summary tables (optional but useful)
# ===============================
percentile_df.to_csv(os.path.join(OUT_DIR, "percentile_by_event_and_combo.csv"),
                     index=False, encoding="utf-8-sig")
mean_rank.reset_index().rename(columns={"rank": "mean_rank"}).to_csv(
    os.path.join(OUT_DIR, "combo_mean_rank.csv"), index=False, encoding="utf-8-sig"
)

print("[DONE]")
print("Saved figures:")
print(" -", box_out_png)
print(" -", hm_out_png)
print(" -", rs_out_png)
print("Saved tables:")
print(" -", os.path.join(OUT_DIR, "percentile_by_event_and_combo.csv"))
print(" -", os.path.join(OUT_DIR, "combo_mean_rank.csv"))
