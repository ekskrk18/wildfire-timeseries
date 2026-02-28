import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# USER SETTINGS
# =========================
EVENTS_CSV = "wildfire_events.csv"

WEATHER_DIR = r"outputs_kma\csv"
WEATHER_PATTERN = r"event_{eid:02d}.csv"
TIME_COL_CANDIDATES = ["datetime", "time", "dt", "date_time"]

# columns
COL_TA = "ta"       # °C
COL_RH = "hm"       # %
COL_WS = "ws_10m"   # m/s
COL_RN = "rn_60m"   # mm/h

# KBDI settings
DEFAULT_MAP_MM = 1200.0
KBDI_INIT = 0.0

# windows
T_WINDOWS_H = [1, 24, 72]   # 1h, 1d, 3d
RH_WINDOWS_H = [1, 24, 72]  # 1h, 1d, 3d
WIND_WINDOW_H = 1           # 풍속은 1시간 원자료만 사용 (rolling X)

# rolling ops
ROLL_T = "mean"
ROLL_RH = "min"

# output
OUT_DIR = "outputs_ffdi_9combos"
FIG_DIR = os.path.join(OUT_DIR, "fig")
CSV_DIR = os.path.join(OUT_DIR, "csv")
DPI = 350

# =========================
# Helpers
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

def pick_time_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"시간 컬럼을 찾지 못했습니다. 후보={candidates}, 실제={list(df.columns)}")

def load_events(path):
    ev = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    if ev["event_time"].isna().any():
        bad = ev[ev["event_time"].isna()]
        raise ValueError(f"event_time 파싱 실패:\n{bad}")
    return ev

def load_weather_for_event(eid):
    fp = os.path.join(WEATHER_DIR, WEATHER_PATTERN.format(eid=eid))
    if not os.path.exists(fp):
        raise FileNotFoundError(f"기상 CSV 없음: {fp}")
    df = pd.read_csv(fp, encoding="utf-8-sig")
    tcol = pick_time_col(df, TIME_COL_CANDIDATES)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).rename(columns={tcol: "datetime"})

    need = [COL_TA, COL_RH, COL_WS, COL_RN]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Event {eid}: 기상 컬럼 누락: {miss} / 실제={list(df.columns)}")

    # datetime을 인덱스로 두면 rolling이 편함 (1시간 간격 가정)
    df = df.set_index("datetime").sort_index()
    return df

def ms_to_kmh(v_ms):
    return v_ms * 3.6

def roll_series(s, win_h, how):
    w = int(win_h)
    # 1시간 간격이므로 window=시간개수
    minp = max(1, w // 3)
    if how == "mean":
        return s.rolling(w, min_periods=minp).mean()
    if how == "min":
        return s.rolling(w, min_periods=minp).min()
    if how == "max":
        return s.rolling(w, min_periods=minp).max()
    raise ValueError("unknown rolling op")

# =========================
# KBDI -> DF
# =========================
def kbdi_to_df(kbdi):
    # 0~800 -> 0~10
    return np.minimum(10.0, np.maximum(0.0, kbdi / 80.0))

def compute_kbdi_daily(daily_df, map_mm, kbdi_init=0.0):
    """
    daily_df columns:
      - date (daily)
      - tmax_c
      - rain_mm (mm/day)
    returns daily with KBDI, DF
    """
    mm_to_in = 1.0 / 25.4
    c_to_f = lambda c: c * 9.0/5.0 + 32.0
    map_in = map_mm * mm_to_in

    kbdi = []
    Q = float(kbdi_init)     # 0~800
    carry_rain_in = 0.0      # 0.2 inch threshold handling (simple)

    for _, r in daily_df.iterrows():
        tmax_f = c_to_f(float(r["tmax_c"]))
        rain_in = float(r["rain_mm"]) * mm_to_in

        carry_rain_in += rain_in
        net_rain_in = 0.0
        if carry_rain_in > 0.2:
            net_rain_in = carry_rain_in - 0.2
            carry_rain_in = 0.0

        if net_rain_in > 0:
            Q = max(0.0, Q - 100.0 * net_rain_in)

        if tmax_f < 50.0:
            dQ = 0.0
        else:
            dQ = (800.0 - Q) * (0.968 * math.exp(0.0486 * tmax_f) - 8.30) * 0.001 \
                 / (1.0 + 10.88 * math.exp(-0.0441 * map_in))
            dQ = max(0.0, dQ)

        Q = min(800.0, Q + dQ)
        kbdi.append(Q)

    out = daily_df.copy()
    out["KBDI"] = kbdi
    out["DF"] = kbdi_to_df(out["KBDI"].values)
    return out

def make_daily_kbdi_df(weather_hourly_df, map_mm):
    """
    weather_hourly_df: index=datetime, columns ta/hm/ws_10m/rn_60m
    """
    daily = weather_hourly_df.copy()
    daily["date"] = daily.index.floor("D")
    daily = daily.groupby("date", as_index=False).agg(
        tmax_c=(COL_TA, "max"),
        rain_mm=(COL_RN, "sum")
    )
    return compute_kbdi_daily(daily, map_mm=map_mm, kbdi_init=KBDI_INIT)

# =========================
# FFDI (wind is 1h only)
# =========================
def compute_ffdi_9combos(dfw, daily_kbdi):
    """
    dfw: index=datetime
    daily_kbdi: columns date, DF
    return: wide df (datetime index) with 9 columns of FFDI
    """
    x = dfw.copy()
    x["date"] = x.index.floor("D")
    x = x.merge(daily_kbdi[["date", "DF"]], on="date", how="left")
    x = x.set_index(dfw.index)  # merge로 index 깨질 수 있어 복구

    # wind: 1h raw -> km/h
    V_kmh = ms_to_kmh(x[COL_WS].astype(float))

    out = pd.DataFrame(index=x.index)
    out["DF"] = x["DF"].values

    for tw in T_WINDOWS_H:
        T_agg = roll_series(x[COL_TA].astype(float), tw, ROLL_T)
        for hw in RH_WINDOWS_H:
            RH_agg = roll_series(x[COL_RH].astype(float), hw, ROLL_RH)

            expo = (T_agg - RH_agg) / 30.0 + 0.0234 * V_kmh
            ffdi = 1.25 * x["DF"].astype(float) * np.exp(expo)

            col = f"FFDI_T{tw}h_RH{hw}h"
            out[col] = ffdi

    out = out.reset_index().rename(columns={"index": "datetime"})
    return out

# =========================
# Plotting
# =========================
def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)

def label_for(tw, hw):
    def f(h):
        if h == 24: return "1d"
        if h == 72: return "3d"
        return f"{h}h"
    return f"T={f(tw)}, RH={f(hw)}, V=1h"

def plot_overlay_9(eid, location, event_time, wide_df, outpath):
    fig, ax = plt.subplots(figsize=(10.6, 5.0), constrained_layout=True)

    # 9개 라인을 그리되 legend가 너무 길지 않게 label 정리
    for tw in T_WINDOWS_H:
        for hw in RH_WINDOWS_H:
            col = f"FFDI_T{tw}h_RH{hw}h"
            ax.plot(wide_df["datetime"], wide_df[col], linewidth=1.1, label=label_for(tw, hw))

    ax.axvline(pd.to_datetime(event_time), linestyle=":", linewidth=1.6, color="red")

    ax.set_title(f"Event {eid:02d} FFDI overlay (9 combos) — {location}")
    ax.set_ylabel("FFDI")
    ax.set_xlabel("Date (MM-DD)")
    format_time_axis(ax)

    # legend: 9개면 3열로 배치하면 보기 좋음
    ax.legend(loc="upper left", frameon=True, ncol=3)

    fig.savefig(outpath + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(outpath + ".pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# =========================
# Main
# =========================
def main():
    ensure_dirs()
    events = load_events(EVENTS_CSV)

    for _, ev in events.iterrows():
        eid = int(ev["id"])
        loc = str(ev.get("location", ""))
        t0 = pd.to_datetime(ev["event_time"])

        print(f"[INFO] Event {eid:02d}: {loc}")

        dfw = load_weather_for_event(eid)

        map_mm = float(ev["map_mm"]) if ("map_mm" in ev.index and pd.notna(ev["map_mm"])) else DEFAULT_MAP_MM
        daily_kbdi = make_daily_kbdi_df(dfw, map_mm=map_mm)

        wide = compute_ffdi_9combos(dfw, daily_kbdi)

        # save csv (wide)
        out_csv = os.path.join(CSV_DIR, f"event_{eid:02d}_FFDI_9combos.csv")
        wide.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # plot overlay
        out_fig = os.path.join(FIG_DIR, f"event_{eid:02d}_FFDI_9combos_overlay")
        plot_overlay_9(eid, loc, t0, wide, out_fig)

        print(f"  -> saved: {out_csv}")
        print(f"  -> saved: {out_fig}.png/.pdf")

    print("[DONE]")

if __name__ == "__main__":
    main()
