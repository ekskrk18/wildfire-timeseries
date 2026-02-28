import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# USER SETTINGS
# =========================
EVENTS_CSV = "wildfire_events.csv"

WEATHER_DIR = r"outputs_kma\csv"
WEATHER_PATTERN = r"event_{eid:02d}.csv"   # 필요시 수정

VGT_DIR = r"outputs_gk2a_vgt\csv"
VGT_PATTERN = r"event_{eid:02d}_VGT_point.csv"     # 필요시 수정

FIG_DIR = "figures_timeseries_split_weather"
DPI = 350  # 300 이상

WEATHER_TIME_COL_CANDIDATES = ["datetime", "time", "dt", "date_time"]
VGT_TIME_COL_CANDIDATES = ["datetime", "time", "dt", "date_time"]

# 기상 컬럼
COL_TA = "ta"
COL_HM = "hm"
COL_WS = "ws_10m"
COL_RN = "rn_60m"

# 위성 컬럼
COL_NDVI = "NDVI"
COL_EVI  = "EVI"
COL_FVC  = "FVC"


# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

def load_weather(eid):
    fp = os.path.join(WEATHER_DIR, WEATHER_PATTERN.format(eid=eid))
    if not os.path.exists(fp):
        raise FileNotFoundError(f"기상 CSV 없음: {fp}")
    df = pd.read_csv(fp, encoding="utf-8-sig")
    tcol = pick_time_col(df, WEATHER_TIME_COL_CANDIDATES)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    df = df.rename(columns={tcol: "datetime"})
    return df

def load_vgt(eid):
    fp = os.path.join(VGT_DIR, VGT_PATTERN.format(eid=eid))
    if not os.path.exists(fp):
        raise FileNotFoundError(f"VGT CSV 없음: {fp}")
    df = pd.read_csv(fp, encoding="utf-8-sig")
    tcol = pick_time_col(df, VGT_TIME_COL_CANDIDATES)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    df = df.rename(columns={tcol: "datetime"})
    return df

def save_fig(fig, out_base):
    fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight")

def format_time_axis(ax, hourly=True):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if hourly:
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)


# =========================
# Plotters
# =========================
def plot_temp_wind(eid, location, event_time, dfw):
    req = [COL_TA, COL_WS]
    missing = [c for c in req if c not in dfw.columns]
    if missing:
        raise ValueError(f"Event {eid}: 컬럼 없음: {missing}")

    fig, ax1 = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    ax2 = ax1.twinx()

    # 구분 잘 되도록: 선 스타일 + 마커 다르게
    l1, = ax1.plot(dfw["datetime"], dfw[COL_TA], linewidth=1.4, label="Temp (°C)")
    l2, = ax2.plot(dfw["datetime"], dfw[COL_WS], linewidth=1.4, linestyle="--", label="Wind (m/s)")

    # 사고시점
    ax1.axvline(event_time, linestyle=":", linewidth=1.6, color="red")
    ax2.axvline(event_time, linestyle=":", linewidth=1.6, color="red")

    ax1.set_title(f"Event {eid:02d} Weather — Temp & Wind — {location}")
    ax1.set_ylabel("Temp (°C)")
    ax2.set_ylabel("Wind (m/s)")
    ax1.set_xlabel("Date (MM-DD)")

    # 범례 합치기
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    format_time_axis(ax1, hourly=True)
    return fig


def plot_humidity_rain(eid, location, event_time, dfw):
    req = [COL_HM, COL_RN]
    missing = [c for c in req if c not in dfw.columns]
    if missing:
        raise ValueError(f"Event {eid}: 컬럼 없음: {missing}")

    fig, ax1 = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    ax2 = ax1.twinx()

    l1, = ax1.plot(dfw["datetime"], dfw[COL_HM], linewidth=1.4, label="RH (%)")
    # 강우는 더 잘 보이게 점선+마커(또는 bar로 바꿀 수도 있음)
    l2, = ax2.plot(dfw["datetime"], dfw[COL_RN], linewidth=1.4, linestyle="--", label="Rain (mm/h)")

    ax1.axvline(event_time, linestyle=":", linewidth=1.6, color="red")
    ax2.axvline(event_time, linestyle=":", linewidth=1.6, color="red")

    ax1.set_title(f"Event {eid:02d} Weather — Humidity & Rain — {location}")
    ax1.set_ylabel("RH (%)")
    ax2.set_ylabel("Rain (mm/h)")
    ax1.set_xlabel("Date (MM-DD)")

    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    format_time_axis(ax1, hourly=True)
    return fig


def plot_vgt_one(eid, location, event_time, dfv):
    req = [COL_NDVI, COL_EVI, COL_FVC]
    missing = [c for c in req if c not in dfv.columns]
    if missing:
        raise ValueError(f"Event {eid}: VGT 컬럼 없음: {missing}")

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)

    ax.plot(dfv["datetime"], dfv[COL_NDVI], marker="o", linewidth=1.2, label="NDVI")
    ax.plot(dfv["datetime"], dfv[COL_EVI],  marker="o", linewidth=1.2, label="EVI")
    ax.plot(dfv["datetime"], dfv[COL_FVC],  marker="o", linewidth=1.2, label="FVC")

    ax.axvline(event_time, linestyle=":", linewidth=1.6, color="red")

    ax.set_title(f"Event {eid:02d} Vegetation (GK2A VGT) — {location}")
    ax.set_ylabel("Index value")
    ax.set_xlabel("Date (MM-DD)")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, which="major", alpha=0.25)

    ax.legend(loc="upper left", frameon=True)
    return fig


# =========================
# Main
# =========================
def main():
    ensure_dir(FIG_DIR)
    events = load_events(EVENTS_CSV)

    for _, row in events.iterrows():
        eid = int(row["id"])
        location = str(row.get("location", ""))
        event_time = pd.to_datetime(row["event_time"])

        print(f"[INFO] Plotting Event {eid:02d} — {location}")

        dfw = load_weather(eid)
        dfv = load_vgt(eid)

        # 1) Temp & Wind
        fig1 = plot_temp_wind(eid, location, event_time, dfw)
        out1 = os.path.join(FIG_DIR, f"event_{eid:02d}_weather_temp_wind")
        save_fig(fig1, out1)
        plt.close(fig1)

        # 2) Humidity & Rain
        fig2 = plot_humidity_rain(eid, location, event_time, dfw)
        out2 = os.path.join(FIG_DIR, f"event_{eid:02d}_weather_humidity_rain")
        save_fig(fig2, out2)
        plt.close(fig2)

        # 3) VGT (그대로 유지)
        fig3 = plot_vgt_one(eid, location, event_time, dfv)
        out3 = os.path.join(FIG_DIR, f"event_{eid:02d}_vgt")
        save_fig(fig3, out3)
        plt.close(fig3)

        print(f"  -> saved: {out1}.png/.pdf")
        print(f"  -> saved: {out2}.png/.pdf")
        print(f"  -> saved: {out3}.png/.pdf")

    print("[DONE] All figures saved.")

if __name__ == "__main__":
    main()
