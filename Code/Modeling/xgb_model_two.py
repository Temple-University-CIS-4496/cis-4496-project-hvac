# =============================================================================
# hvac runtime prediction — xgboost with integrated raw-data preprocessing
# target: daily_runtime_hours
# data:   raw kaggle thermostats + local outdoor weather csv
# =============================================================================

import os
import warnings
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

KAGGLE_TOKEN      = "KGAT_6d26ebf584b9d44052424f267dffec66"
OUTDOOR_PATH      = "/Users/zack/projects/datasci/cis-4496-project-hvac/Sample_Data/Raw/outdoorweather.csv"

# preprocessing constants (mirror the original pipeline)
TIME_THRESHOLD_MINUTES = 30
NUMERIC_RESET_TOKEN    = -9999.9
STEP_LIKE_COLS         = ["Setpoint", "FanState"]
OUTDOOR_TREND_WINDOW   = 7
LAG_DAYS               = 7

# features fed to xgboost — daily-level, no leakage
FEATURE_COLS = [
    # runtime history lags
    "daily_runtime_hours_lag_1",
    "daily_runtime_hours_lag_2",
    "daily_runtime_hours_lag_3",
    "daily_runtime_hours_lag_7",
    # heating / cooling history
    "daily_heating_hours_lag_1",
    "daily_cooling_hours_lag_1",
    # indoor thermal state lags
    "indoor_temp_time_weighted_mean_lag_1",
    "setpoint_time_weighted_mean_lag_1",
    "setpoint_gap_mean_lag_1",
    "indoor_temp_std_lag_1",
    # today's weather (exogenous — known at prediction time)
    "true_outside_min",
    "true_outside_max",
    "true_outside_mean",
    "true_humidity_mean",
    "outdoor_temp_trend_gradient",
    # calendar
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    # thermostat identity stats (computed from training set only)
    "thermo_mean_runtime",
    "thermo_std_runtime",
]

TARGET_COL = "daily_runtime_hours"


# =============================================================================
# SECTION 1 — PREPROCESSING (adapted from the original pipeline)
# =============================================================================

def process_outdoor_weather(path):
    """read and standardise the outdoor weather csv."""
    dfw = pd.read_csv(path, sep=";", low_memory=False)
    dfw["Timestamp"] = pd.to_datetime(dfw["Timestamp"], errors="coerce")
    dfw = dfw.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    dfw = dfw.rename(columns={"Temperature": "Outdoor_Temperature"})
    for c in ["Outdoor_Temperature", "outsideMinTemp", "outsideMaxTemp", "outsideHumidity"]:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce")
    return dfw[["Timestamp", "Outdoor_Temperature", "outsideMinTemp",
                "outsideMaxTemp", "outsideHumidity"]]


def standardize_indoor_columns(df):
    rename_map = {
        "output_state": "OutputState",
        "fan_state":    "FanState",
        "running_mode": "RunningMode",
        "hvac_state":   "RunningMode",
        "temp":         "Temperature",
        "set_point":    "Setpoint",
    }
    return df.rename(columns=rename_map)


def resolve_same_timestamp_bursts(df):
    data_cols = [c for c in df.columns if c != "Timestamp"]
    df[data_cols] = df.groupby("Timestamp", sort=False)[data_cols].ffill()
    return df.groupby("Timestamp", sort=False, as_index=False).tail(1).reset_index(drop=True)


def normalize_running_mode(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"heat", "heating"}:     return "heat"
    if s in {"cool", "cooling"}:     return "cool"
    if s in {"off", "idle", "none", "false", "0"}: return "off"
    return "unknown"


def inject_virtual_expiration_rows(df):
    gap_sec = df["Timestamp"].diff().dt.total_seconds()
    blackout_mask = gap_sec > (TIME_THRESHOLD_MINUTES * 60)
    if not blackout_mask.any():
        return df

    bl_ts    = df.loc[blackout_mask, "Timestamp"]
    bl_gap   = gap_sec[blackout_mask]
    gap_starts = bl_ts - pd.to_timedelta(bl_gap, unit="s")
    virtual_ts = gap_starts.values + pd.Timedelta(minutes=TIME_THRESHOLD_MINUTES)

    v_rows = pd.DataFrame({
        "Timestamp":            virtual_ts,
        "Equipment_ID":         df.loc[blackout_mask, "Equipment_ID"].values,
        "RunningMode_clean":    "unknown",
        "Setpoint":             NUMERIC_RESET_TOKEN,
        "FanState":             "unknown",
        "is_virtual_expiration": True,
    })
    df = pd.concat([df, v_rows], ignore_index=True)
    return df.sort_values("Timestamp").reset_index(drop=True)


def bounded_time_interpolate(series, times):
    s = pd.to_numeric(series, errors="coerce")
    prev_valid_ts = times.where(s.notna()).ffill()
    next_valid_ts = times.where(s.notna()).bfill()
    total_gap_min = (next_valid_ts - prev_valid_ts).dt.total_seconds() / 60.0
    temp = pd.DataFrame({"t": times, "v": s}).set_index("t")
    interp_vals = temp["v"].interpolate(method="time").values
    ok_mask = s.notna() | (total_gap_min <= TIME_THRESHOLD_MINUTES)
    return pd.Series(np.where(ok_mask, interp_vals, np.nan), index=series.index)


def attach_weather_to_indoor(indoor_df, outdoor_df):
    indoor_ts = indoor_df[["Timestamp"]].copy().sort_values("Timestamp")

    prev_w = pd.merge_asof(
        indoor_ts,
        outdoor_df[["Timestamp", "Outdoor_Temperature", "outsideHumidity"]].rename(
            columns={"Timestamp": "prev_ts",
                     "Outdoor_Temperature": "prev_temp",
                     "outsideHumidity": "prev_hum"}),
        left_on="Timestamp", right_on="prev_ts", direction="backward"
    )
    next_w = pd.merge_asof(
        indoor_ts,
        outdoor_df.rename(
            columns={"Timestamp": "next_ts",
                     "Outdoor_Temperature": "next_temp",
                     "outsideHumidity": "next_hum"}),
        left_on="Timestamp", right_on="next_ts", direction="forward"
    )

    total_gap_min = (next_w["next_ts"] - prev_w["prev_ts"]).dt.total_seconds() / 60.0
    ok_mask = total_gap_min <= TIME_THRESHOLD_MINUTES
    weight  = (
        (indoor_ts["Timestamp"] - prev_w["prev_ts"]).dt.total_seconds()
        / (total_gap_min * 60.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    indoor_df["Outdoor_Temperature"] = np.where(
        ok_mask, prev_w["prev_temp"] * (1 - weight) + next_w["next_temp"] * weight, np.nan)
    indoor_df["outsideHumidity"] = np.where(
        ok_mask, prev_w["prev_hum"] * (1 - weight) + next_w["next_hum"] * weight, np.nan)
    indoor_df["outsideMinTemp"] = next_w["outsideMinTemp"].values
    indoor_df["outsideMaxTemp"] = next_w["outsideMaxTemp"].values
    return indoor_df


def weighted_stats_for_slices(values, weights, prefix):
    keys = ["min", "q25", "median", "q75", "max", "range",
            "mean", "std", "variance", "iqr",
            "skewness", "kurtosis_excess", "raw_moment_2", "raw_moment_3"]
    out = {f"{prefix}_{k}": np.nan for k in keys}

    v = pd.to_numeric(pd.Series(values), errors="coerce").values
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    n = len(v)
    if n == 0:
        return out

    w_sum  = w.sum()
    w_mean = np.dot(v, w) / w_sum
    out[f"{prefix}_min"]          = v.min()
    out[f"{prefix}_max"]          = v.max()
    out[f"{prefix}_range"]        = v.max() - v.min()
    out[f"{prefix}_mean"]         = w_mean
    out[f"{prefix}_raw_moment_2"] = np.dot(v**2, w) / w_sum
    out[f"{prefix}_raw_moment_3"] = np.dot(v**3, w) / w_sum

    sort_idx  = np.argsort(v)
    v_s, w_s  = v[sort_idx], w[sort_idx]
    cum_w_n   = (np.cumsum(w_s) - 0.5 * w_s) / w_sum
    for q, name in [(0.25, "q25"), (0.50, "median"), (0.75, "q75")]:
        out[f"{prefix}_{name}"] = np.interp(q, cum_w_n, v_s)
    out[f"{prefix}_iqr"] = out[f"{prefix}_q75"] - out[f"{prefix}_q25"]

    if n >= 2:
        dev = v - w_mean
        w_var = np.dot(dev**2, w) / (w_sum - w_sum / n)
        out[f"{prefix}_variance"] = w_var
        out[f"{prefix}_std"]      = np.sqrt(w_var)

    if n >= 3 and out.get(f"{prefix}_std", 0) > 0:
        dev = v - w_mean
        m3  = np.dot(dev**3, w) / w_sum
        out[f"{prefix}_skewness"] = m3 / (out[f"{prefix}_std"]**3)

    if n >= 4 and out.get(f"{prefix}_std", 0) > 0:
        dev = v - w_mean
        m4  = np.dot(dev**4, w) / w_sum
        out[f"{prefix}_kurtosis_excess"] = m4 / (out[f"{prefix}_std"]**4) - 3.0

    return out


def aggregate_to_daily(df_event):
    if len(df_event) < 2:
        return pd.DataFrame()

    starts     = df_event["Timestamp"].iloc[:-1].values
    ends       = df_event["Timestamp"].iloc[1:].values
    modes      = df_event["Interval_RunningMode"].iloc[1:].values
    temps      = pd.to_numeric(df_event["Temperature"].iloc[:-1],          errors="coerce").values
    sps        = pd.to_numeric(df_event["Setpoint"].iloc[:-1],             errors="coerce").values
    ot         = pd.to_numeric(df_event["Outdoor_Temperature"].iloc[:-1],  errors="coerce").values
    oh         = pd.to_numeric(df_event["outsideHumidity"].iloc[:-1],      errors="coerce").values
    fan_active = (
        df_event["FanState"].iloc[:-1]
        .astype(str).str.strip().str.lower()
        .isin({"true", "1"}).astype(int).values
    )

    intervals = pd.DataFrame({
        "start": pd.DatetimeIndex(starts), "end": pd.DatetimeIndex(ends),
        "RunningMode": modes, "Temperature": temps, "Setpoint": sps,
        "Outdoor_Temperature": ot, "outsideHumidity": oh,
        "FanState_Active": fan_active,
    }).dropna(subset=["start", "end"])

    state_cols    = ["RunningMode", "Temperature", "Setpoint",
                     "Outdoor_Temperature", "outsideHumidity", "FanState_Active"]
    same_day_mask = intervals["start"].dt.normalize() == intervals["end"].dt.normalize()

    same          = intervals[same_day_mask].copy()
    same["Date"]  = same["start"].dt.date
    same["Duration_Seconds"] = (same["end"] - same["start"]).dt.total_seconds()

    cross_pieces = []
    for _, row in intervals[~same_day_mask].iterrows():
        curr = row["start"]
        while curr < row["end"]:
            nxt      = curr.normalize() + pd.Timedelta(days=1)
            piece_end = min(row["end"], nxt)
            piece = {"Date": curr.date(),
                     "Duration_Seconds": (piece_end - curr).total_seconds()}
            for c in state_cols:
                piece[c] = row[c]
            cross_pieces.append(piece)
            curr = piece_end

    keep = ["Date", "Duration_Seconds"] + state_cols
    res  = (pd.concat([same[keep], pd.DataFrame(cross_pieces)], ignore_index=True)
            if cross_pieces else same[keep].reset_index(drop=True))

    df_event["Date_ext"] = df_event["Timestamp"].dt.date
    daily_rows = []

    for date, grp_slice in res.groupby("Date"):
        grp_ping = df_event[df_event["Date_ext"] == date]

        m_sec    = grp_slice.groupby("RunningMode")["Duration_Seconds"].sum().to_dict()
        h_hrs    = m_sec.get("heat",    0) / 3600.0
        c_hrs    = m_sec.get("cool",    0) / 3600.0
        o_hrs    = m_sec.get("off",     0) / 3600.0
        u_hrs    = m_sec.get("unknown", 0) / 3600.0
        rt_hrs   = h_hrs + c_hrs
        known    = h_hrs + c_hrs + o_hrs

        fan_sec  = (grp_slice["FanState_Active"] * grp_slice["Duration_Seconds"]).sum()
        fan_hrs  = fan_sec / 3600.0
        fan_ratio = fan_hrs / known if known > 0 else 0.0

        def wm(col):
            v = grp_slice[grp_slice[col].notna()]
            s = v["Duration_Seconds"].sum()
            return (v[col] * v["Duration_Seconds"]).sum() / s if s > 0 else np.nan

        wt_temp    = wm("Temperature")
        wt_sp      = wm("Setpoint")
        wt_out_t   = wm("Outdoor_Temperature")
        wt_out_h   = wm("outsideHumidity")

        sp_changes = (grp_ping["Setpoint"].dropna().diff().dropna().ne(0).sum()
                      if "Setpoint" in grp_ping else 0)

        if "Occupied" in grp_ping.columns:
            occ = grp_ping["Occupied"].astype(str).str.strip().str.lower()
            occ_t, occ_f = (occ == "true").sum(), (occ == "false").sum()
        else:
            occ_t, occ_f = 0, 0

        dow = pd.Timestamp(date).dayofweek
        mth = pd.Timestamp(date).month

        day = {
            "Equipment_ID": df_event.iloc[0]["Equipment_ID"],
            "Date": pd.to_datetime(date),
            "daily_heating_hours": h_hrs,
            "daily_cooling_hours": c_hrs,
            "daily_off_hours":     o_hrs,
            "daily_unknown_hours": u_hrs,
            "daily_runtime_hours": rt_hrs,
            "daily_fan_on_hours":  fan_hrs,
            "fan_runtime_ratio":   fan_ratio,
            "setpoint_change_count": sp_changes,
            "occupied_ping_count":   occ_t,
            "unoccupied_ping_count": occ_f,
            "indoor_temp_time_weighted_mean":    wt_temp,
            "setpoint_time_weighted_mean":       wt_sp,
            "outdoor_temp_time_weighted_mean":   wt_out_t,
            "outdoor_humidity_time_weighted_mean": wt_out_h,
            "setpoint_gap_mean": (wt_sp - wt_temp) if pd.notna(wt_sp) and pd.notna(wt_temp) else np.nan,
            "day_of_week":  dow,
            "is_weekend":   1 if dow >= 5 else 0,
            "month":        mth,
            "month_sin":    np.sin(2 * np.pi * mth / 12),
            "month_cos":    np.cos(2 * np.pi * mth / 12),
        }

        for vals, wts, pfx in [
            (grp_slice["Temperature"].values,        grp_slice["Duration_Seconds"].values, "indoor_temp"),
            (grp_slice["Setpoint"].values,           grp_slice["Duration_Seconds"].values, "setpoint"),
            (grp_slice["Outdoor_Temperature"].values, grp_slice["Duration_Seconds"].values, "outdoor_temp"),
            (grp_slice["outsideHumidity"].values,    grp_slice["Duration_Seconds"].values, "outdoor_hum"),
        ]:
            day.update(weighted_stats_for_slices(vals, wts, pfx))

        daily_rows.append(day)

    return pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True)


def compute_outdoor_gradient(temps, window):
    temps = np.asarray(temps, dtype=float)
    grads = np.full(len(temps), np.nan)
    for i in range(window, len(temps)):
        w = temps[i - window:i]
        valid = ~np.isnan(w)
        if valid.sum() < 3:
            continue
        x = np.arange(window, dtype=float)[valid]
        y = w[valid]
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm)**2).sum()
        if denom > 0:
            grads[i] = ((x - xm) * (y - ym)).sum() / denom
    return grads


def preprocess_raw_file(path, equipment_id, outdoor_df):
    """run the full event-level preprocessing on one raw csv."""
    raw = pd.read_csv(path, sep=";", low_memory=False)
    raw = standardize_indoor_columns(raw)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = raw.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    raw["Equipment_ID"] = equipment_id
    raw["raw_row_id"]   = np.arange(len(raw))

    event = resolve_same_timestamp_bursts(raw)
    event["RunningMode_raw"]   = event.get("RunningMode", np.nan)
    event["RunningMode_clean"] = event["RunningMode_raw"].map(normalize_running_mode)
    event = inject_virtual_expiration_rows(event)
    event["RunningMode"] = event["RunningMode_clean"].ffill()
    event["Temperature"] = bounded_time_interpolate(event["Temperature"], event["Timestamp"])

    if "RentalStatus" in event.columns:
        event["RentalStatus"] = event["RentalStatus"].ffill().bfill()

    for c in STEP_LIKE_COLS:
        if c in event.columns:
            if c == "Setpoint":
                event[c] = pd.to_numeric(event[c], errors="coerce")
            event[c] = event[c].ffill()
            if c == "Setpoint":
                event[c] = event[c].replace(NUMERIC_RESET_TOKEN, np.nan)

    event = event.sort_values(["Timestamp", "raw_row_id"], kind="mergesort").reset_index(drop=True)
    event["Interval_RunningMode"] = event["RunningMode"].shift(1).fillna("unknown")
    event = attach_weather_to_indoor(event, outdoor_df)

    return aggregate_to_daily(event)


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================

def load_kaggle_raw_thermostats(outdoor_df, min_months=3):
    """download raw kaggle data, preprocess each file, return list of daily dfs."""
    print("\ndownloading raw thermostat dataset from kaggle...")
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    dataset_path = kagglehub.dataset_download("lsobieski/raw-thermostat-data")
    print(f"  dataset cached at: {dataset_path}")

    # build true daily weather aggregates + gradient (shared across all thermostats)
    outdoor_df["Date"] = pd.to_datetime(outdoor_df["Timestamp"].dt.date)
    true_weather = (
        outdoor_df.groupby("Date")
        .agg(
            true_outside_min=("outsideMinTemp",      "min"),
            true_outside_max=("outsideMaxTemp",      "max"),
            true_outside_mean=("Outdoor_Temperature","mean"),
            true_humidity_mean=("outsideHumidity",   "mean"),
        )
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    true_weather["outdoor_temp_trend_gradient"] = compute_outdoor_gradient(
        true_weather["true_outside_mean"].values, OUTDOOR_TREND_WINDOW
    )

    all_daily = []
    csv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".csv")])
    print(f"  found {len(csv_files)} csv files")

    for i, filename in enumerate(csv_files):
        filepath = os.path.join(dataset_path, filename)
        equipment_id = os.path.splitext(filename)[0]
        try:
            # quick date-range check before full preprocessing
            raw_check = pd.read_csv(filepath, sep=";", low_memory=False,
                                    usecols=["Timestamp"])
            raw_check["Timestamp"] = pd.to_datetime(raw_check["Timestamp"], errors="coerce")
            span_months = (raw_check["Timestamp"].max() - raw_check["Timestamp"].min()).days / 30
            if span_months < min_months:
                print(f"  [{i+1}/{len(csv_files)}] skipping {filename} — {span_months:.1f} months")
                continue

            daily = preprocess_raw_file(filepath, equipment_id, outdoor_df)
            if daily.empty:
                continue

            # merge true weather
            daily["Date"] = pd.to_datetime(daily["Date"])
            daily = pd.merge(daily, true_weather, on="Date", how="left")
            daily["temp_gradient_mean"] = (
                daily["indoor_temp_time_weighted_mean"] - daily["true_outside_mean"]
            )

            # drop diagnostic columns
            daily = daily.drop(columns=[c for c in daily.columns if "count_nonnull" in c],
                               errors="ignore")

            all_daily.append(daily)
            print(f"  [{i+1}/{len(csv_files)}] {filename} → {len(daily)} daily rows")

        except Exception as e:
            print(f"  [{i+1}/{len(csv_files)}] skipping {filename}: {e}")

    print(f"\n  preprocessed {len(all_daily)} thermostats successfully")
    return all_daily


# =============================================================================
# SECTION 3 — LAG FEATURE GENERATION
# =============================================================================

LAG_COLS = [
    "daily_runtime_hours",
    "daily_heating_hours",
    "daily_cooling_hours",
    "indoor_temp_time_weighted_mean",
    "setpoint_time_weighted_mean",
    "setpoint_gap_mean",
    "indoor_temp_std",
]

def add_lag_features(df):
    """generate lag_1..lag_7 for lag-worthy columns, per equipment id."""
    df = df.sort_values(["Equipment_ID", "Date"]).reset_index(drop=True)

    # enforce a continuous daily calendar per thermostat to avoid lag bleed
    grids = []
    for eid, grp in df.groupby("Equipment_ID"):
        dr = pd.date_range(start=grp["Date"].min(), end=grp["Date"].max(), freq="D")
        g  = grp.set_index("Date").reindex(dr).rename_axis("Date").reset_index()
        g["Equipment_ID"] = g["Equipment_ID"].ffill().bfill()
        grids.append(g)
    df = pd.concat(grids, ignore_index=True)

    lag_cols_present = [c for c in LAG_COLS if c in df.columns]
    for i in range(1, LAG_DAYS + 1):
        shifted = df.groupby("Equipment_ID")[lag_cols_present].shift(i)
        shifted.columns = [f"{c}_lag_{i}" for c in lag_cols_present]
        df = pd.concat([df, shifted], axis=1)

    return df


# =============================================================================
# SECTION 4 — THERMOSTAT IDENTITY STATS (no-leakage merge)
# =============================================================================

def add_thermostat_stats(train, test):
    stats = (
        train.groupby("Equipment_ID")[TARGET_COL]
        .agg(thermo_mean_runtime="mean", thermo_std_runtime="std")
        .reset_index()
    )
    train = train.merge(stats, on="Equipment_ID", how="left")
    test  = test.merge(stats,  on="Equipment_ID", how="left")
    return train, test


# =============================================================================
# SECTION 5 — MODEL, EVALUATION, PLOTS
# =============================================================================

def evaluate(y_true, y_pred, label="xgboost"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    cv   = (rmse / y_true.mean()) * 100 if y_true.mean() > 0 else float("nan")
    print(f"\n{'='*50}")
    print(f"  {label} results")
    print(f"{'='*50}")
    print(f"  rmse : {rmse:.4f} hrs")
    print(f"  mae  : {mae:.4f} hrs")
    print(f"  r2   : {r2:.4f}")
    print(f"  cv   : {cv:.1f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "cv_pct": cv}


def run_xgboost(X_train, X_test, y_train, y_test, feature_cols):
    model = xgb.XGBRegressor(
        n_estimators          = 1000,
        learning_rate         = 0.02,
        max_depth             = 8,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        gamma                 = 1,
        objective             = "reg:squarederror",
        random_state          = 42,
        verbosity             = 0,
        early_stopping_rounds = 20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds   = np.clip(model.predict(X_test), 0, None)   # runtime hours >= 0
    metrics = evaluate(y_test, preds)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  top 10 features:")
    print(importance.nlargest(10).to_string())
    return model, preds, metrics


def plot_results(test_df, y_test, preds, model, metrics, feature_cols):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("xgboost — daily hvac runtime prediction", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(test_df["Date"].values, y_test.values, label="actual",    alpha=0.7, lw=1.2)
    ax.plot(test_df["Date"].values, preds,         label="predicted", alpha=0.7, lw=1.2, ls="--")
    ax.set_title("actual vs predicted runtime (test set)")
    ax.set_xlabel("date"); ax.set_ylabel("runtime (hrs/day)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax = axes[0, 1]
    ax.scatter(y_test, preds, alpha=0.4, s=15, color="steelblue")
    lim = [0, max(y_test.max(), preds.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"actual vs predicted scatter  (r2={metrics['r2']:.3f})")
    ax.set_xlabel("actual (hrs)"); ax.set_ylabel("predicted (hrs)"); ax.legend()

    ax = axes[1, 0]
    residuals = np.array(y_test) - preds
    ax.plot(test_df["Date"].values, residuals, alpha=0.6, lw=0.8, color="coral")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title(f"residuals  (rmse={metrics['rmse']:.4f} hrs)")
    ax.set_xlabel("date"); ax.set_ylabel("actual − predicted (hrs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax = axes[1, 1]
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("feature importance"); ax.set_xlabel("score")

    plt.tight_layout()
    plt.savefig("xgb_results_two/xgboost_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_r2_histogram(test):
    r2_values = []
    for eid in test["Equipment_ID"].unique():
        te = test[test["Equipment_ID"] == eid]
        if len(te) < 10:
            continue
        r2_values.append(r2_score(te[TARGET_COL], te["predicted"]))

    r2_values = np.array(r2_values)
    mean_r2, median_r2 = np.mean(r2_values), np.median(r2_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(r2_values, bins=20, edgecolor="white", lw=0.8)
    for patch, val in zip(patches, bins):
        patch.set_facecolor("#D85A30" if val < 0 else "#378ADD")
    ax.axvline(mean_r2,   color="black", lw=1.5, ls="--", label=f"mean   = {mean_r2:.3f}")
    ax.axvline(median_r2, color="gray",  lw=1.5, ls=":",  label=f"median = {median_r2:.3f}")
    ax.axvline(0, color="red", lw=1.0, ls="-", alpha=0.4)
    ax.set_xlabel("r2 (per thermostat)"); ax.set_ylabel("count")
    ax.set_title("distribution of r2 across all thermostats (pooled model)")
    ax.legend()
    ax.text(0.98, 0.97,
            f"n={len(r2_values)}  |  r2≥0.3: {(r2_values>=0.3).sum()}  |  r2<0: {(r2_values<0).sum()}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right", color="gray")
    plt.tight_layout()
    plt.savefig("xgb_results_two/r2_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_per_thermostat(train, test, feature_cols):
    print("\n  per-thermostat results (individual models):")
    print(f"  {'id':>6}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'train':>6}  {'test':>5}")
    print("  " + "-"*44)
    for eid in sorted(train["Equipment_ID"].unique()):
        tr = train[train["Equipment_ID"] == eid]
        te = test[test["Equipment_ID"] == eid]
        if len(te) < 10 or len(tr) < 40:
            continue
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="reg:squarederror", random_state=42, verbosity=0)
        m.fit(tr[feature_cols], tr[TARGET_COL])
        preds = np.clip(m.predict(te[feature_cols]), 0, None)
        r2   = r2_score(te[TARGET_COL], preds)
        rmse = np.sqrt(mean_squared_error(te[TARGET_COL], preds))
        mae  = mean_absolute_error(te[TARGET_COL], preds)
        print(f"  {str(eid)[:6]:>6}  {r2:>6.3f}  {rmse:>6.4f}  {mae:>6.4f}  {len(tr):>6}  {len(te):>5}")


def run_per_month(test):
    print("\n  per-month results (pooled model):")
    print(f"  {'month':>10}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'rows':>6}")
    print("  " + "-"*44)
    test = test.copy()
    test["month_label"] = test["Date"].dt.to_period("M").astype(str)
    for month in sorted(test["month_label"].unique()):
        sub = test[test["month_label"] == month]
        if len(sub) < 10:
            continue
        r2   = r2_score(sub[TARGET_COL], sub["predicted"])
        rmse = np.sqrt(mean_squared_error(sub[TARGET_COL], sub["predicted"]))
        mae  = mean_absolute_error(sub[TARGET_COL], sub["predicted"])
        print(f"  {month:>10}  {r2:>6.3f}  {rmse:>6.4f}  {mae:>6.4f}  {len(sub):>6}")


def run_time_series_cv(combined, feature_cols, n_splits=5):
    combined = combined.sort_values("Date").reset_index(drop=True)
    time_vals = combined["Date"]
    quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
    cutoffs   = [time_vals.quantile(q) for q in quantiles]

    print(f"\n{'='*60}")
    print(f"  time-series cross-validation  ({n_splits} folds)")
    print(f"{'='*60}")
    print(f"  {'fold':>4}  {'train':>8}  {'test':>7}  {'r2':>6}  {'rmse':>7}  {'mae':>7}")
    print("  " + "-"*50)

    fold_results = []
    for fold in range(n_splits):
        train_cut  = cutoffs[fold]
        test_start = cutoffs[fold]
        test_end   = cutoffs[fold + 1] if fold + 1 < len(cutoffs) else time_vals.max()

        tr = combined[combined["Date"] < train_cut].copy()
        te = combined[(combined["Date"] >= test_start) & (combined["Date"] <= test_end)].copy()

        if len(tr) < 100 or len(te) < 10:
            continue

        stats = (tr.groupby("Equipment_ID")[TARGET_COL]
                 .agg(thermo_mean_runtime="mean", thermo_std_runtime="std")
                 .reset_index())
        # drop any pre-existing stat columns to avoid _x/_y suffixes on re-merge
        for _col in ["thermo_mean_runtime", "thermo_std_runtime"]:
            tr = tr.drop(columns=[_col], errors="ignore")
            te = te.drop(columns=[_col], errors="ignore")
        tr = tr.merge(stats, on="Equipment_ID", how="left")
        te = te.merge(stats, on="Equipment_ID", how="left")
        # drop test rows whose thermostat never appeared in training
        te = te.dropna(subset=["thermo_mean_runtime"])

        avail = [f for f in feature_cols if f in tr.columns and f in te.columns]
        # only drop on target — xgboost handles feature NaNs natively
        tr_clean = tr.dropna(subset=[TARGET_COL])
        te_clean = te.dropna(subset=[TARGET_COL])

        m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.02, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=1, objective="reg:squarederror", random_state=42,
            verbosity=0, early_stopping_rounds=20,
        )
        m.fit(tr_clean[avail], tr_clean[TARGET_COL],
              eval_set=[(te_clean[avail], te_clean[TARGET_COL])], verbose=False)

        preds = np.clip(m.predict(te_clean[avail]), 0, None)
        r2    = r2_score(te_clean[TARGET_COL], preds)
        rmse  = np.sqrt(mean_squared_error(te_clean[TARGET_COL], preds))
        mae   = mean_absolute_error(te_clean[TARGET_COL], preds)

        fold_results.append({"fold": fold+1, "train": len(tr_clean), "test": len(te_clean),
                              "r2": r2, "rmse": rmse, "mae": mae})
        print(f"  {fold+1:>4}  {len(tr_clean):>8}  {len(te_clean):>7}  {r2:>6.3f}  {rmse:>7.4f}  {mae:>7.4f}")

    if fold_results:
        rdf = pd.DataFrame(fold_results)
        print("\n  summary across folds:")
        for metric in ["r2", "rmse", "mae"]:
            v = rdf[metric]
            print(f"  {metric:>5}: {v.mean():.4f} ± {v.std():.4f}")
        r2_range = rdf["r2"].max() - rdf["r2"].min()
        stability = ("stable" if r2_range < 0.05
                     else "moderate variance" if r2_range < 0.15
                     else "high variance — single split may be unreliable")
        print(f"  r2 range = {r2_range:.3f} → {stability}")

    return pd.DataFrame(fold_results)



# =============================================================================
# SECTION 6b — OPTUNA HYPERPARAMETER TUNING
# =============================================================================

def tune_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """
    bayesian hyperparameter search using optuna.
    uses the same time-ordered train/val split as the main model.
    returns the best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  optuna not installed — run: pip install optuna")
        return {}

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 5, 50),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
            "objective":         "reg:squarederror",
            "random_state":      42,
            "verbosity":         0,
            "early_stopping_rounds": 20,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.clip(m.predict(X_val), 0, None)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  optuna best params ({n_trials} trials):")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"  best val rmse: {study.best_value:.4f} hrs")
    return best


# =============================================================================
# SECTION 7 — PER-THERMOSTAT DIAGNOSTIC REPORT
# =============================================================================

def thermostat_diagnostic_report(combined, train, test, feature_cols):
    """
    for every thermostat, compute:
      - runtime stats (mean, std, cv, pct_zero)  -> flags low-variance units
      - pooled model r2 on test rows
      - individual model r2 (same as run_per_thermostat)
      - data quality: total days, train days, test days, missing day pct
      - classification: GOOD / LOW_VARIANCE / DATA_SPARSE / UNPREDICTABLE

    prints a ranked table and a scatter plot of runtime_std vs r2.
    """
    print("\n" + "="*80)
    print("  per-thermostat diagnostic report")
    print("="*80)

    # ── runtime variance stats from the full combined dataset ──────────────
    runtime_stats = (
        combined.groupby("Equipment_ID")[TARGET_COL]
        .agg(
            rt_mean="mean",
            rt_std="std",
            rt_min="min",
            rt_max="max",
            rt_pct_zero=lambda x: (x == 0).mean() * 100,
            total_days="count",
        )
        .reset_index()
    )
    runtime_stats["rt_cv"] = runtime_stats["rt_std"] / runtime_stats["rt_mean"].replace(0, np.nan)

    # ── train / test day counts ─────────────────────────────────────────────
    train_counts = train.groupby("Equipment_ID").size().rename("train_days")
    test_counts  = test.groupby("Equipment_ID").size().rename("test_days")

    # ── pooled model r2 per thermostat (already predicted) ─────────────────
    pooled_r2 = {}
    for eid in test["Equipment_ID"].unique():
        te = test[test["Equipment_ID"] == eid]
        if len(te) < 5 or "predicted" not in te.columns:
            continue
        if te[TARGET_COL].std() < 0.01:
            pooled_r2[eid] = np.nan   # undefined for flat series
        else:
            pooled_r2[eid] = r2_score(te[TARGET_COL], te["predicted"])
    pooled_r2_s = pd.Series(pooled_r2, name="pooled_r2")

    # ── individual model r2 per thermostat ─────────────────────────────────
    indiv_r2 = {}
    for eid in sorted(train["Equipment_ID"].unique()):
        tr = train[train["Equipment_ID"] == eid]
        te = test[test["Equipment_ID"] == eid]
        if len(tr) < 40 or len(te) < 10:
            continue
        if te[TARGET_COL].std() < 0.01:
            indiv_r2[eid] = np.nan
            continue
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0
        )
        m.fit(tr[feature_cols], tr[TARGET_COL])
        preds = np.clip(m.predict(te[feature_cols]), 0, None)
        indiv_r2[eid] = r2_score(te[TARGET_COL], preds)
    indiv_r2_s = pd.Series(indiv_r2, name="indiv_r2")

    # ── assemble report dataframe ───────────────────────────────────────────
    report = (
        runtime_stats
        .set_index("Equipment_ID")
        .join(train_counts)
        .join(test_counts)
        .join(pooled_r2_s)
        .join(indiv_r2_s)
        .reset_index()
    )

    # ── classify each thermostat ────────────────────────────────────────────
    def classify(row):
        if row["rt_std"] < 0.5:
            return "LOW_VARIANCE"          # runtime barely moves — r2 is meaningless
        if row.get("train_days", 0) < 100:
            return "DATA_SPARSE"           # not enough history to learn
        if pd.isna(row["pooled_r2"]):
            return "UNDEFINED"
        if row["pooled_r2"] >= 0.5:
            return "GOOD"
        if row["pooled_r2"] >= 0.2:
            return "MODERATE"
        return "UNPREDICTABLE"

    report["category"] = report.apply(classify, axis=1)

    # ── print ranked table ──────────────────────────────────────────────────
    report_sorted = report.sort_values("pooled_r2", ascending=False, na_position="last")
    header = (f"  {'equipment_id':<45}  {'cat':<14}  {'rt_mean':>7}  "
              f"{'rt_std':>6}  {'rt_cv':>5}  {'pct0':>5}  "
              f"{'train':>5}  {'test':>4}  {'pool_r2':>7}  {'indiv_r2':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in report_sorted.iterrows():
        pr2 = f"{row['pooled_r2']:>7.3f}" if pd.notna(row["pooled_r2"]) else f"{'n/a':>7}"
        ir2 = f"{row['indiv_r2']:>8.3f}" if pd.notna(row["indiv_r2"]) else f"{'n/a':>8}"
        cv  = f"{row['rt_cv']:>5.2f}"    if pd.notna(row["rt_cv"])    else f"{'n/a':>5}"
        print(
            f"  {str(row['Equipment_ID']):<45}  {row['category']:<14}  "
            f"{row['rt_mean']:>7.2f}  {row['rt_std']:>6.2f}  {cv}  "
            f"{row['rt_pct_zero']:>5.1f}  "
            f"{int(row['train_days']) if pd.notna(row['train_days']) else 0:>5}  "
            f"{int(row['test_days'])  if pd.notna(row['test_days'])  else 0:>4}  "
            f"{pr2}  {ir2}"
        )

    # ── category summary ────────────────────────────────────────────────────
    print("\n  category summary:")
    for cat, grp in report.groupby("category"):
        pr2_vals = grp["pooled_r2"].dropna()
        mean_pr2 = f"{pr2_vals.mean():.3f}" if len(pr2_vals) else "n/a"
        print(f"    {cat:<14}: {len(grp):>3} thermostats  |  mean pooled_r2 = {mean_pr2}")

    # ── scatter plot: runtime std vs pooled r2 ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("thermostat diagnostic", fontsize=13, fontweight="bold")

    ax = axes[0]
    palette = {
        "GOOD": "#378ADD", "MODERATE": "#6CB86C",
        "LOW_VARIANCE": "#AAAAAA", "DATA_SPARSE": "#E8A838",
        "UNPREDICTABLE": "#D85A30", "UNDEFINED": "#CCCCCC",
    }
    for cat, grp in report.dropna(subset=["pooled_r2"]).groupby("category"):
        ax.scatter(grp["rt_std"], grp["pooled_r2"],
                   label=cat, color=palette.get(cat, "gray"),
                   alpha=0.8, s=50, edgecolors="white", linewidths=0.4)
    ax.axhline(0,   color="red",   lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.5, color="green", lw=0.8, ls="--", alpha=0.5, label="r2=0.5 threshold")
    ax.axvline(0.5, color="gray",  lw=0.8, ls=":",  alpha=0.5, label="std=0.5 threshold")
    ax.set_xlabel("runtime std (hrs/day)")
    ax.set_ylabel("pooled model r2")
    ax.set_title("runtime variability vs predictability")
    ax.legend(fontsize=8)

    ax = axes[1]
    cat_order  = ["GOOD", "MODERATE", "UNPREDICTABLE", "DATA_SPARSE", "LOW_VARIANCE"]
    cat_counts = report["category"].value_counts().reindex(cat_order, fill_value=0)
    colors     = [palette.get(c, "gray") for c in cat_order]
    bars = ax.barh(cat_order, cat_counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_xlabel("number of thermostats")
    ax.set_title("thermostat category breakdown")
    ax.set_xlim(0, cat_counts.max() * 1.2)

    plt.tight_layout()
    plt.savefig("thermostat_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()

    return report

# =============================================================================
# SECTION 6 — MAIN
# =============================================================================

def main():
    # load outdoor weather
    print("loading outdoor weather...")
    outdoor_df = process_outdoor_weather(OUTDOOR_PATH)
    print(f"  {len(outdoor_df)} outdoor rows from {outdoor_df['Timestamp'].min().date()} "
          f"to {outdoor_df['Timestamp'].max().date()}")

    # preprocess all raw thermostat files from kaggle
    all_daily = load_kaggle_raw_thermostats(outdoor_df, min_months=3)

    if not all_daily:
        print("no thermostats survived preprocessing — check paths and column names.")
        return

    # combine and add lag features
    combined = pd.concat(all_daily, ignore_index=True)
    combined = add_lag_features(combined)
    combined = combined.dropna(subset=[TARGET_COL])
    print(f"\n  combined daily dataset: {len(combined)} rows, "
          f"{combined['Equipment_ID'].nunique()} thermostats")

    # time-based train/test split (80/20 by date)
    cutoff = combined["Date"].quantile(0.8)
    print(f"  global train/test cutoff: {cutoff.date()}")

    train = combined[combined["Date"] <= cutoff].copy()
    test  = combined[combined["Date"] >  cutoff].copy()
    print(f"  train: {len(train)} rows | test: {len(test)} rows")

    # add thermostat identity stats (derived from train only)
    train, test = add_thermostat_stats(train, test)

    avail_features = [f for f in FEATURE_COLS if f in train.columns]
    missing = set(FEATURE_COLS) - set(avail_features)
    if missing:
        print(f"\n  note: {len(missing)} feature(s) not found and will be skipped: {missing}")

    # only drop rows where the TARGET is null — xgboost handles feature NaNs natively
    # (mirrors the original lightgbm pipeline which also only dropna on the target)
    train_clean = train.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    test_clean  = test.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  after dropping nulls — train: {len(train_clean)} | test: {len(test_clean)}")

    X_train, y_train = train_clean[avail_features], train_clean[TARGET_COL]
    X_test,  y_test  = test_clean[avail_features],  test_clean[TARGET_COL]

    # time-series cross validation (stats computed per-fold inside, no pre-merge)
    run_time_series_cv(combined.copy(), avail_features, n_splits=5)

    # optuna hyperparameter tuning (set n_trials=0 to skip)
    N_OPTUNA_TRIALS = 50
    if N_OPTUNA_TRIALS > 0:
        # use a validation slice from train for tuning (last 20% by date)
        tune_cut = train_clean['Date'].quantile(0.8)
        tune_tr  = train_clean[train_clean['Date'] <= tune_cut]
        tune_val = train_clean[train_clean['Date'] >  tune_cut]
        best_params = tune_xgboost_optuna(
            tune_tr[avail_features],  tune_tr[TARGET_COL],
            tune_val[avail_features], tune_val[TARGET_COL],
            n_trials=N_OPTUNA_TRIALS
        )
    else:
        best_params = {}

    # final pooled model — use tuned params if available, else defaults
    def run_xgboost_with_params(X_tr, X_te, y_tr, y_te, feature_cols, extra_params=None):
        base = dict(
            n_estimators=1000, learning_rate=0.02, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=1, objective='reg:squarederror', random_state=42,
            verbosity=0, early_stopping_rounds=20,
        )
        if extra_params:
            base.update(extra_params)
        m = xgb.XGBRegressor(**base)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        preds = np.clip(m.predict(X_te), 0, None)
        return m, preds, evaluate(y_te, preds)

    model, preds, metrics = run_xgboost_with_params(
        X_train, X_test, y_train, y_test, avail_features, best_params
    )
    test_clean["predicted"] = preds

    # plots and per-thermostat/month breakdowns
    plot_results(test_clean, y_test, preds, model, metrics, avail_features)
    plot_r2_histogram(test_clean)
    run_per_month(test_clean)
    run_per_thermostat(train_clean, test_clean, avail_features)
    thermostat_diagnostic_report(combined, train_clean, test_clean, avail_features)


if __name__ == "__main__":
    main()