# linear regression

# gradient boosted tree
# xgboost

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# data loading
def load_and_merge(inside_path, outside_path):
    # inside data
    inside = pd.read_csv(inside_path, parse_dates=["Timestamp"])
    inside = inside.sort_values("Timestamp").drop_duplicates("Timestamp")

    # outside data
    outside = pd.read_csv(outside_path, parse_dates=["Timestamp"])
    outside = outside.sort_values("Timestamp").drop_duplicates("Timestamp")

    # unified 15 min timelines
    start = max(inside["Timestamp"].min(), outside["Timestamp"].min())
    end   = min(inside["Timestamp"].max(), outside["Timestamp"].max())
    timeline = pd.date_range(start=start, end=end, freq="15min")

    # reindex
    inside  = inside.set_index("Timestamp").reindex(timeline).interpolate("time")
    outside = outside.set_index("Timestamp").reindex(timeline).interpolate("time")

    df = inside.join(outside, rsuffix="_out")
    df.index.name = "Timestamp"
    df = df.reset_index()
    return df

# deriving binary 'is_running" from outputstate, then compute runtime minutes per clock-hour (target)
def compute_runtime_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    # OutputState == 1  →  unit is actively conditioning (heat OR cool).
    # OutputState == 0  →  standby / off.

    # binary running signal
    df["is_running"] = (df["OutputState"] == 1).astype(int)

    # each 15-min row = 15 min of potential runtime
    df["runtime_minutes"] = df["is_running"] * 15

    # aggregate to hourly runtime
    df["hour_bucket"] = df["Timestamp"].dt.floor("h")
    hourly = (
        df.groupby("hour_bucket")
          .agg(
              runtime_minutes   = ("runtime_minutes",    "sum"),
              Temperature       = ("Temperature",        "mean"),
              Setpoint          = ("Setpoint",           "mean"),
              Outdoor_Temp      = ("outside_temp",       "mean"),
              Outdoor_Temp_Min  = ("outside_temp_min",   "mean"),
              Outdoor_Temp_Max  = ("outside_temp_max",   "mean"),
              Outdoor_Humidity  = ("outside_humidity",   "mean"),
          )
          .reset_index()
    )
    return hourly

# add time-based and physics-inspired features.
def engineer_features(df):
    df = df.copy()
    df["hour_of_day"]   = df["hour_bucket"].dt.hour
    df["day_of_week"]   = df["hour_bucket"].dt.dayofweek
    df["month"]         = df["hour_bucket"].dt.month

    # temp delta: how hard the HVAC has to work
    df["temp_delta"]    = df["Setpoint"] - df["Outdoor_Temp"]
    df["temp_range"]    = df["Outdoor_Temp_Max"] - df["Outdoor_Temp_Min"]

    # lag features (how long did it run the previous hour?)
    df["runtime_lag1"]  = df["runtime_minutes"].shift(1)
    df["runtime_lag24"] = df["runtime_minutes"].shift(24)   # same hour yesterday

    # rolling mean (recent trend)
    df["runtime_roll3"] = df["runtime_minutes"].shift(1).rolling(3).mean()

    df = df.dropna()   # drop rows where lag features are NaN
    return df

# feature and target split
FEATURE_COLS = [
    "Temperature", "Setpoint", "Outdoor_Temp",
    "Outdoor_Temp_Min", "Outdoor_Temp_Max", "Outdoor_Humidity",
    "hour_of_day", "day_of_week", "month",
    "temp_delta", "temp_range",
    "runtime_lag1", "runtime_lag24", "runtime_roll3",
]
TARGET_COL = "runtime_minutes"

# evaluate
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    cv   = (rmse / y_true.mean()) * 100 if y_true.mean() > 0 else float("nan")
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  RMSE : {rmse:.2f} min  (target < 5 min/hr)")
    print(f"  MAE  : {mae:.2f} min")
    print(f"  R-squared   : {r2:.3f}")
    print(f"  CV   : {cv:.1f}%      (target < 30% per ASHRAE)")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "cv_pct": cv}

# linear baseline function
def run_linear_baseline(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_tr, y_train)
    preds = np.clip(model.predict(X_te), 0, 60)   # runtime can't exceed 60 min/hr
    return evaluate("Linear Regression (Baseline)", y_test, preds), model


def main(inside_path="../../Sample_Data/Processed/processed_inside.csv", outside_path="outsideweather.csv"):
    df = load_and_merge(inside_path, outside_path)
    df = compute_runtime_per_hour(df)
    df = engineer_features(df)

    print(f"Dataset shape: {df.shape} | Runtime Range: {df[TARGET_COL].min():.0f}–{df[TARGET_COL].max():.0f} min/hr")
    # chronological train/test split (80/20)
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]

    print(f"\nTrain: {len(train)} rows | Test: {len(test)} rows")

    # run models
    results = []
    lr_metrics, _ = run_linear_baseline(X_train, X_test, y_train, y_test)

    results.append(lr_metrics)

    # summary table
    print("\n")
    print("model comparison summary\n")
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string())

if __name__ == "__main__":
    import sys
    inside_path  = sys.argv[1] if len(sys.argv) > 1 else "inside.csv"
    outside_path = sys.argv[2] if len(sys.argv) > 2 else "outside.csv"
    main(inside_path, outside_path)