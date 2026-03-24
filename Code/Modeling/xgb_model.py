# gradient boosted tree (xgboost) for hvac runtime prediction
# target: runtime minutes per hour (0-60)
# features: indoor temp, setpoint, outdoor weather, time, lag features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# data loading 

def load_data(inside_path):
    df = pd.read_csv(inside_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp")

    # drop exact duplicate rows first (same timestamp + same values)
    df = df.drop_duplicates()

    # then deduplicate on state change — only keep a row when something actually changed
    # this is what the thermostat is actually recording: state transitions
    state_cols = ["OutputState", "RunningMode", "FanState", "Setpoint"]
    df = df[df[state_cols].ne(df[state_cols].shift()).any(axis=1)]

    df = df.reset_index(drop=True)
    return df


# runtime aggregation

def compute_runtime_per_hour(df):
    df = df.copy()

    # duration of each state = time until next state change
    df["duration_minutes"] = (
        df["Timestamp"].shift(-1) - df["Timestamp"]
    ).dt.total_seconds() / 60

    # cap each segment at 60 min (sanity check for gaps in data)
    df["duration_minutes"] = df["duration_minutes"].clip(0, 60)

    # only count duration when unit is actively running
    df["runtime_minutes"] = df["duration_minutes"] * (df["OutputState"] == 1).astype(int)

    df["hour_bucket"] = df["Timestamp"].dt.floor("h")
    hourly = (
        df.groupby("hour_bucket")
          .agg(
              runtime_minutes  = ("runtime_minutes",    "sum"),
              Temperature      = ("Temperature",        "mean"),
              Setpoint         = ("Setpoint",           "mean"),
              Outdoor_Temp     = ("Outdoor_Temperature","mean"),
              Outdoor_Temp_Min = ("outsideMinTemp",     "mean"),
              Outdoor_Temp_Max = ("outsideMaxTemp",     "mean"),
              Outdoor_Humidity = ("outsideHumidity",    "mean"),
          )
          .reset_index()
    )

    # still cap at 60 just in case
    hourly["runtime_minutes"] = hourly["runtime_minutes"].clip(0, 60)
    return hourly


# feature engineering

def engineer_features(df):
    df = df.copy()

    # time features
    df["hour_of_day"] = df["hour_bucket"].dt.hour
    df["day_of_week"] = df["hour_bucket"].dt.dayofweek
    df["month"]       = df["hour_bucket"].dt.month

    # physics-based: how hard does the hvac have to work?
    df["temp_delta"]  = df["Setpoint"] - df["Outdoor_Temp"]
    df["temp_range"]  = df["Outdoor_Temp_Max"] - df["Outdoor_Temp_Min"]

    # lag features — runtime from previous hours
    df["runtime_lag1"]  = df["runtime_minutes"].shift(1)   # 1 hour ago
    df["runtime_lag24"] = df["runtime_minutes"].shift(24)  # same hour yesterday

    # rolling average over last 3 hours (recent trend)
    df["runtime_roll3"] = df["runtime_minutes"].shift(1).rolling(3).mean()

    df = df.dropna()
    return df


# config

FEATURE_COLS = [
    "Temperature", "Setpoint", "Outdoor_Temp",
    "Outdoor_Temp_Min", "Outdoor_Temp_Max", "Outdoor_Humidity",
    "hour_of_day", "day_of_week", "month",
    "temp_delta", "temp_range",
    "runtime_lag1", "runtime_lag24", "runtime_roll3",
]
TARGET_COL = "runtime_minutes"


# evaluation

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    cv   = (rmse / y_true.mean()) * 100 if y_true.mean() > 0 else float("nan")

    print(f"\n{'='*50}")
    print(f"  xgboost results")
    print(f"{'='*50}")
    print(f"  rmse : {rmse:.2f} min  (target < 5 min/hr)")
    print(f"  mae  : {mae:.2f} min")
    print(f"  r2   : {r2:.3f}")
    print(f"  cv   : {cv:.1f}%      (target < 30% per ashrae)")
    return {"rmse": rmse, "mae": mae, "r2": r2, "cv_pct": cv}


# xgboost model

def run_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        n_estimators     = 300,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "reg:squarederror",
        random_state     = 42,
        verbosity        = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False,
    )

    preds = np.clip(model.predict(X_test), 0, 60)
    metrics = evaluate(y_test, preds)

    # top features
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\n  top 5 features:")
    print(importance.nlargest(5).to_string())

    return model, preds, metrics


# plots

def plot_results(test_df, y_test, preds, model, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("xgboost — hvac runtime prediction", fontsize=14, fontweight="bold", y=1.01)

    # plot actual vs predicted over time
    ax = axes[0, 0]
    ax.plot(test_df["hour_bucket"].values, y_test.values, label="actual", alpha=0.7, linewidth=1.2)
    ax.plot(test_df["hour_bucket"].values, preds,         label="predicted", alpha=0.7, linewidth=1.2, linestyle="--")
    ax.set_title("actual vs predicted runtime (test set)")
    ax.set_xlabel("date")
    ax.set_ylabel("runtime (min/hr)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # plot scatter actual vs predicted
    ax = axes[0, 1]
    ax.scatter(y_test, preds, alpha=0.4, s=15, color="steelblue")
    lims = [0, 60]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"actual vs predicted scatter  (r² = {metrics['r2']:.3f})")
    ax.set_xlabel("actual runtime (min/hr)")
    ax.set_ylabel("predicted runtime (min/hr)")
    ax.legend()

    # plot residuals over time
    ax = axes[1, 0]
    residuals = np.array(y_test) - preds
    ax.plot(test_df["hour_bucket"].values, residuals, alpha=0.6, linewidth=0.8, color="coral")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"residuals over time  (rmse = {metrics['rmse']:.2f} min)")
    ax.set_xlabel("date")
    ax.set_ylabel("actual − predicted (min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # plot feature importance 
    ax = axes[1, 1]
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("feature importance")
    ax.set_xlabel("importance score")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
    print("\n  plot saved → xgboost_results.png")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main(inside_path="../../Sample_Data/Processed/processed_inside.csv"):
    print("loading data...")
    df = load_data(inside_path)

    print("computing hourly runtime...")
    df = compute_runtime_per_hour(df)

    print("engineering features...")
    df = engineer_features(df)

    print(f"dataset shape: {df.shape} | runtime range: {df[TARGET_COL].min():.0f}–{df[TARGET_COL].max():.0f} min/hr")

    # chronological 80/20 split — no shuffling, respects time order
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]

    print(f"train: {len(train)} rows | test: {len(test)} rows")

    # run xgboost
    model, preds, metrics = run_xgboost(X_train, X_test, y_train, y_test)

    # generate plots
    plot_results(test, y_test, preds, model, metrics)


if __name__ == "__main__":
    import sys
    inside_path = sys.argv[1] if len(sys.argv) > 1 else "../../Sample_Data/Processed/processed_inside.csv"
    main(inside_path)