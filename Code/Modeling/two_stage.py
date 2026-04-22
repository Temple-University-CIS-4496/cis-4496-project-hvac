# gradient boosted tree (xgboost) for hvac runtime prediction
# target: runtime minutes per hour (0-60)
# features: indoor temp, setpoint, outdoor weather, time, lag features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, roc_auc_score, confusion_matrix
)
import warnings
import kagglehub
import os
warnings.filterwarnings("ignore")

KAGGLE_TOKEN = "KGAT_6d26ebf584b9d44052424f267dffec66"

# data loading
def load_data(inside_path):
    df = pd.read_csv(inside_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates()

    if "RunningMode" not in df.columns:
        if "RunningMode_cool" in df.columns:
            df["RunningMode"] = np.select(
                [df["RunningMode_cool"] == 1, df["RunningMode_heat"] == 1],
                ["cool", "heat"],
                default="off"
            )
        else:
            df["RunningMode"] = "unknown"

    state_cols = ["OutputState", "RunningMode", "FanState", "Setpoint"]
    state_cols = [c for c in state_cols if c in df.columns]
    df = df[df[state_cols].ne(df[state_cols].shift()).any(axis=1)]
    df = df.reset_index(drop=True)
    return df


# runtime aggregation
def compute_runtime_per_hour(df):
    df = df.copy()
    df["duration_minutes"] = (
        df["Timestamp"].shift(-1) - df["Timestamp"]
    ).dt.total_seconds() / 60
    df["duration_minutes"] = df["duration_minutes"].clip(0, 60)
    df["runtime_minutes"]  = df["duration_minutes"] * (df["OutputState"] == 1).astype(int)
    df["hour_bucket"]      = df["Timestamp"].dt.floor("h")

    hourly = (
        df.groupby("hour_bucket")
          .agg(
              runtime_minutes  = ("runtime_minutes",     "sum"),
              Temperature      = ("Temperature",         "mean"),
              Setpoint         = ("Setpoint",            "mean"),
              Outdoor_Temp     = ("Outdoor_Temperature", "mean"),
              Outdoor_Temp_Min = ("outsideMinTemp",      "mean"),
              Outdoor_Temp_Max = ("outsideMaxTemp",      "mean"),
              Outdoor_Humidity = ("outsideHumidity",     "mean"),
          )
          .reset_index()
    )
    hourly["runtime_minutes"] = hourly["runtime_minutes"].clip(0, 60)
    return hourly


# feature engineering
def engineer_features(df):
    df = df.copy()
    df["hour_of_day"]   = df["hour_bucket"].dt.hour
    df["day_of_week"]   = df["hour_bucket"].dt.dayofweek
    df["month"]         = df["hour_bucket"].dt.month
    df["temp_delta"]    = df["Setpoint"] - df["Outdoor_Temp"]
    df["temp_range"]    = df["Outdoor_Temp_Max"] - df["Outdoor_Temp_Min"]
    df["runtime_lag1"]  = df["runtime_minutes"].shift(1)
    df["runtime_lag24"] = df["runtime_minutes"].shift(24)
    df["runtime_roll3"] = df["runtime_minutes"].shift(1).rolling(3).mean()
    df = df.dropna()
    return df


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
def run_xgboost(X_train, X_test, y_train, y_test, feature_cols):
    model = xgb.XGBRegressor(
        n_estimators     = 1000,
        learning_rate    = 0.02,
        max_depth        = 8,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 10,
        gamma            = 1,
        objective        = "reg:squarederror",
        random_state     = 42,
        verbosity        = 0,
        early_stopping_rounds = 20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds   = np.clip(model.predict(X_test), 0, 60)
    metrics = evaluate(y_test, preds)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  top 5 features:")
    print(importance.nlargest(5).to_string())

    return model, preds, metrics


# plots
def plot_results(test_df, y_test, preds, model, metrics, feature_cols):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("xgboost — hvac runtime prediction", fontsize=14, fontweight="bold", y=1.01)

    ax = axes[0, 0]
    ax.plot(test_df["hour_bucket"].values, y_test.values, label="actual",    alpha=0.7, linewidth=1.2)
    ax.plot(test_df["hour_bucket"].values, preds,         label="predicted", alpha=0.7, linewidth=1.2, linestyle="--")
    ax.set_title("actual vs predicted runtime (test set)")
    ax.set_xlabel("date"); ax.set_ylabel("runtime (min/hr)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax = axes[0, 1]
    ax.scatter(y_test, preds, alpha=0.4, s=15, color="steelblue")
    lims = [0, 60]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"actual vs predicted scatter  (r2 = {metrics['r2']:.3f})")
    ax.set_xlabel("actual runtime (min/hr)"); ax.set_ylabel("predicted runtime (min/hr)"); ax.legend()

    ax = axes[1, 0]
    residuals = np.array(y_test) - preds
    ax.plot(test_df["hour_bucket"].values, residuals, alpha=0.6, linewidth=0.8, color="coral")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"residuals over time  (rmse = {metrics['rmse']:.2f} min)")
    ax.set_xlabel("date"); ax.set_ylabel("actual − predicted (min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax = axes[1, 1]
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("feature importance"); ax.set_xlabel("importance score")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
    plt.show()


# per-thermostat breakdown
def run_per_thermostat(train, test, feature_cols):
    print("\n  per-thermostat results:")
    print(f"  {'id':>4}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'train_rows':>10}  {'test_rows':>9}")
    print("  " + "-"*50)

    for tid in sorted(train["thermostat_id"].unique()):
        tr = train[train["thermostat_id"] == tid]
        te = test[test["thermostat_id"] == tid]
        if len(te) < 10:
            print(f"  {tid:>4}  skipped (too few test rows)")
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0
        )
        model.fit(tr[feature_cols], tr[TARGET_COL])
        preds = np.clip(model.predict(te[feature_cols]), 0, 60)

        r2   = r2_score(te[TARGET_COL], preds)
        rmse = np.sqrt(mean_squared_error(te[TARGET_COL], preds))
        mae  = mean_absolute_error(te[TARGET_COL], preds)
        print(f"  {tid:>4}  {r2:>6.3f}  {rmse:>6.2f}  {mae:>6.2f}  {len(tr):>10}  {len(te):>9}")


# per-month breakdown
def run_per_month(test):
    print("\n  per-month results (pooled model, test set):")
    print(f"  {'month':>10}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'test_rows':>9}")
    print("  " + "-"*50)

    test = test.copy()
    test["month_label"] = test["hour_bucket"].dt.to_period("M").astype(str)

    for month in sorted(test["month_label"].unique()):
        subset = test[test["month_label"] == month]
        if len(subset) < 10:
            continue
        y_true = subset[TARGET_COL]
        y_pred = subset["predicted"]
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        print(f"  {month:>10}  {r2:>6.3f}  {rmse:>6.2f}  {mae:>6.2f}  {len(subset):>9}")


# per-thermostat per-month breakdown 
def run_per_thermostat_per_month(test):
    print("\n  per-thermostat per-month results (pooled model, test set):")
    print(f"  {'id':>4}  {'month':>10}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'rows':>6}")
    print("  " + "-"*55)

    test = test.copy()
    test["month_label"] = test["hour_bucket"].dt.to_period("M").astype(str)

    for tid in sorted(test["thermostat_id"].unique()):
        for month in sorted(test["month_label"].unique()):
            subset = test[(test["thermostat_id"] == tid) & (test["month_label"] == month)]
            if len(subset) < 10:
                continue
            y_true = subset[TARGET_COL]
            y_pred = subset["predicted"]
            r2   = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            print(f"  {tid:>4}  {month:>10}  {r2:>6.3f}  {rmse:>6.2f}  {mae:>6.2f}  {len(subset):>6}")
        print()


# data loading helpers 
def load_all_thermostats(file_paths):
    all_dfs = []
    for i, path in enumerate(file_paths):
        print(f"  loading thermostat {i+1}: {path}")
        try:
            df = load_data(path)
            df = compute_runtime_per_hour(df)
            df = engineer_features(df)
            df["thermostat_id"] = i + 1
            all_dfs.append(df)
            print(f"    → {len(df)} rows")
        except Exception as e:
            print(f"    → skipping, error: {e}")
    return all_dfs


def load_kaggle_thermostats(min_months=6):
    print("\ndownloading dataset from kaggle...")
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    path = kagglehub.dataset_download("lsobieski/processed-thermostat-data")
    print(f"  dataset cached at: {path}")

    viable = []
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(path, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=["Timestamp"])
            date_range = (df["Timestamp"].max() - df["Timestamp"].min()).days / 30
            if date_range >= min_months:
                viable.append((filename, df))
            else:
                print(f"  skipping {filename} — only {date_range:.1f} months")
        except Exception as e:
            print(f"  could not parse {filename}: {e}")

    print(f"  {len(viable)} files pass {min_months}+ month filter")
    return viable

def add_thermostat_stats(train, test):
    stats = (
        train.groupby("thermostat_id")["runtime_minutes"]
        .agg(thermo_mean="mean", thermo_std="std", thermo_p25=lambda x: x.quantile(0.25))
        .reset_index()
    )
    train = train.merge(stats, on="thermostat_id", how="left")
    test  = test.merge(stats,  on="thermostat_id", how="left")
    return train, test
    
def plot_r2_histogram(train, test, feature_cols):
    # collect per-thermostat R^2 from the pooled model predictions
    r2_values = []
    test = test.copy()
    test["month_label"] = test["hour_bucket"].dt.to_period("M").astype(str)

    for tid in sorted(test["thermostat_id"].unique()):
        te = test[test["thermostat_id"] == tid]
        if len(te) < 10:
            continue
        r2 = r2_score(te[TARGET_COL], te["predicted"])
        r2_values.append(r2)

    r2_values = np.array(r2_values)
    mean_r2   = np.mean(r2_values)
    median_r2 = np.median(r2_values)
    pct_pos   = (r2_values >= 0.3).sum()
    pct_neg   = (r2_values < 0).sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#D85A30" if v < 0 else "#378ADD" for v in r2_values]
    n, bins, patches = ax.hist(r2_values, bins=20, edgecolor="white", linewidth=0.8)
    for patch, val in zip(patches, bins):
        patch.set_facecolor("#D85A30" if val < 0 else "#378ADD")

    ax.axvline(mean_r2,   color="black",  linewidth=1.5, linestyle="--", label=f"mean   = {mean_r2:.3f}")
    ax.axvline(median_r2, color="gray",   linewidth=1.5, linestyle=":",  label=f"median = {median_r2:.3f}")
    ax.axvline(0,         color="red",    linewidth=1.0, linestyle="-",  alpha=0.4)

    ax.set_xlabel("R^2 score (per thermostat)")
    ax.set_ylabel("count")
    ax.set_title("distribution of R^2 across all thermostats (pooled model)")
    ax.legend()

    textstr = f"n = {len(r2_values)}  |  R^2 ≥ 0.3: {pct_pos}  |  R^2 < 0: {pct_neg}"
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
            fontsize=10, va="top", ha="right", color="gray")

    plt.tight_layout()
    plt.savefig("r2_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()

# cross validation
def run_time_series_cv(combined, feature_cols, target_col="runtime_minutes", n_splits=5):
    combined = combined.sort_values("hour_bucket").reset_index(drop=True)

    # build fold boundaries using time quantiles
    # each fold's test window is a contiguous slice of time
    time_vals = combined["hour_bucket"]
    quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]   # n_splits interior cuts
    cutoffs = [time_vals.quantile(q) for q in quantiles]

    print(f"\n{'='*60}")
    print(f"  time-series cross-validation  ({n_splits} folds)")
    print(f"{'='*60}")
    print(f"  {'fold':>4}  {'train_rows':>10}  {'test_rows':>9}  "
          f"{'r2':>6}  {'rmse':>6}  {'mae':>6}  {'cv%':>6}")
    print("  " + "-"*58)

    fold_results = []

    for fold in range(n_splits):
        # Train: everything strictly before this fold's test window
        # Test:  the slice between cutoff[fold] and cutoff[fold+1]
        #        (last fold takes everything after cutoff[-1])
        train_cut = cutoffs[fold]
        test_start = cutoffs[fold]
        test_end   = cutoffs[fold + 1] if fold + 1 < len(cutoffs) else time_vals.max()

        train_mask = combined["hour_bucket"] < train_cut
        test_mask  = (combined["hour_bucket"] >= test_start) & \
                     (combined["hour_bucket"] <= test_end)

        tr = combined[train_mask].copy()
        te = combined[test_mask].copy()

        if len(tr) < 100 or len(te) < 10:
            print(f"  {fold+1:>4}  skipped — not enough rows "
                  f"(train={len(tr)}, test={len(te)})")
            continue

        # re-compute thermostat stats from training fold only (no leakage)
        stats = (
            tr.groupby("thermostat_id")["runtime_minutes"]
            .agg(thermo_mean="mean",
                 thermo_std="std",
                 thermo_p25=lambda x: x.quantile(0.25))
            .reset_index()
        )
        tr = tr.merge(stats, on="thermostat_id", how="left")
        te = te.merge(stats, on="thermostat_id", how="left")

        # drop rows where a thermostat appeared only in test 
        te = te.dropna(subset=["thermo_mean"])

        X_tr, y_tr = tr[feature_cols], tr[target_col]
        X_te, y_te = te[feature_cols], te[target_col]

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
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)

        preds = np.clip(model.predict(X_te), 0, 60)

        r2   = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae  = mean_absolute_error(y_te, preds)
        cv_pct = (rmse / y_te.mean() * 100) if y_te.mean() > 0 else float("nan")

        fold_results.append({
            "fold":       fold + 1,
            "train_rows": len(tr),
            "test_rows":  len(te),
            "r2":         r2,
            "rmse":       rmse,
            "mae":        mae,
            "cv_pct":     cv_pct,
            "test_start": test_start,
            "test_end":   test_end,
        })

        print(f"  {fold+1:>4}  {len(tr):>10}  {len(te):>9}  "
              f"{r2:>6.3f}  {rmse:>6.2f}  {mae:>6.2f}  {cv_pct:>5.1f}%")

    results_df = pd.DataFrame(fold_results)

    if len(results_df):
        print("\n  summary across folds:")
        for metric in ["r2", "rmse", "mae", "cv_pct"]:
            vals = results_df[metric]
            print(f"  {metric:>8}: {vals.mean():.3f} ± {vals.std():.3f}  "
                  f"(min {vals.min():.3f} / max {vals.max():.3f})")

        print(f"\n  stability check:")
        r2_range = results_df["r2"].max() - results_df["r2"].min()
        if r2_range < 0.05:
            print(f"  R^2 range = {r2_range:.3f} → stable estimate, your 0.440 is reliable")
        elif r2_range < 0.15:
            print(f"  R^2 range = {r2_range:.3f} → moderate variance, "
                  f"some sensitivity to the split")
        else:
            print(f"  R^2 range = {r2_range:.3f} → high variance, "
                  f"the single-split estimate was unreliable")

    return results_df


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, roc_auc_score, confusion_matrix
)


# ── stage 1: binary classifier ─────────────────────────────────────────────

def train_stage1_classifier(X_train, y_train_runtime, X_test, y_test_runtime):
    """
    Trains an XGBoost binary classifier to predict whether runtime_minutes > 0.
    Returns the model and binary predictions/probabilities on the test set.
    """
    y_train_bin = (y_train_runtime > 0).astype(int)
    y_test_bin  = (y_test_runtime  > 0).astype(int)

    # class imbalance: weight the minority class proportionally
    n_off = (y_train_bin == 0).sum()
    n_on  = (y_train_bin == 1).sum()
    scale_pos_weight = n_off / n_on
    print(f"\n  stage 1 class balance (train):")
    print(f"    off (0): {n_off:,}  |  on (1): {n_on:,}  "
          f"|  scale_pos_weight: {scale_pos_weight:.2f}")

    clf = xgb.XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.02,
        max_depth             = 8,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        gamma                 = 1,
        scale_pos_weight      = scale_pos_weight,
        objective             = "binary:logistic",
        eval_metric           = "auc",
        random_state          = 42,
        verbosity             = 0,
        early_stopping_rounds = 20,
    )
    clf.fit(X_train, y_train_bin,
            eval_set=[(X_test, y_test_bin)],
            verbose=False)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    print(f"\n  stage 1 classifier results:")
    print(f"  roc-auc : {roc_auc_score(y_test_bin, proba):.3f}")
    print(f"\n  confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test_bin, preds)
    print(f"           pred_off  pred_on")
    print(f"  act_off  {cm[0,0]:>8}  {cm[0,1]:>7}")
    print(f"  act_on   {cm[1,0]:>8}  {cm[1,1]:>7}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  false-on  rate (pred on, actually off): "
          f"{fp/(fp+tn)*100:.1f}%")
    print(f"  false-off rate (pred off, actually on): "
          f"{fn/(fn+tp)*100:.1f}%")
    print(f"\n{classification_report(y_test_bin, preds, target_names=['off','on'])}")

    return clf, preds, proba, y_test_bin


# ── stage 2: runtime regressor (on-hours only) ─────────────────────────────

def train_stage2_regressor(train, test, feature_cols, on_mask_train, on_mask_test):
    """
    Trains an XGBoost regressor on hours where runtime_minutes > 0.
    on_mask_train / on_mask_test are boolean arrays of the actual on-hours.
    Returns the model and predictions (already clipped to 1–60).
    """
    tr_on = train[on_mask_train]
    te_on = test[on_mask_test]

    X_tr = tr_on[feature_cols]
    y_tr = tr_on["runtime_minutes"]
    X_te = te_on[feature_cols]
    y_te = te_on["runtime_minutes"]

    print(f"\n  stage 2 regressor — on-hours only:")
    print(f"    train rows: {len(tr_on):,}  |  test rows: {len(te_on):,}")
    print(f"    train runtime mean: {y_tr.mean():.1f} min  "
          f"|  test runtime mean: {y_te.mean():.1f} min")

    reg = xgb.XGBRegressor(
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
    reg.fit(X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False)

    preds = np.clip(reg.predict(X_te), 1, 60)  # on-hours: floor at 1, not 0

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae  = mean_absolute_error(y_te, preds)
    r2   = r2_score(y_te, preds)
    print(f"\n  stage 2 results (on-hours only):")
    print(f"  rmse: {rmse:.2f}  mae: {mae:.2f}  r2: {r2:.3f}")

    importance = pd.Series(reg.feature_importances_,
                           index=feature_cols).sort_values(ascending=False)
    print(f"\n  top 5 features (stage 2):")
    print(importance.nlargest(5).to_string())

    return reg, preds, y_te


# ── combine stages → final predictions ─────────────────────────────────────

def combine_stages(stage1_preds_bin, stage2_preds_on, test, feature_cols):
    """
    Assembles final predictions for the full test set.
    - Hours where stage 1 predicts off → 0
    - Hours where stage 1 predicts on  → stage 2 prediction
    Returns a full-length array aligned to test index.
    """
    final_preds = np.zeros(len(test))

    # indices in test where stage 1 said "on"
    on_idx = np.where(stage1_preds_bin == 1)[0]

    if len(on_idx) != len(stage2_preds_on):
        raise ValueError(
            f"stage 2 predictions length ({len(stage2_preds_on)}) "
            f"doesn't match stage 1 on-count ({len(on_idx)})"
        )

    final_preds[on_idx] = stage2_preds_on
    return final_preds


# ── end-to-end evaluation ──────────────────────────────────────────────────

def evaluate_two_stage(y_true, final_preds, label="two-stage model"):
    rmse   = np.sqrt(mean_squared_error(y_true, final_preds))
    mae    = mean_absolute_error(y_true, final_preds)
    r2     = r2_score(y_true, final_preds)
    cv_pct = (rmse / y_true.mean() * 100) if y_true.mean() > 0 else float("nan")

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  rmse : {rmse:.2f} min  (target < 5 min/hr)")
    print(f"  mae  : {mae:.2f} min")
    print(f"  r2   : {r2:.3f}")
    print(f"  cv   : {cv_pct:.1f}%      (target < 30% per ashrae)")
    return {"rmse": rmse, "mae": mae, "r2": r2, "cv_pct": cv_pct}


# ── per-month breakdown (reuses your existing logic) ───────────────────────

def run_per_month_two_stage(test, final_preds):
    test = test.copy()
    test["predicted_2s"] = final_preds
    test["month_label"]  = test["hour_bucket"].dt.to_period("M").astype(str)

    print(f"\n  per-month results (two-stage model, test set):")
    print(f"  {'month':>10}  {'r2':>6}  {'rmse':>6}  {'mae':>6}  {'test_rows':>9}")
    print("  " + "-"*50)

    for month in sorted(test["month_label"].unique()):
        subset = test[test["month_label"] == month]
        if len(subset) < 10:
            continue
        r2   = r2_score(subset["runtime_minutes"], subset["predicted_2s"])
        rmse = np.sqrt(mean_squared_error(
                   subset["runtime_minutes"], subset["predicted_2s"]))
        mae  = mean_absolute_error(
                   subset["runtime_minutes"], subset["predicted_2s"])
        print(f"  {month:>10}  {r2:>6.3f}  {rmse:>6.2f}  {mae:>6.2f}  "
              f"{len(subset):>9}")


# ── main: drop this into your existing main() ──────────────────────────────

def run_two_stage_model(train, test, feature_cols):
    """
    Full two-stage pipeline. Expects train/test to already have
    thermostat stats merged (thermo_mean, thermo_std, thermo_p25).
    """
    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train["runtime_minutes"]
    y_test  = test["runtime_minutes"]

    # ── stage 1 ──
    clf, s1_preds, s1_proba, y_test_bin = train_stage1_classifier(
        X_train, y_train, X_test, y_test
    )

    # ── stage 2: train on actual on-hours, predict on stage-1-predicted on-hours ──
    # important: stage 2 trains on true on-hours (ground truth),
    # but predicts on whatever stage 1 called "on" at test time
    on_mask_train = (y_train > 0).values
    on_mask_test  = (s1_preds == 1)        # stage 1 decision, not ground truth

    _, s2_preds_on, _ = train_stage2_regressor(
        train, test, feature_cols, on_mask_train, on_mask_test
    )

    # ── combine ──
    final_preds = combine_stages(s1_preds, s2_preds_on, test, feature_cols)

    # ── evaluate ──
    metrics_2s = evaluate_two_stage(y_test.values, final_preds)
    run_per_month_two_stage(test, final_preds)

    # ── direct comparison with single-stage ──
    print(f"\n  head-to-head vs single-stage xgboost:")
    print(f"  {'metric':>6}  {'single':>8}  {'two-stage':>10}  {'delta':>8}")
    print("  " + "-"*38)
    single = {"r2": 0.440, "rmse": 12.42, "mae": 9.12, "cv_pct": 49.6}
    for k in ["r2", "rmse", "mae", "cv_pct"]:
        delta = metrics_2s[k] - single[k]
        arrow = "▲" if (k == "r2" and delta > 0) or \
                       (k != "r2" and delta < 0) else "▼"
        print(f"  {k:>6}  {single[k]:>8.3f}  {metrics_2s[k]:>10.3f}  "
              f"{arrow} {abs(delta):.3f}")

    return clf, final_preds, metrics_2s

# main
def main():
    # load kaggle thermostats only
    kaggle_dfs = load_kaggle_thermostats(min_months=6)

    all_dfs = []
    for i, (filename, raw_df) in enumerate(kaggle_dfs):
        try:
            tmp = f"/tmp/kaggle_{i}.csv"
            raw_df.to_csv(tmp, index=False)
            df = load_data(tmp)
            df = compute_runtime_per_hour(df)
            df = engineer_features(df)
            df["thermostat_id"] = 100 + i
            all_dfs.append(df)
        except Exception as e:
            print(f"  skipping {filename}: {e}")

    print(f"  processed {len(all_dfs)} kaggle thermostats")

    # combine and split
    combined = pd.concat(all_dfs, ignore_index=True)
    cutoff = combined["hour_bucket"].quantile(0.8)
    print(f"\n  global cutoff: {cutoff}")

    train = combined[combined["hour_bucket"] <= cutoff].copy()
    test  = combined[combined["hour_bucket"] >  cutoff].copy()
    print(f"  train: {len(train)} rows | test: {len(test)} rows")

    # train pooled model
    feature_cols = FEATURE_COLS + ["thermostat_id", "thermo_mean", "thermo_std", "thermo_p25"]
    train, test = add_thermostat_stats(train, test)
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test,  y_test  = test[feature_cols],  test[TARGET_COL]

    clf, final_preds, metrics_2s = run_two_stage_model(train, test, feature_cols)
    test["predicted"] = final_preds
    plot_results(test, y_test, final_preds, clf, metrics_2s, feature_cols)

    # plots and breakdowns
    plot_r2_histogram(train, test, feature_cols)
    run_per_month(test)
    run_per_thermostat_per_month(test)
    run_per_thermostat(train, test, feature_cols)


if __name__ == "__main__":
    main()