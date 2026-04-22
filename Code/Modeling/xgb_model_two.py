# =============================================================================
# hvac_model.py
# compares multiple xgboost training strategies for daily hvac runtime prediction
#
# strategies:
#   1. pooled       — one global model across all thermostats
#   2. local        — one model per thermostat
#   3. seasonal     — separate models per calendar season
#   4. clustered    — thermostats grouped by runtime behavior, one model per cluster
#
# prerequisite: run hvac_preprocess.py once to create the processed dataset
# processed dataset: zackaid/processed-thermostat-data-daily
# =============================================================================

import os
import warnings
import textwrap
import kagglehub
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── credentials ──────────────────────────────────────────────────────────────
KAGGLE_TOKEN           = "KGAT_6d26ebf584b9d44052424f267dffec66"
PROCESSED_DATASET_SLUG = "zackaid/processed-thermostat-data-daily"

# ── feature config ───────────────────────────────────────────────────────────
LAG_DAYS = 14  # extended to capture 2-week patterns
LAG_COLS = [
    "daily_runtime_hours",
    "daily_heating_hours",
    "daily_cooling_hours",
    "indoor_temp_time_weighted_mean",
    "setpoint_time_weighted_mean",
    "setpoint_gap_mean",
    "indoor_temp_std",
]
FEATURE_COLS = [
    # runtime lags — 1-7 plus 14 (were computing lag 4/5/6 but not using them)
    "daily_runtime_hours_lag_1",
    "daily_runtime_hours_lag_2",
    "daily_runtime_hours_lag_3",
    "daily_runtime_hours_lag_4",
    "daily_runtime_hours_lag_5",
    "daily_runtime_hours_lag_6",
    "daily_runtime_hours_lag_7",
    "daily_runtime_hours_lag_14",
    # heating / cooling lags
    "daily_heating_hours_lag_1",
    "daily_heating_hours_lag_7",
    "daily_cooling_hours_lag_1",
    "daily_cooling_hours_lag_7",
    # indoor thermal state lags
    "indoor_temp_time_weighted_mean_lag_1",
    "setpoint_time_weighted_mean_lag_1",
    "setpoint_gap_mean_lag_1",
    "indoor_temp_std_lag_1",
    # outdoor / calendar
    "true_outside_min",
    "true_outside_max",
    "true_outside_mean",
    "true_humidity_mean",
    "outdoor_temp_trend_gradient",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    # per-thermostat baseline stats (from train only)
    "thermo_mean_runtime",
    "thermo_std_runtime",
]
TARGET = "daily_runtime_hours"

# ── xgboost base params (optuna-tuned) ───────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators       = 1182,
    learning_rate      = 0.0105,
    max_depth          = 9,
    subsample          = 0.7467,
    colsample_bytree   = 0.4635,
    min_child_weight   = 50,
    gamma              = 4.501,
    reg_alpha          = 1.998,
    reg_lambda         = 4.794,
    objective          = "reg:squarederror",
    random_state       = 42,
    verbosity          = 0,
    early_stopping_rounds = 20,
)

# ── per-season optuna trials (set 0 to skip tuning) ─────────────────────────
SEASONAL_OPTUNA_TRIALS = 40

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
     3: "Spring", 4: "Spring", 5: "Spring",
     6: "Summer", 7: "Summer", 8: "Summer",
     9: "Autumn",10: "Autumn",11: "Autumn",
}

# =============================================================================
# PRINTING UTILITIES
# =============================================================================

SEP  = "─" * 72
SEP2 = "═" * 72

def banner(title: str):
    pad = (72 - len(title) - 2) // 2
    print(f"\n{'═' * 72}")
    print(f"{'═' * pad}  {title}  {'═' * (72 - pad - len(title) - 2)}")
    print(f"{'═' * 72}")

def section(title: str):
    print(f"\n  ┌─ {title} {'─' * (66 - len(title))}┐")

def row_sep():
    print(f"  {'─' * 70}")

def metrics_block(label: str, rmse: float, mae: float, r2: float,
                  n_train: int = None, n_test: int = None, indent: int = 4):
    pad = " " * indent
    extra = ""
    if n_train is not None:
        extra = f"  │  train {n_train:,}  test {n_test:,}"
    print(f"{pad}{label:<30}  rmse {rmse:5.3f}  mae {mae:5.3f}  r² {r2:+.3f}{extra}")

def table_header(cols: list[tuple[str, int, str]]):
    """cols = [(label, width, align), ...]"""
    parts = []
    for label, width, align in cols:
        if align == "r":
            parts.append(f"{label:>{width}}")
        elif align == "c":
            parts.append(f"{label:^{width}}")
        else:
            parts.append(f"{label:<{width}}")
    print("  " + "  ".join(parts))
    print("  " + "  ".join("─" * w for _, w, _ in cols))

def table_row(values: list, cols: list[tuple[str, int, str]]):
    parts = []
    for val, (_, width, align) in zip(values, cols):
        s = str(val)
        if align == "r":
            parts.append(f"{s:>{width}}")
        elif align == "c":
            parts.append(f"{s:^{width}}")
        else:
            parts.append(f"{s:<{width}}")
    print("  " + "  ".join(parts))


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data(min_days: int = 90) -> list[pd.DataFrame]:
    print(f"\n  loading dataset from kaggle…")
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    path = kagglehub.dataset_download(PROCESSED_DATASET_SLUG)
    files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
    print(f"  found {len(files)} csv files")

    dfs, skipped = [], 0
    for fn in files:
        try:
            df = pd.read_csv(os.path.join(path, fn), low_memory=False)
            df["Date"] = pd.to_datetime(df["Date"])
            if len(df) < min_days:
                skipped += 1
                continue
            dfs.append(df)
        except Exception:
            skipped += 1

    print(f"  loaded {len(dfs)} thermostats  ({skipped} skipped — < {min_days} days)")
    return dfs


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Equipment_ID", "Date"]).reset_index(drop=True)
    grids = []
    for eid, grp in df.groupby("Equipment_ID"):
        dr = pd.date_range(grp["Date"].min(), grp["Date"].max(), freq="D")
        g  = grp.set_index("Date").reindex(dr).rename_axis("Date").reset_index()
        g["Equipment_ID"] = eid
        grids.append(g)
    df = pd.concat(grids, ignore_index=True)

    present = [c for c in LAG_COLS if c in df.columns]
    for i in range(1, LAG_DAYS + 1):
        shifted = df.groupby("Equipment_ID")[present].shift(i)
        shifted.columns = [f"{c}_lag_{i}" for c in present]
        df = pd.concat([df, shifted], axis=1)

    return df


def add_thermostat_stats(train: pd.DataFrame,
                         test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = (
        train.groupby("Equipment_ID")[TARGET]
        .agg(thermo_mean_runtime="mean", thermo_std_runtime="std")
        .reset_index()
    )
    train = train.merge(stats, on="Equipment_ID", how="left")
    test  = test.merge(stats,  on="Equipment_ID", how="left")
    return train, test


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["Date"].dt.month.map(SEASONS)
    return df


# =============================================================================
# SECTION 3 — MODEL HELPERS
# =============================================================================

def _fit_predict(X_tr, y_tr, X_te, params=None):
    p = {**XGB_PARAMS, **(params or {})}
    m = xgb.XGBRegressor(**p)
    # Use a real validation split (last 15% of train) so early stopping is meaningful.
    # Previously eval_set used np.zeros which made early stopping fire arbitrarily.
    val_size = max(int(len(X_tr) * 0.15), 1)
    X_val, y_val = X_tr.iloc[-val_size:], y_tr.iloc[-val_size:]
    X_fit, y_fit = X_tr.iloc[:-val_size],  y_tr.iloc[:-val_size]
    if len(X_fit) < 20:
        # Too small to split — train on everything, no early stopping
        m_ns = xgb.XGBRegressor(**{k: v for k, v in p.items() if k != "early_stopping_rounds"})
        m_ns.fit(X_tr, y_tr, verbose=False)
        return np.clip(m_ns.predict(X_te), 0, None), m_ns
    m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return np.clip(m.predict(X_te), 0, None), m


def _score(y_true, y_pred) -> dict:
    return dict(
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae  = float(mean_absolute_error(y_true, y_pred)),
        r2   = float(r2_score(y_true, y_pred)),
        n    = int(len(y_true)),
    )


def _avail(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_COLS if f in df.columns]


# =============================================================================
# SECTION 4 — STRATEGY 1: POOLED MODEL
# =============================================================================

def run_pooled(train: pd.DataFrame,
               test:  pd.DataFrame) -> tuple[dict, pd.Series]:
    banner("STRATEGY 1 — POOLED  (one global model)")

    feats = _avail(train)
    X_tr, y_tr = train[feats], train[TARGET]
    X_te, y_te = test[feats],  test[TARGET]

    preds, model = _fit_predict(X_tr, y_tr, X_te)
    overall = _score(y_te, preds)

    print(f"\n  overall   rmse {overall['rmse']:.3f}  mae {overall['mae']:.3f}"
          f"  r² {overall['r2']:+.3f}  "
          f"(train {len(train):,}  test {len(test):,})")

    # per-month breakdown
    test = test.copy()
    test["_pred"] = preds
    test["_month"] = test["Date"].dt.to_period("M").astype(str)

    section("per-month breakdown")
    cols = [("month", 9, "l"), ("r²", 6, "r"), ("rmse", 6, "r"),
            ("mae", 6, "r"), ("rows", 6, "r")]
    table_header(cols)
    for month in sorted(test["_month"].unique()):
        sub = test[test["_month"] == month]
        if len(sub) < 10:
            continue
        s = _score(sub[TARGET], sub["_pred"])
        table_row([month, f"{s['r2']:+.3f}", f"{s['rmse']:.3f}",
                   f"{s['mae']:.3f}", s['n']], cols)

    # per-thermostat distribution
    per_thermo = {}
    for eid, grp in test.groupby("Equipment_ID"):
        if len(grp) >= 10:
            per_thermo[eid] = _score(grp[TARGET], grp["_pred"])

    r2_vals = [v["r2"] for v in per_thermo.values()]
    section("per-thermostat r² distribution")
    bins = [(-999, 0, "r² < 0   (poor)"),
            (0,  0.3, "0 – 0.3  (weak)"),
            (0.3,0.5, "0.3–0.5  (moderate)"),
            (0.5,0.7, "0.5–0.7  (good)"),
            (0.7,999, "r² > 0.7 (excellent)")]
    cols2 = [("range", 20, "l"), ("count", 7, "r"), ("share", 8, "r")]
    table_header(cols2)
    for lo, hi, label in bins:
        cnt = sum(lo <= v < hi for v in r2_vals)
        table_row([label, cnt, f"{cnt/len(r2_vals)*100:.0f}%"], cols2)
    print(f"\n  median r² across thermostats: {np.median(r2_vals):+.3f}")
    print(f"  mean   r² across thermostats: {np.mean(r2_vals):+.3f}")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    section("top 10 features")
    cols3 = [("feature", 38, "l"), ("importance", 10, "r")]
    table_header(cols3)
    for feat, val in imp.head(10).items():
        table_row([feat, f"{val:.4f}"], cols3)

    # collect monthly metrics for plotting
    monthly_metrics = {}
    for month in sorted(test["_month"].unique()):
        sub = test[test["_month"] == month]
        if len(sub) >= 10:
            monthly_metrics[month] = _score(sub[TARGET], sub["_pred"])

    # collect actual vs predicted for scatter plot
    pooled_scatter = pd.DataFrame({"actual": y_te.values, "pred": preds})

    return (overall,
            pd.Series({eid: v["r2"] for eid, v in per_thermo.items()}),
            imp,
            monthly_metrics,
            pooled_scatter)


# =============================================================================
# SECTION 5 — STRATEGY 2: LOCAL (PER-THERMOSTAT) MODELS
# =============================================================================

def run_local(train: pd.DataFrame,
              test:  pd.DataFrame) -> tuple[dict, pd.Series]:
    banner("STRATEGY 2 — LOCAL  (one model per thermostat)")

    feats = _avail(train)
    all_true, all_pred = [], []
    per_thermo = {}
    skipped = 0

    for eid in sorted(train["Equipment_ID"].unique()):
        tr = train[train["Equipment_ID"] == eid]
        te = test[test["Equipment_ID"]  == eid]
        if len(tr) < 60 or len(te) < 10:
            skipped += 1
            continue
        preds, _ = _fit_predict(tr[feats], tr[TARGET], te[feats])
        s = _score(te[TARGET], preds)
        per_thermo[eid] = s
        all_true.extend(te[TARGET].tolist())
        all_pred.extend(preds.tolist())

    overall = _score(np.array(all_true), np.array(all_pred))
    print(f"\n  overall   rmse {overall['rmse']:.3f}  mae {overall['mae']:.3f}"
          f"  r² {overall['r2']:+.3f}")
    print(f"  modelled {len(per_thermo)} thermostats  ({skipped} skipped — insufficient data)")

    r2_vals = [v["r2"] for v in per_thermo.values()]
    section("per-thermostat r² distribution")
    bins = [(-999, 0, "r² < 0   (poor)"),
            (0,  0.3, "0 – 0.3  (weak)"),
            (0.3,0.5, "0.3–0.5  (moderate)"),
            (0.5,0.7, "0.5–0.7  (good)"),
            (0.7,999, "r² > 0.7 (excellent)")]
    cols2 = [("range", 20, "l"), ("count", 7, "r"), ("share", 8, "r")]
    table_header(cols2)
    for lo, hi, label in bins:
        cnt = sum(lo <= v < hi for v in r2_vals)
        table_row([label, cnt, f"{cnt/len(r2_vals)*100:.0f}%"], cols2)
    print(f"\n  median r²: {np.median(r2_vals):+.3f}   mean r²: {np.mean(r2_vals):+.3f}")

    # top / bottom 5 thermostats
    sorted_items = sorted(per_thermo.items(), key=lambda x: x[1]["r2"], reverse=True)
    section("best 5 thermostats")
    cols3 = [("equipment_id", 45, "l"), ("r²", 6, "r"), ("rmse", 6, "r"),
             ("mae", 6, "r"), ("train", 6, "r"), ("test", 5, "r")]
    table_header(cols3)
    for eid, s in sorted_items[:5]:
        n_tr = len(train[train["Equipment_ID"] == eid])
        n_te = len(test[test["Equipment_ID"]  == eid])
        table_row([str(eid)[:45], f"{s['r2']:+.3f}", f"{s['rmse']:.3f}",
                   f"{s['mae']:.3f}", n_tr, n_te], cols3)

    section("worst 5 thermostats")
    table_header(cols3)
    for eid, s in sorted_items[-5:]:
        n_tr = len(train[train["Equipment_ID"] == eid])
        n_te = len(test[test["Equipment_ID"]  == eid])
        table_row([str(eid)[:45], f"{s['r2']:+.3f}", f"{s['rmse']:.3f}",
                   f"{s['mae']:.3f}", n_tr, n_te], cols3)

    return overall, pd.Series({eid: v["r2"] for eid, v in per_thermo.items()})


# =============================================================================
# SECTION 6 — STRATEGY 3: SEASONAL MODELS
# =============================================================================

def _tune_season(X_tr, y_tr, n_trials: int = 30) -> dict:
    """run optuna for a single season's train split. returns best xgb params."""
    if not OPTUNA_AVAILABLE or n_trials == 0:
        return {}

    # inner val split: last 20% of this season's train (time-ordered)
    val_size  = max(int(len(X_tr) * 0.20), 1)
    X_fit, y_fit = X_tr.iloc[:-val_size], y_tr.iloc[:-val_size]
    X_val, y_val = X_tr.iloc[-val_size:], y_tr.iloc[-val_size:]

    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            max_depth         = trial.suggest_int("max_depth", 4, 10),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.3, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 10, 100),
            gamma             = trial.suggest_float("gamma", 0.0, 6.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 8.0),
            objective         = "reg:squarederror",
            random_state      = 42,
            verbosity         = 0,
            early_stopping_rounds = 20,
        )
        m = xgb.XGBRegressor(**p)
        m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.clip(m.predict(X_val), 0, None)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_seasonal(combined: pd.DataFrame, n_optuna_trials: int = 30) -> dict:
    banner("STRATEGY 3 — SEASONAL  (per-season Optuna tuning)")

    combined = add_season(combined)
    results  = {}

    tune_label = f"optuna {n_optuna_trials} trials" if (OPTUNA_AVAILABLE and n_optuna_trials > 0) else "global params"
    cols = [("season", 8, "l"), ("r²", 6, "r"), ("rmse", 6, "r"),
            ("mae", 6, "r"), ("train", 8, "r"), ("test", 7, "r"), ("best_rmse", 9, "r")]
    section(f"per-season results  (80/20 time split, {tune_label})")
    table_header(cols)

    all_true, all_pred = [], []

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sub = combined[combined["season"] == season].copy()
        if len(sub) < 200:
            continue

        cutoff = sub["Date"].quantile(0.8)
        tr = sub[sub["Date"] <= cutoff].copy()
        te = sub[sub["Date"] >  cutoff].copy()

        stats = (tr.groupby("Equipment_ID")[TARGET]
                   .agg(thermo_mean_runtime="mean", thermo_std_runtime="std")
                   .reset_index())
        for col in ["thermo_mean_runtime", "thermo_std_runtime"]:
            tr = tr.drop(columns=[col], errors="ignore")
            te = te.drop(columns=[col], errors="ignore")
        tr = tr.merge(stats, on="Equipment_ID", how="left")
        te = te.merge(stats, on="Equipment_ID", how="left")
        te = te.dropna(subset=["thermo_mean_runtime"])

        tr = tr.dropna(subset=[TARGET])
        te = te.dropna(subset=[TARGET])
        feats = _avail(tr)
        if len(tr) < 50 or len(te) < 10:
            continue

        # ── per-season Optuna tuning ──────────────────────────────────────────
        print(f"  tuning {season}…", flush=True)
        best = _tune_season(tr[feats], tr[TARGET], n_trials=n_optuna_trials)

        # build final params: start from global base, override with season-best
        season_params = {**XGB_PARAMS}
        if best:
            season_params.update(best)

        preds, _ = _fit_predict(tr[feats], tr[TARGET], te[feats], params=season_params)
        s = _score(te[TARGET], preds)
        results[season] = {**s, "params": season_params}
        all_true.extend(te[TARGET].tolist())
        all_pred.extend(preds.tolist())

        val_size = max(int(len(tr) * 0.20), 1)
        X_val_check = tr[feats].iloc[-val_size:]
        y_val_check = tr[TARGET].iloc[-val_size:]
        val_preds   = np.clip(
            xgb.XGBRegressor(**{k: v for k, v in season_params.items()
                                if k != "early_stopping_rounds"})
            .fit(tr[feats].iloc[:-val_size], tr[TARGET].iloc[:-val_size], verbose=False)
            .predict(X_val_check), 0, None)
        best_val_rmse = float(np.sqrt(mean_squared_error(y_val_check, val_preds)))

        table_row([season, f"{s['r2']:+.3f}", f"{s['rmse']:.3f}",
                   f"{s['mae']:.3f}", len(tr), len(te), f"{best_val_rmse:.3f}"], cols)

    overall = _score(np.array(all_true), np.array(all_pred))
    print(f"\n  combined  rmse {overall['rmse']:.3f}  mae {overall['mae']:.3f}"
          f"  r² {overall['r2']:+.3f}")

    # print the winning params per season for reference
    section("best params per season")
    for season, res in results.items():
        p = res.get("params", {})
        print(f"\n  {season}:")
        for k in ["n_estimators", "learning_rate", "max_depth", "min_child_weight",
                  "gamma", "reg_alpha", "reg_lambda"]:
            if k in p:
                print(f"    {k:<22} {p[k]}")

    return overall, results


# =============================================================================
# SECTION 7 — STRATEGY 4: CLUSTERED MODELS
# =============================================================================

def run_clustered(train: pd.DataFrame,
                  test:  pd.DataFrame,
                  n_clusters: int = 4) -> dict:
    banner(f"STRATEGY 4 — CLUSTERED  ({n_clusters} behavior clusters)")

    # build per-thermostat behavior fingerprint from train
    profile_feats = ["daily_runtime_hours", "daily_heating_hours",
                     "daily_cooling_hours", "true_outside_mean"]
    profile_feats = [f for f in profile_feats if f in train.columns]

    profile = (
        train.groupby("Equipment_ID")[profile_feats]
        .agg(["mean", "std"])
        .fillna(0)
    )
    profile.columns = ["_".join(c) for c in profile.columns]

    scaler = StandardScaler()
    X_clust = scaler.fit_transform(profile)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_clust)
    cluster_map = dict(zip(profile.index, cluster_labels))

    train = train.copy()
    test  = test.copy()
    train["cluster"] = train["Equipment_ID"].map(cluster_map)
    test["cluster"]  = test["Equipment_ID"].map(cluster_map)

    feats = _avail(train)
    results = {}
    all_true, all_pred = [], []

    cols = [("cluster", 9, "l"), ("thermostats", 11, "r"), ("r²", 6, "r"),
            ("rmse", 6, "r"), ("mae", 6, "r"), ("train", 8, "r"), ("test", 7, "r")]
    section("per-cluster results")
    table_header(cols)

    for c in range(n_clusters):
        tr = train[train["cluster"] == c].dropna(subset=[TARGET])
        te = test[test["cluster"]   == c].dropna(subset=[TARGET])
        if len(tr) < 50 or len(te) < 10:
            continue
        n_thermos = tr["Equipment_ID"].nunique()
        preds, _ = _fit_predict(tr[feats], tr[TARGET], te[feats])
        s = _score(te[TARGET], preds)
        results[c] = s
        all_true.extend(te[TARGET].tolist())
        all_pred.extend(preds.tolist())
        table_row([f"cluster {c}", n_thermos, f"{s['r2']:+.3f}",
                   f"{s['rmse']:.3f}", f"{s['mae']:.3f}", len(tr), len(te)], cols)

    # cluster profiles
    section("cluster behavioral profiles  (train-set means)")
    p_cols = [("cluster", 9, "l"), ("thermostats", 11, "r"),
              ("mean runtime", 12, "r"), ("runtime std", 11, "r"),
              ("mean heating", 12, "r"), ("mean cooling", 12, "r")]
    table_header(p_cols)
    for c in range(n_clusters):
        tr = train[train["cluster"] == c]
        n_t = tr["Equipment_ID"].nunique()
        table_row([
            f"cluster {c}", n_t,
            f"{tr['daily_runtime_hours'].mean():.2f} hr",
            f"{tr['daily_runtime_hours'].std():.2f} hr",
            f"{tr.get('daily_heating_hours', pd.Series([0])).mean():.2f} hr",
            f"{tr.get('daily_cooling_hours', pd.Series([0])).mean():.2f} hr",
        ], p_cols)

    overall = _score(np.array(all_true), np.array(all_pred))
    print(f"\n  combined  rmse {overall['rmse']:.3f}  mae {overall['mae']:.3f}"
          f"  r² {overall['r2']:+.3f}")
    return overall, results


# =============================================================================
# SECTION 8 — TIME-SERIES CROSS VALIDATION
# =============================================================================

def run_tscv(combined: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    banner("TIME-SERIES CROSS-VALIDATION  (expanding window, 5 folds)")

    combined  = combined.sort_values("Date").reset_index(drop=True)
    quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
    cutoffs   = [combined["Date"].quantile(q) for q in quantiles]

    cols = [("fold", 5, "r"), ("train rows", 11, "r"), ("test rows", 10, "r"),
            ("r²", 7, "r"), ("rmse", 7, "r"), ("mae", 7, "r")]
    table_header(cols)

    fold_results = []
    for fold in range(n_splits):
        train_cut  = cutoffs[fold]
        test_start = cutoffs[fold]
        test_end   = cutoffs[fold + 1] if fold + 1 < len(cutoffs) else combined["Date"].max()

        tr = combined[combined["Date"] <  train_cut].copy()
        te = combined[(combined["Date"] >= test_start) &
                      (combined["Date"] <= test_end)].copy()
        if len(tr) < 100 or len(te) < 10:
            continue

        stats = (tr.groupby("Equipment_ID")[TARGET]
                   .agg(thermo_mean_runtime="mean", thermo_std_runtime="std")
                   .reset_index())
        for col in ["thermo_mean_runtime", "thermo_std_runtime"]:
            tr = tr.drop(columns=[col], errors="ignore")
            te = te.drop(columns=[col], errors="ignore")
        tr = tr.merge(stats, on="Equipment_ID", how="left")
        te = te.merge(stats, on="Equipment_ID", how="left")
        te = te.dropna(subset=["thermo_mean_runtime"])

        feats = _avail(tr)
        tr = tr.dropna(subset=[TARGET])
        te = te.dropna(subset=[TARGET])

        preds, _ = _fit_predict(tr[feats], tr[TARGET], te[feats])
        s = _score(te[TARGET], preds)
        fold_results.append({"fold": fold+1, **s})
        table_row([fold+1, f"{len(tr):,}", f"{len(te):,}",
                   f"{s['r2']:+.3f}", f"{s['rmse']:.4f}", f"{s['mae']:.4f}"], cols)

    if fold_results:
        rdf = pd.DataFrame(fold_results)
        print(f"\n  {'metric':<6}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
        print(f"  {'──────':<6}  {'───────':>7}  {'───────':>7}  {'───────':>7}  {'───────':>7}")
        for metric in ["r2", "rmse", "mae"]:
            v = rdf[metric]
            print(f"  {metric:<6}  {v.mean():>7.4f}  {v.std():>7.4f}"
                  f"  {v.min():>7.4f}  {v.max():>7.4f}")
        r2_range = rdf["r2"].max() - rdf["r2"].min()
        stability = ("stable" if r2_range < 0.05
                     else "moderate variance" if r2_range < 0.15
                     else "high variance")
        print(f"\n  r² range {r2_range:.3f} → {stability}")

    return pd.DataFrame(fold_results) if fold_results else pd.DataFrame()




def print_comparison(results: dict[str, dict]):
    banner("STRATEGY COMPARISON SUMMARY")

    cols = [("strategy", 32, "l"), ("r²", 7, "r"), ("rmse", 7, "r"),
            ("mae", 7, "r"), ("notes", 22, "l")]
    table_header(cols)

    notes = {
        "pooled":    "single global model",
        "local":     "one model / thermostat",
        "seasonal":  "four season-specific models",
        "clustered": "four behavior-cluster models",
    }
    best_r2 = max(v["r2"] for v in results.values())
    for name, s in results.items():
        marker = " ◀ best" if abs(s["r2"] - best_r2) < 1e-6 else ""
        table_row([
            name.capitalize(),
            f"{s['r2']:+.3f}",
            f"{s['rmse']:.3f}",
            f"{s['mae']:.3f}",
            notes.get(name, "") + marker,
        ], cols)

    print()
    winner = max(results, key=lambda k: results[k]["r2"])
    print(f"  best strategy by r²: {winner.upper()}")
    print(f"  (note: local models may overfit on thermostats with sparse data)")
    print()


# =============================================================================
# SECTION 9 — REPORTING CHARTS
# =============================================================================

def plot_results(
    strategy_results: dict,
    pooled_per_thermo: pd.Series,
    local_per_thermo:  pd.Series,
    feature_imp:       pd.Series,
    monthly_metrics:   dict,
    seasonal_detail:   dict,
    cluster_detail:    dict,
    tscv_df:           pd.DataFrame,
    pooled_scatter:    pd.DataFrame,
    out_dir:           str = "./xgb_results_two/final/",
):
    """Generate report-ready figures and save to out_dir."""

    # ── shared style ──────────────────────────────────────────────────────────
    DARK   = "#0d1117"
    MID    = "#161b22"
    PANEL  = "#21262d"
    BORDER = "#30363d"
    TEXT   = "#e6edf3"
    MUTED  = "#8b949e"
    ACCENT = "#58a6ff"
    GREEN  = "#3fb950"
    AMBER  = "#d29922"
    RED    = "#f85149"
    PURPLE = "#bc8cff"
    TEAL   = "#39d353"

    STRATEGY_COLORS = {
        "pooled":    ACCENT,
        "local":     GREEN,
        "seasonal":  AMBER,
        "clustered": PURPLE,
    }
    SEASON_COLORS = {
        "Winter": "#58a6ff",
        "Spring": "#3fb950",
        "Summer": "#f85149",
        "Autumn": "#d29922",
    }

    plt.rcParams.update({
        "figure.facecolor":  DARK,
        "axes.facecolor":    MID,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # =========================================================================
    # FIGURE 1 — strategy overview  (2×3 grid)
    # =========================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig1.patch.set_facecolor(DARK)
    fig1.suptitle("HVAC Runtime Prediction — Strategy Overview",
                  fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    # ── 1a: strategy comparison bar chart (r², rmse, mae) ────────────────────
    ax = axes[0, 0]
    strategies = list(strategy_results.keys())
    metrics    = ["r2", "rmse", "mae"]
    metric_labels = ["R²", "RMSE", "MAE"]
    x = np.arange(len(strategies))
    width = 0.25
    bar_colors = [ACCENT, AMBER, RED]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, bar_colors)):
        vals = [strategy_results[s][metric] for s in strategies]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor=BORDER, linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in strategies], fontsize=9)
    ax.set_title("Strategy Comparison (R², RMSE, MAE)", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 1b: pooled per-thermostat r² histogram ────────────────────────────────
    ax = axes[0, 1]
    vals = pooled_per_thermo.values
    ax.hist(vals, bins=30, color=ACCENT, edgecolor=DARK, alpha=0.85)
    ax.axvline(np.median(vals), color=AMBER, linestyle="--", linewidth=1.5,
               label=f"median {np.median(vals):+.3f}")
    ax.axvline(np.mean(vals),   color=GREEN,  linestyle=":",  linewidth=1.5,
               label=f"mean   {np.mean(vals):+.3f}")
    ax.set_xlabel("R² per thermostat", fontsize=9)
    ax.set_ylabel("Thermostats", fontsize=9)
    ax.set_title("Pooled Model — Per-Thermostat R² Distribution", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 1c: local per-thermostat r² histogram ────────────────────────────────
    ax = axes[0, 2]
    vals_l = local_per_thermo.values
    ax.hist(vals_l, bins=30, color=GREEN, edgecolor=DARK, alpha=0.85)
    ax.axvline(np.median(vals_l), color=AMBER, linestyle="--", linewidth=1.5,
               label=f"median {np.median(vals_l):+.3f}")
    ax.axvline(np.mean(vals_l),   color=ACCENT, linestyle=":",  linewidth=1.5,
               label=f"mean   {np.mean(vals_l):+.3f}")
    ax.set_xlabel("R² per thermostat", fontsize=9)
    ax.set_ylabel("Thermostats", fontsize=9)
    ax.set_title("Local Model — Per-Thermostat R² Distribution", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 1d: pooled vs local scatter (per thermostat r²) ─────────────────────
    ax = axes[1, 0]
    shared = sorted(set(pooled_per_thermo.index) & set(local_per_thermo.index))
    p_r2 = [pooled_per_thermo[e] for e in shared]
    l_r2 = [local_per_thermo[e]  for e in shared]
    colors_pt = [GREEN if l > p else RED for p, l in zip(p_r2, l_r2)]
    ax.scatter(p_r2, l_r2, c=colors_pt, alpha=0.45, s=18, edgecolors="none")
    lim_lo = min(min(p_r2), min(l_r2)) - 0.05
    lim_hi = max(max(p_r2), max(l_r2)) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=MUTED,
            linestyle="--", linewidth=1.2, label="parity")
    ax.set_xlabel("Pooled R²",  fontsize=9)
    ax.set_ylabel("Local R²",   fontsize=9)
    ax.set_title("Pooled vs Local R² (per thermostat)", fontsize=10, pad=8)
    wins_local = sum(l > p for p, l in zip(p_r2, l_r2))
    ax.text(0.04, 0.92, f"local better: {wins_local}/{len(shared)}",
            transform=ax.transAxes, fontsize=8, color=GREEN)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL)
    ax.set_facecolor(MID)
    ax.grid(True, alpha=0.3)

    # ── 1e: monthly r² (pooled model) ────────────────────────────────────────
    ax = axes[1, 1]
    if monthly_metrics:
        months = sorted(monthly_metrics.keys())
        m_r2   = [monthly_metrics[m]["r2"]   for m in months]
        m_rmse = [monthly_metrics[m]["rmse"] for m in months]
        x_m    = range(len(months))
        ax.bar(x_m, m_r2, color=ACCENT, alpha=0.8, edgecolor=DARK, linewidth=0.4,
               label="R²")
        ax2 = ax.twinx()
        ax2.plot(x_m, m_rmse, color=AMBER, marker="o", markersize=5,
                 linewidth=1.5, label="RMSE")
        ax2.set_ylabel("RMSE", fontsize=9, color=AMBER)
        ax2.tick_params(axis="y", colors=AMBER)
        ax2.spines["right"].set_edgecolor(AMBER)
        ax.set_xticks(list(x_m))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("R²", fontsize=9)
        ax.set_title("Pooled Model — Monthly R² & RMSE", fontsize=10, pad=8)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, framealpha=0.2, facecolor=PANEL, loc="lower left")
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 1f: TSCV fold metrics ─────────────────────────────────────────────────
    ax = axes[1, 2]
    if not tscv_df.empty:
        folds = tscv_df["fold"].astype(int).tolist()
        ax.plot(folds, tscv_df["r2"],   color=ACCENT, marker="o", linewidth=2,
                markersize=6, label="R²")
        ax.fill_between(folds, tscv_df["r2"], alpha=0.15, color=ACCENT)
        ax2b = ax.twinx()
        ax2b.plot(folds, tscv_df["rmse"], color=AMBER, marker="s", linewidth=1.8,
                  markersize=5, linestyle="--", label="RMSE")
        ax2b.set_ylabel("RMSE", fontsize=9, color=AMBER)
        ax2b.tick_params(axis="y", colors=AMBER)
        ax2b.spines["right"].set_edgecolor(AMBER)
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_ylabel("R²",   fontsize=9)
        ax.set_title("Time-Series Cross-Validation", fontsize=10, pad=8)
        ax.set_xticks(folds)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, framealpha=0.2, facecolor=PANEL)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = f"{out_dir}/hvac_report_fig1_strategy_overview.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig1)
    print(f"  saved → {out1}")

    # =========================================================================
    # FIGURE 2 — feature & model detail  (2×3 grid)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    fig2.patch.set_facecolor(DARK)
    fig2.suptitle("HVAC Runtime Prediction — Feature & Model Detail",
                  fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    # ── 2a: top-15 feature importances (horizontal bar) ──────────────────────
    ax = axes2[0, 0]
    top_imp = feature_imp.head(15).iloc[::-1]
    bar_c   = [ACCENT if i >= 10 else GREEN if i >= 5 else AMBER
               for i in range(len(top_imp))]
    ax.barh(top_imp.index, top_imp.values, color=bar_c[::-1],
            edgecolor=DARK, linewidth=0.4, height=0.7)
    ax.set_xlabel("Feature Importance", fontsize=9)
    ax.set_title("Top 15 Feature Importances (Pooled)", fontsize=10, pad=8)
    ax.set_facecolor(MID)
    ax.xaxis.grid(True, alpha=0.4)
    ax.tick_params(axis="y", labelsize=7.5)

    # ── 2b: actual vs predicted scatter (pooled, sample) ─────────────────────
    ax = axes2[0, 1]
    samp = pooled_scatter.sample(min(3000, len(pooled_scatter)), random_state=42)
    sc = ax.scatter(samp["actual"], samp["pred"], c=samp["actual"] - samp["pred"],
                    cmap="RdYlGn", alpha=0.4, s=10, edgecolors="none",
                    vmin=-5, vmax=5)
    hi = max(samp["actual"].max(), samp["pred"].max()) + 0.5
    ax.plot([0, hi], [0, hi], color=MUTED, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual Runtime (hrs)", fontsize=9)
    ax.set_ylabel("Predicted Runtime (hrs)", fontsize=9)
    ax.set_title("Pooled Model — Actual vs Predicted", fontsize=10, pad=8)
    cbar = fig2.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Residual (hrs)", fontsize=8, color=MUTED)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)
    ax.set_facecolor(MID)
    ax.grid(True, alpha=0.3)

    # ── 2c: seasonal r² / rmse grouped bar ───────────────────────────────────
    ax = axes2[0, 2]
    seasons_ordered = ["Winter", "Spring", "Summer", "Autumn"]
    seasons_present = [s for s in seasons_ordered if s in seasonal_detail]
    s_r2   = [seasonal_detail[s]["r2"]   for s in seasons_present]
    s_rmse = [seasonal_detail[s]["rmse"] for s in seasons_present]
    s_x    = np.arange(len(seasons_present))
    w      = 0.35
    bars_r2   = ax.bar(s_x - w/2, s_r2,   w, color=[SEASON_COLORS.get(s, ACCENT) for s in seasons_present],
                       alpha=0.85, edgecolor=DARK, linewidth=0.5, label="R²")
    ax2c = ax.twinx()
    bars_rmse = ax2c.bar(s_x + w/2, s_rmse, w, color=[SEASON_COLORS.get(s, AMBER) for s in seasons_present],
                         alpha=0.55, edgecolor=DARK, linewidth=0.5, label="RMSE")
    ax2c.set_ylabel("RMSE", fontsize=9, color=AMBER)
    ax2c.tick_params(axis="y", colors=AMBER)
    ax2c.spines["right"].set_edgecolor(AMBER)
    ax.set_xticks(s_x)
    ax.set_xticklabels(seasons_present, fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("Seasonal Models — R² & RMSE by Season", fontsize=10, pad=8)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)
    handles = [bars_r2, bars_rmse]
    labels  = ["R²", "RMSE"]
    ax.legend(handles, labels, fontsize=8, framealpha=0.2, facecolor=PANEL)

    # ── 2d: cluster r² bar chart ─────────────────────────────────────────────
    ax = axes2[1, 0]
    clusters_present = sorted(cluster_detail.keys())
    c_r2   = [cluster_detail[c]["r2"]   for c in clusters_present]
    c_rmse = [cluster_detail[c]["rmse"] for c in clusters_present]
    c_x    = np.arange(len(clusters_present))
    cluster_palette = [ACCENT, GREEN, AMBER, PURPLE, RED, TEAL]
    ax.bar(c_x, c_r2, color=[cluster_palette[i % len(cluster_palette)]
                              for i in range(len(clusters_present))],
           alpha=0.85, edgecolor=DARK, linewidth=0.5)
    for xi, (r2, rmse) in enumerate(zip(c_r2, c_rmse)):
        ax.text(xi, r2 + 0.005, f"r²={r2:+.3f}\nrmse={rmse:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_xticks(c_x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters_present], fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("Clustered Model — R² per Behavior Cluster", fontsize=10, pad=8)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 2e: per-thermostat delta (local - pooled) distribution ───────────────
    ax = axes2[1, 1]
    shared_ids = sorted(set(pooled_per_thermo.index) & set(local_per_thermo.index))
    deltas = [local_per_thermo[e] - pooled_per_thermo[e] for e in shared_ids]
    pos_c = [GREEN if d > 0 else RED for d in deltas]
    sorted_deltas = sorted(deltas)
    ax.bar(range(len(sorted_deltas)), sorted_deltas,
           color=[GREEN if d > 0 else RED for d in sorted_deltas],
           alpha=0.7, edgecolor="none", width=1.0)
    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_xlabel("Thermostat rank (sorted by Δ)", fontsize=9)
    ax.set_ylabel("Local R² − Pooled R²", fontsize=9)
    ax.set_title("Per-Thermostat: Local minus Pooled R²", fontsize=10, pad=8)
    pct_better = sum(d > 0 for d in deltas) / len(deltas) * 100
    ax.text(0.02, 0.92, f"{pct_better:.0f}% favour local",
            transform=ax.transAxes, fontsize=8, color=GREEN)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    # ── 2f: TSCV fold R² stability box-style annotation ───────────────────────
    ax = axes2[1, 2]
    metric_names = ["r2", "rmse", "mae"]
    metric_display = ["R²", "RMSE", "MAE"]
    if not tscv_df.empty:
        data_to_plot = [tscv_df[m].values for m in metric_names]
        bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                        medianprops=dict(color=DARK, linewidth=2))
        box_colors = [ACCENT, AMBER, RED]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set(color=MUTED, linewidth=1.2)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(metric_display, fontsize=9)
        ax.set_title("TSCV Metric Stability (5 folds)", fontsize=10, pad=8)
        ax.set_ylabel("Value", fontsize=9)

        # annotate mean ± std
        for xi, (m, label) in enumerate(zip(metric_names, metric_display), start=1):
            v = tscv_df[m]
            ax.text(xi, v.max() + (v.max() - v.min()) * 0.08,
                    f"μ={v.mean():.3f}\nσ={v.std():.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_facecolor(MID)
    ax.yaxis.grid(True, alpha=0.4)

    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = f"{out_dir}/hvac_report_fig2_feature_detail.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig2)
    print(f"  saved → {out2}")

    print(f"\n  2 figures saved to {out_dir}/")


# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

def main():
    # ── load ──────────────────────────────────────────────────────────────────
    all_daily = load_data(min_days=90)
    if not all_daily:
        print("  no thermostats loaded.")
        return

    combined = pd.concat(all_daily, ignore_index=True)
    combined = add_lag_features(combined)
    combined = combined.dropna(subset=[TARGET])

    n_thermos = combined["Equipment_ID"].nunique()
    date_min  = combined["Date"].min().date()
    date_max  = combined["Date"].max().date()

    banner("DATASET OVERVIEW")
    print(f"\n  rows         : {len(combined):,}")
    print(f"  thermostats  : {n_thermos}")
    print(f"  date range   : {date_min} → {date_max}")

    # ── global 80/20 time split ────────────────────────────────────────────────
    cutoff = combined["Date"].quantile(0.8)
    train  = combined[combined["Date"] <= cutoff].copy()
    test   = combined[combined["Date"] >  cutoff].copy()
    train, test = add_thermostat_stats(train, test)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test  = test.dropna(subset=[TARGET]).reset_index(drop=True)

    print(f"\n  train/test cutoff : {cutoff.date()}")
    print(f"  train rows        : {len(train):,}  "
          f"({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"  test  rows        : {len(test):,}  "
          f"({test['Date'].min().date()} → {test['Date'].max().date()})")

    # ── time-series cross-validation ─────────────────────────────────────────
    tscv_df = run_tscv(combined.copy())

    # ── strategy 1: pooled ───────────────────────────────────────────────────
    pooled_overall, pooled_per_thermo, feature_imp, monthly_metrics, pooled_scatter = \
        run_pooled(train.copy(), test.copy())

    # ── strategy 2: local ────────────────────────────────────────────────────
    local_overall, local_per_thermo = run_local(train.copy(), test.copy())

    # ── strategy 3: seasonal ─────────────────────────────────────────────────
    seasonal_overall, seasonal_detail = run_seasonal(
        combined.copy(), n_optuna_trials=SEASONAL_OPTUNA_TRIALS)

    # ── strategy 4: clustered ────────────────────────────────────────────────
    clustered_overall, cluster_detail = run_clustered(
        train.copy(), test.copy(), n_clusters=4)

    # ── head-to-head comparison ───────────────────────────────────────────────
    strategy_results = {
        "pooled":    pooled_overall,
        "local":     local_overall,
        "seasonal":  seasonal_overall,
        "clustered": clustered_overall,
    }
    print_comparison(strategy_results)

    # ── per-thermostat pooled vs local ────────────────────────────────────────
    banner("POOLED vs LOCAL  (per-thermostat r²)")
    shared = sorted(set(pooled_per_thermo.index) & set(local_per_thermo.index))
    diffs  = [(eid,
               pooled_per_thermo[eid],
               local_per_thermo[eid],
               local_per_thermo[eid] - pooled_per_thermo[eid])
              for eid in shared]
    diffs.sort(key=lambda x: x[3], reverse=True)

    cols = [("equipment_id", 45, "l"), ("pooled r²", 10, "r"),
            ("local r²", 9, "r"), ("Δ", 8, "r")]
    section("thermostats where LOCAL beats pooled by most")
    table_header(cols)
    for eid, pr2, lr2, delta in diffs[:8]:
        table_row([str(eid)[:45], f"{pr2:+.3f}", f"{lr2:+.3f}", f"{delta:+.3f}"], cols)

    section("thermostats where POOLED beats local by most")
    table_header(cols)
    for eid, pr2, lr2, delta in diffs[-8:]:
        table_row([str(eid)[:45], f"{pr2:+.3f}", f"{lr2:+.3f}", f"{delta:+.3f}"], cols)

    wins_local  = sum(1 for *_, d in diffs if d > 0)
    wins_pooled = sum(1 for *_, d in diffs if d <= 0)
    print(f"\n  local beats pooled : {wins_local}/{len(diffs)} thermostats")
    print(f"  pooled beats local : {wins_pooled}/{len(diffs)} thermostats")
    print()

    # ── generate report figures ───────────────────────────────────────────────
    banner("GENERATING REPORT FIGURES")
    plot_results(
        strategy_results  = strategy_results,
        pooled_per_thermo = pooled_per_thermo,
        local_per_thermo  = local_per_thermo,
        feature_imp       = feature_imp,
        monthly_metrics   = monthly_metrics,
        seasonal_detail   = seasonal_detail,
        cluster_detail    = cluster_detail,
        tscv_df           = tscv_df,
        pooled_scatter    = pooled_scatter,
        out_dir           = ".",
    )


if __name__ == "__main__":
    main()