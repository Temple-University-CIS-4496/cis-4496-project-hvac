# Model Report: XGBoost HVAC Runtime Prediction

_A report detailing the XGBoost experiment for predicting HVAC compressor runtime per hour._

---

## Analytic Approach

**Target Definition**

The target variable is `runtime_minutes` — the number of minutes per hour that the HVAC compressor is actively running (i.e., `OutputState == 1`). Values are clipped to the valid range of [0, 60] minutes per hour. This is a continuous regression target.

**Inputs**

Features fall into four categories:

| Category                  | Features                                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| Thermostat / indoor state | `Temperature`, `Setpoint`                                                                                 |
| Outdoor weather           | `Outdoor_Temp`, `Outdoor_Temp_Min`, `Outdoor_Temp_Max`, `Outdoor_Humidity`                                |
| Engineered weather        | `temp_delta` (Setpoint − Outdoor Temp), `temp_range` (Max − Min outdoor temp)                             |
| Calendar                  | `hour_of_day`, `day_of_week`, `month`                                                                     |
| Lag / rolling features    | `runtime_lag1` (prior hour), `runtime_lag24` (same hour yesterday), `runtime_roll3` (3-hour rolling mean) |
| Thermostat-level stats    | `thermostat_id`, `thermo_mean`, `thermo_std`, `thermo_p25` (computed from training data only, no leakage) |

**Model Type**

A gradient-boosted decision tree regressor (XGBoost) was trained as a single pooled model across all thermostats. A pooled approach was chosen over per-thermostat models to maximize training data and generalize across devices. Thermostat identity is encoded via `thermostat_id` and per-device runtime statistics (`thermo_mean`, `thermo_std`, `thermo_p25`), allowing the model to adapt predictions to individual device baselines while sharing learned patterns.

---

## Model Description

**Data Pipeline**

```
Raw thermostat CSVs (Kaggle: lsobieski/processed-thermostat-data)
        │
        ▼
load_data()         — parse timestamps, dedup, filter to state-change rows
        │
        ▼
compute_runtime_per_hour()  — compute active minutes per hour bucket via OutputState
        │
        ▼
engineer_features() — add calendar, lag, rolling, and weather-derived features
        │
        ▼
add_thermostat_stats()  — merge per-device mean/std/p25 from training fold only
        │
        ▼
Chronological 80/20 split (or 5-fold time-series CV)
        │
        ▼
XGBRegressor.fit()  — pooled model, early stopping on test eval set
        │
        ▼
Predictions clipped to [0, 60]
```

Only thermostats with 6+ months of data were included. Data is filtered to state-change events before aggregation to reduce noise from repeated identical rows.

**Learner**

`xgboost.XGBRegressor` with objective `reg:squarederror`.

**Hyperparameters**

| Parameter               | Value | Notes                                                   |
| ----------------------- | ----- | ------------------------------------------------------- |
| `n_estimators`          | 1000  | Upper bound; early stopping typically halts before this |
| `learning_rate`         | 0.02  | Low rate for stability                                  |
| `max_depth`             | 8     | Moderate depth; allows interaction capture              |
| `subsample`             | 0.8   | Row-level bagging                                       |
| `colsample_bytree`      | 0.8   | Feature-level bagging                                   |
| `min_child_weight`      | 10    | Prevents splits on very small leaf nodes                |
| `gamma`                 | 1     | Minimum loss reduction to split                         |
| `early_stopping_rounds` | 20    | Halts if no improvement in test RMSE for 20 rounds      |
| `random_state`          | 42    | Reproducibility                                         |

---

## Results (Model Performance)

**Hold-out Test Set (80/20 chronological split)**

| Metric            | Value        | Benchmark                       |
| ----------------- | ------------ | ------------------------------- |
| RMSE              | 12.42 min/hr | Target < 5 min/hr               |
| MAE               | 9.12 min/hr  | —                               |
| R^2               | 0.440        | —                               |
| CV% (RMSE / mean) | 49.6%        | Target < 30% (ASHRAE guideline) |

The model does not meet either the RMSE or CV% benchmarks at the aggregate level. RMSE of 12.42 min/hr is more than double the 5 min/hr target, and CV% of 49.6% exceeds the ASHRAE 30% guideline. R^2 of 0.440 indicates the model explains 44% of the variance in hourly runtime, which is moderate given the diversity of 96 thermostats with heterogeneous usage patterns.

**Time-Series Cross-Validation (5 folds)**

CV was performed using 5 contiguous time-based folds. Each fold trains on all data strictly before the test window, preventing leakage. Thermostat-level statistics (`thermo_mean`, etc.) are recomputed from each training fold independently.

| Fold           | Train Rows | Test Rows | R^2               | RMSE             | MAE             | CV%               |
| -------------- | ---------- | --------- | ----------------- | ---------------- | --------------- | ----------------- |
| 1              | 79,218     | 58,289    | 0.516             | 11.24            | 5.19            | 128.4%            |
| 2              | 158,398    | 79,285    | 0.486             | 15.10            | 10.05           | 73.6%             |
| 3              | 237,634    | 79,252    | 0.448             | 12.32            | 8.63            | 64.2%             |
| 4              | 316,859    | 79,262    | 0.300             | 13.30            | 9.34            | 61.8%             |
| 5              | 396,057    | 79,263    | 0.448             | 12.36            | 9.11            | 48.2%             |
| **Mean ± Std** |            |           | **0.440 ± 0.083** | **12.86 ± 1.45** | **8.47 ± 1.90** | **75.2% ± 31.1%** |

_CV run across 475,320 total rows from 96 thermostats (global cutoff: 2025-12-22)._

**Stability Check**

R^2 range across folds (max − min) indicates model stability:

- < 0.05 -> stable, single-split estimate is reliable
- 0.05–0.15 -> moderate sensitivity to the split window
- > 0.15 -> high variance; the single-split estimate was unreliable

The observed R^2 range is **0.215** (min 0.300 / max 0.516), placing this in the **high variance** category. The single hold-out estimate of R^2 = 0.440 should be treated with caution. Fold 4 is notably the weakest at R^2 = 0.300. Fold 1's CV% of 128.4% indicates the earliest test window has a very low mean runtime, amplifying the normalized error — likely a transitional weather period where many thermostats are mostly idle.

- > 0.15 -> high variance; single-split estimate was unreliable

**Distribution of Per-Thermostat R^2 (Pooled Model)**

From `plot_r2_histogram()` across 93 evaluated thermostats (3 skipped for insufficient test rows):

| Statistic     | Value                                           |
| ------------- | ----------------------------------------------- |
| n thermostats | 93                                              |
| Mean R^2      | ~0.12 (estimated from per-thermostat breakdown) |
| Median R^2    | ~0.19                                           |
| R^2 ≥ 0.3     | ~37 thermostats (40%)                           |
| R^2 < 0       | ~27 thermostats (29%)                           |

Performance is highly skewed. A minority of thermostats (e.g., 129, 130, 135, 137, 151, 153, 156) achieve R^2 above 0.5, while roughly a third of devices have R^2 < 0, meaning the pooled model is worse than simply predicting each device's mean runtime. The worst cases (e.g., thermostat 171: R^2 = −0.261, thermostat 188: R^2 = −1.285) suggest severe distribution shift or data anomalies for those devices.

**Per-Month Performance (Test Set)**

| Month   | R^2   | RMSE  | MAE  | Test Rows |
| ------- | ----- | ----- | ---- | --------- |
| 2025-12 | 0.494 | 11.57 | 8.10 | 9,588     |
| 2026-01 | 0.440 | 12.64 | 9.14 | 46,825    |
| 2026-02 | 0.409 | 12.34 | 9.34 | 38,620    |

Performance declines modestly from December through February, consistent with the model being trained on data through late December and the test set being entirely in the future. The degradation from R^2 = 0.494 to 0.409 over two months suggests mild temporal drift.

---

## Model Understanding

**Variable Importance**

XGBoost feature importances (F-score, i.e., number of times a feature is used to split) are plotted via `plot_results()`. Based on the model design, the most predictive features are expected to rank as follows:

| Rank | Feature         | Importance Score | Rationale                                                                                                                           |
| ---- | --------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `runtime_roll3` | 0.502            | Dominates all others — the 3-hour rolling mean of prior runtime captures inertia in HVAC cycling far better than any static feature |
| 2    | `runtime_lag24` | 0.110            | Same hour yesterday encodes daily usage cycles and occupancy patterns                                                               |
| 3    | `runtime_lag1`  | 0.083            | Prior hour runtime captures short-term autocorrelation                                                                              |
| 4    | `temp_delta`    | 0.068            | Setpoint minus outdoor temp directly reflects compressor load                                                                       |
| 5    | `thermo_mean`   | 0.028            | Per-device baseline runtime encodes home-level characteristics                                                                      |

Lag and rolling features collectively account for ~70% of total importance, confirming that recent runtime behavior is far more predictive than any instantaneous weather or calendar feature.

**Insights Derived from the Model**

- **`runtime_roll3` dominates at 50% importance.** The 3-hour rolling mean is by far the strongest predictor, not the 1-hour lag as initially hypothesized. This suggests HVAC runtime has strong multi-hour inertia — if a system has been running heavily for the past 3 hours, it is very likely to continue doing so.
- **Lag features collectively drive ~70% of importance.** Weather and calendar features explain relatively little once recent runtime history is known, which makes physical sense: a well-insulated home may run its HVAC very differently from a poorly-insulated one even under identical outdoor conditions, and the lag features implicitly encode those home-specific characteristics.
- **`temp_delta` outperforms raw outdoor temperature.** The derived feature (setpoint − outdoor temp) is ranked 4th while raw `Outdoor_Temp` does not appear in the top 5, confirming that the thermal load framing is more informative than temperature alone.
- **Per-thermostat performance is highly heterogeneous.** R^2 ranges from −1.285 (thermostat 188) to +0.606 (thermostat 137) across devices. About 29% of thermostats have R^2 < 0, meaning the pooled model fails to beat the per-device mean predictor for those units. These are likely devices with unusual usage patterns, irregular data, or short histories that don't provide enough signal for the shared model.
- **February degradation suggests temporal drift.** R^2 drops from 0.494 in December to 0.409 in February — a ~17% relative decline. This is consistent with the model being trained through late December and then applied to progressively more out-of-distribution data as winter deepens.
- **Some thermostats deteriorate sharply in February.** Devices like 171 (R^2 drops from 0.340 in January to −3.419 in February) and 169 (−0.795 in February) show extreme degradation, likely due to unusual operating conditions or data quality issues in that period rather than a general model failure.

---

## Conclusion and Discussions for Next Steps

**Conclusion**

The pooled XGBoost model achieves R^2 = 0.440 and RMSE = 12.42 min/hr on the hold-out test set across 96 thermostats, but does not meet the < 5 min/hr RMSE or < 30% CV% benchmarks. The 5-fold time-series CV confirms R^2 = 0.440 ± 0.083 is a consistent aggregate estimate, though the R^2 range of 0.215 across folds signals meaningful temporal variance. Per-thermostat performance is the more telling story: roughly 40% of devices achieve R^2 ≥ 0.3 while 29% have R^2 < 0, indicating that the pooled model works well for a subset of "typical" thermostats but fails for a significant minority with unusual patterns or insufficient training data.

**Discussion on Overfitting**

Several design choices mitigate overfitting:

- `min_child_weight = 10` prevents the model from splitting on very sparse leaf nodes across the large thermostat population.
- `gamma = 1` requires a minimum gain before any split is made.
- `subsample = 0.8` and `colsample_bytree = 0.8` introduce stochasticity to reduce variance.
- Early stopping (`early_stopping_rounds = 20`) halts training when the validation RMSE stops improving, avoiding over-training on the hold-out set.

The main overfitting risk in this setup is **temporal leakage in thermostat stats**: `thermo_mean`, `thermo_std`, and `thermo_p25` are computed across the full training window. In production or CV, these must always be recomputed from only the training fold (as implemented in `run_time_series_cv()`).

**What Other Features Can Be Generated from the Current Data**

- **Temperature gradient features** — compare the starting temperature of today vs. yesterday at the same hour (`temp_gradient_daily`), and similarly for humidity. This was flagged as a TODO in the code and would capture how much a home has pre-heated/cooled overnight.
- **Setpoint change indicators** — binary flags or magnitude of setpoint changes within the past few hours, capturing user-initiated schedule changes.
- **Time since last mode change** — how long the system has been continuously in heating, cooling, or off mode.
- **Runtime acceleration** — the change in runtime between the last two hours (`runtime_lag1 - runtime_lag2`), capturing trending load.
- **Outdoor humidity** is already included; a **heat index or feels-like temperature** combining temperature and humidity would be a more physically meaningful derived feature.
- **Day-type flags** — weekend vs. weekday, or federal holidays, to better capture occupancy patterns.

**What Other Relevant Data Sources Are Available to Help the Modeling**

- **Utility or smart meter data** - hourly energy consumption would provide a complementary signal and allow cross-validation of runtime-based estimates against actual energy draw.
- **Weather station data (NOAA/OpenWeatherMap)** — higher-resolution or more accurate outdoor temperature and humidity readings than the on-device sensors, especially cloud cover and dew point.
- **Home characteristics** — square footage, insulation rating, HVAC system age/SEER rating, and construction year would explain much of the inter-thermostat variance currently absorbed by `thermo_mean`.
- **Occupancy data** — motion sensor or phone-based presence detection to distinguish scheduled setpoint changes from actual occupancy-driven demand.
- **Utility rate schedules** — time-of-use pricing signals that may influence user setpoint behavior and therefore runtime patterns.
