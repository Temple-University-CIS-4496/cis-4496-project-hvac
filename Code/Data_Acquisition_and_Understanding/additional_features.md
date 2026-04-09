# Feature List for XGBoost

## Lag Features

- `runtime_lag1` - The runtime from previous hour. Short term predictor: if system ran hard in the last hour, likely will run this hour too.
- `runtime_lag24` - The runtime from the same hour yesterday. Captures the "same time yesterday" baseline. Recurring daily patterns.
- `runtime_roll3` - A 3-hour rolling average of the prior hours' runtime (shifted by 1 to avoid leakage from runtime_lag1). Smooths out noise and gives a short-term trend signal. **was most important feature for xgboost**.
