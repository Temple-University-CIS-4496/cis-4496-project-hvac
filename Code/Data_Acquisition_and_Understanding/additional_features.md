# Feature List for XGBoost

## Lag Features

- `runtime_lag1` - The runtime from previous hour. Short term predictor: if system ran hard in the last hour, likely will run this hour too.
- `runtime_lag24` - The runtime from the same hour yesterday. Captures the "same time yesterday" baseline. Recurring daily patterns.
- `runtime_roll3` - A 3-hour rolling average of the prior hours' runtime (shifted by 1 to avoid leakage from runtime_lag1). Smooths out noise and gives a short-term trend signal. **was most important feature for xgboost**.

```py
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
```
