# linear regression

# gradient boosted tree
# xgboost

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def main(inside_path="../../Sample_Data/Processed/processed_inside.csv", outside_path="outsideweather.csv"):
    df = load_and_merge(inside_path, outside_path)

if __name__ == "__main__":
    import sys
    inside_path  = sys.argv[1] if len(sys.argv) > 1 else "inside.csv"
    outside_path = sys.argv[2] if len(sys.argv) > 2 else "outside.csv"
    main(inside_path, outside_path)