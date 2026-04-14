import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "../../Sample_Data/Processed/timeseries_table_6_daily.csv"  
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape            : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Date range       : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Equipment IDs    : {df['Equipment_ID'].nunique()}  →  {df['Equipment_ID'].unique()}")
print(f"Weekends in data : {df['is_weekend'].sum()} / {len(df)} days")
print()

print("=" * 60)
print("MISSING VALUES (columns with any nulls)")
print("=" * 60)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("  None")
else:
    print(missing.to_string())
print()

print("=" * 60)
print("DESCRIPTIVE STATISTICS (numeric columns)")
print("=" * 60)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(df[num_cols].describe().T.to_string())
print()

ops_cols = [
    "daily_heating_hours", "daily_cooling_hours",
    "daily_off_hours", "daily_runtime_hours",
    "daily_fan_on_hours", "fan_runtime_ratio",
    "setpoint_change_count",
    "occupied_ping_count", "unoccupied_ping_count",
]
ops_cols = [c for c in ops_cols if c in df.columns]

print("=" * 60)
print("OPERATIONAL COLUMNS SUMMARY")
print("=" * 60)
print(df[ops_cols].describe().T.to_string())
print()

key_cols = [
    "daily_heating_hours", "daily_cooling_hours",
    "daily_off_hours", "daily_runtime_hours",
    "indoor_temp_time_weighted_mean", "setpoint_time_weighted_mean",
    "outdoor_temp_time_weighted_mean", "outdoor_humidity_time_weighted_mean",
    "setpoint_gap_mean", "occupied_ping_count",
    "indoor_temp_range", "outdoor_temp_range",
    "temp_gradient_mean",
]
key_cols = [c for c in key_cols if c in df.columns]
corr = df[key_cols].corr()

print("=" * 60)
print("TOP CORRELATIONS WITH setpoint_gap_mean")
print("=" * 60)
if "setpoint_gap_mean" in corr:
    gap_corr = corr["setpoint_gap_mean"].drop("setpoint_gap_mean").sort_values(key=abs, ascending=False)
    print(gap_corr.to_string())
print()

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# 1. Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix – Key Features", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_correlation_heatmap.png")
plt.close()
print("Saved: 01_correlation_heatmap.png")

# 2. Temperature overview: indoor vs outdoor vs setpoint over time
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df["indoor_temp_time_weighted_mean"], label="Indoor (TWM)", marker="o", ms=4)
ax.plot(df["Date"], df["setpoint_time_weighted_mean"], label="Setpoint (TWM)", linestyle="--", marker="s", ms=4)
ax.plot(df["Date"], df["outdoor_temp_time_weighted_mean"], label="Outdoor (TWM)", linestyle=":", marker="^", ms=4)
ax.fill_between(df["Date"],
                df["indoor_temp_min"], df["indoor_temp_max"],
                alpha=0.12, label="Indoor min–max range")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C or °F)")
ax.set_title("Temperature Timeline: Indoor / Setpoint / Outdoor")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_temperature_timeline.png")
plt.close()
print("Saved: 02_temperature_timeline.png")

# 3. Daily mode hours stacked bar
fig, ax = plt.subplots(figsize=(10, 4))
mode_cols = ["daily_heating_hours", "daily_cooling_hours", "daily_off_hours", "daily_unknown_hours"]
mode_cols = [c for c in mode_cols if c in df.columns]
df_mode = df[["Date"] + mode_cols].set_index("Date")
df_mode.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Hours")
ax.set_title("Daily Operating Mode Hours")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_daily_mode_hours.png")
plt.close()
print("Saved: 03_daily_mode_hours.png")

# 4. Setpoint gap over time + outdoor humidity
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.bar(df["Date"], df["setpoint_gap_mean"], color="steelblue", alpha=0.7, label="Setpoint Gap Mean")
ax2.plot(df["Date"], df["outdoor_humidity_time_weighted_mean"], color="darkorange",
         marker="o", ms=4, label="Outdoor Humidity (TWM)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Setpoint Gap (°)")
ax2.set_ylabel("Humidity (%)")
ax1.set_title("Setpoint Gap vs Outdoor Humidity")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_setpoint_gap_vs_humidity.png")
plt.close()
print("Saved: 04_setpoint_gap_vs_humidity.png")

# 5. Indoor temp distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["indoor_temp_time_weighted_mean"].hist(bins=15, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Indoor Temp TWM – Distribution")
axes[0].set_xlabel("Temp")
axes[0].set_ylabel("Count")

df["indoor_temp_range"].hist(bins=15, ax=axes[1], color="salmon", edgecolor="white")
axes[1].set_title("Indoor Temp Daily Range")
axes[1].set_xlabel("Range (max − min)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_indoor_temp_distributions.png")
plt.close()
print("Saved: 05_indoor_temp_distributions.png")

# 6. Outdoor temp & humidity daily range ribbon
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(df["Date"], df["outdoor_temp_min"], df["outdoor_temp_max"],
                alpha=0.35, color="tab:orange", label="Outdoor Temp Range")
ax.plot(df["Date"], df["outdoor_temp_time_weighted_mean"], color="tab:orange", lw=2, label="Outdoor Temp TWM")
ax2 = ax.twinx()
ax2.fill_between(df["Date"], df["outdoor_hum_min"], df["outdoor_hum_max"],
                 alpha=0.2, color="tab:blue", label="Humidity Range")
ax2.plot(df["Date"], df["outdoor_humidity_time_weighted_mean"], color="tab:blue", lw=2, label="Humidity TWM")
ax.set_xlabel("Date")
ax.set_ylabel("Outdoor Temp")
ax2.set_ylabel("Humidity (%)")
ax.set_title("Outdoor Conditions: Temperature & Humidity Ribbon")
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_outdoor_conditions_ribbon.png")
plt.close()
print("Saved: 06_outdoor_conditions_ribbon.png")

# 7. Pairplot of core variables
pair_cols = [
    "setpoint_gap_mean", "indoor_temp_time_weighted_mean",
    "outdoor_temp_time_weighted_mean", "outdoor_humidity_time_weighted_mean",
    "temp_gradient_mean",
]
pair_cols = [c for c in pair_cols if c in df.columns]
pair_df = df[pair_cols].dropna()
if len(pair_df) > 1:
    pg = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.7, "s": 40})
    pg.fig.suptitle("Pairplot – Core Environmental Variables", y=1.02)
    pg.fig.savefig(OUTPUT_DIR / "07_pairplot_core_vars.png", bbox_inches="tight")
    plt.close()
    print("Saved: 07_pairplot_core_vars.png")

# 8. Weekend vs weekday boxplot
if "is_weekend" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["setpoint_gap_mean", "indoor_temp_range"]):
        if col in df.columns:
            df.boxplot(column=col, by="is_weekend", ax=ax)
            ax.set_title(col)
            ax.set_xlabel("is_weekend (0=weekday, 1=weekend)")
    plt.suptitle("Weekend vs Weekday Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_weekend_vs_weekday.png")
    plt.close()
    print("Saved: 08_weekend_vs_weekday.png")

print()
print("=" * 60)
print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")
print("=" * 60)