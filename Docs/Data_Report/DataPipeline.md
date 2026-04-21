# Data Pipeline Report

## 1. Overview of the Data Pipeline

This data pipeline is designed to process and prepare HVAC indoor sensor data along with outdoor weather data for analysis. The system takes raw CSV files as input, applies a series of cleaning and transformation steps, and produces a final, structured dataset.

The main script (`dataPrep.py`) handles the full pipeline, including merging indoor and outdoor data. A supporting script (`outsideDataPrep.py`) is used to clean and standardize the weather data before it is used in the main pipeline.

## 2. Data Sources

### Indoor Data

- Multiple CSV files stored in an input directory
- Contains HVAC-related information such as:
  - Temperature
  - Setpoint
  - Fan state
  - Running mode

These files are collected and processed in `dataPrep.py`:

```python
indoor_csvs = [
    f for f in glob.glob(os.path.join(input_dir, "*.csv"))
    if os.path.abspath(f) != os.path.abspath(outdoor_path)
]
```

### Outdoor Data

- A single CSV file containing weather data
- Includes:

  - Outdoor temperature
  - Minimum and maximum temperature
  - Humidity

Before merging, this data is cleaned in `outsideDataPrep.py`.

## 3. Outdoor Data Preprocessing (`outsideDataPrep.py`)

The outdoor data is first standardized and cleaned before entering the main pipeline.

### Column Renaming

Columns are converted to a consistent naming format:

```python
df = df.rename(columns={
    "Temperature": "outside_temp",
    "outsideMinTemp": "outside_temp_min",
    "outsideMaxTemp": "outside_temp_max",
    "outsideHumidity": "outside_humidity"
})
```

### Data Type Conversion

All relevant columns are converted to numeric values:

```python
for col in ["outside_temp", "outside_temp_min", "outside_temp_max", "outside_humidity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

### Handling Missing Values

Missing values are handled using forward and backward fill:

```python
df["outside_temp_min"] = df["outside_temp_min"].ffill().bfill()
```

### Sorting and Saving

The dataset is sorted by timestamp and saved to a processed folder:

```python
df = df.sort_values("Timestamp").reset_index(drop=True)

processed_dir = os.path.join(base_dir, "Processed")
df.to_csv(output_path, index=False)

```

This preprocessing step ensures that the outdoor data is clean and consistent before it is merged with indoor data.

## 4. Pipeline Flow

The main pipeline in `dataPrep.py` follows these steps:

### Step 1: Data Ingestion

- Load indoor and outdoor CSV files using pandas:

```python
pd.read_csv(..., parse_dates=["Timestamp"])
```

### Step 2: Data Cleaning

- Handle missing values using interpolation and fill methods:

```python
df["Temperature"] = df["Temperature"].astype(float).interpolate().bfill().ffill()
```

### Step 3: Data Transformation

- Rename columns for consistency:

```python
df = df.rename(columns={"temp": "Temperature"})
```

- Encode categorical data:

```python
df = pd.get_dummies(df, columns=["RunningMode"])
```

### Step 4: Data Integration

- Merge indoor and outdoor datasets:

```python
merged = pd.merge(df, outdoor_df, on="Timestamp", how="outer")
```

### Step 5: Data Output

- Save processed data:

```python
df.to_csv(output_path, index=False)
```

## 5. Data Movement Frequency

This is a **batch processing pipeline**:

- Data is processed only when the script is run
- There is no real-time or streaming component
- Suitable for periodic or offline data analysis

## 6. Data Processing Steps

### Standardizing Column Names

- Ensures consistency across datasets

### Handling Missing Values

- Interpolation for continuous values (temperature)
- Forward/backward fill for discrete values

### Encoding Categorical Data

- Running modes converted into binary columns:

```python
RunningMode_off, RunningMode_heat, RunningMode_cool
```

### Sorting Data

- Maintains correct time order:

```python
df.sort_values("Timestamp")
```

## 7. Data Merging Strategy

- Uses an **outer join** to combine datasets:

```python
how="outer"
```

- Aligns outdoor data using time-based interpolation:

```python
merged["Outdoor_Temperature"].interpolate(method="time")
```

- Fills remaining gaps:

```python
merged[cols].ffill().bfill()
```

- Filters back to indoor timestamps after merging

This ensures that every indoor record has corresponding outdoor data.

## 8. Feature Engineering

### Cumulative Runtime

```python
df["CumulativeRuntime"] = df["runtime_change"].cumsum()
```

### Cumulative Cost

```python
df["CumulativeCost"] = (df["CumulativeRuntime"] / 3600.0) * 0.17 * 3.0
```

### Running Mode Indicators

- One-hot encoded columns for system modes

These features help analyze system performance and cost over time.

## 9. Final Dataset

The final dataset includes:

- Clean indoor HVAC data
- Processed outdoor weather data
- Engineered features

Unnecessary columns are removed:

```python
df.drop(columns=[...], errors="ignore")
```

Output files are saved as:

```text
processed_<filename>.csv
```

## 10. Logical Diagram

```
Raw Indoor CSV Files        Raw Outdoor CSV
          │                      │
          │        		(outsideDataPrep.py)
          │                      ↓
          │            	Cleaned Outdoor Data
          └──────────┬───────────┘
                     ↓
               Data Cleaning
                     ↓
            Data Transformation
                     ↓
               Data Merging
                     ↓
	        Feature Engineering
                     ↓
            Final Processed Data

```

## Conclusion

This pipeline provides a clear and structured way to process HVAC and weather data. The use of a separate preprocessing step for outdoor data (`outsideDataPrep.py`) improves data quality before merging. Combined with cleaning, transformation, and feature engineering in `dataPrep.py`, the pipeline produces a reliable dataset that is ready for analysis or modeling.
