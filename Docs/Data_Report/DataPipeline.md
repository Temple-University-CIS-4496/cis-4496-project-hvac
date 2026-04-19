1. Overview of Data Pipeline

- Description of pipeline purpose (HVAC + weather data integration)
- Type of pipeline (batch processing using CSV files)


2. Data Sources

- Indoor sensor CSV files
- Outdoor weather CSV file


3. Pipeline Flow (Logical Steps)

- Ingestion (load CSVs)
- Cleaning (handle missing values, type casting)
- Transformation (feature engineering, encoding, interpolation)
- Integration (merge indoor + outdoor data on timestamp)
- Output (write processed CSVs)


4. Data Movement Frequency

- Batched pipeline (runs manually or periodically on stored data)


5. Data Processing Steps

- Standardizing column names
- Handling missing values (ffill, bfill, interpolation)
- Encoding categorical variables (RunningMode → one-hot)
- Time-based calculations (runtime, cost)


6. Data Merging Strategy

- Outer join on timestamp
- Time interpolation for alignment
- Forward/backward fill for environmental variables


7. Feature Engineering

- Cumulative runtime
- Cumulative cost
- Running mode indicators (heat/cool/off)


8. Final Dataset Structure

- Cleaned + merged dataset
- Ready for analysis/modeling


9. Logical Diagram

- Simple flow:  Raw Indoor + Raw Outdoor → Cleaning → Transformation → Merge → Final Dataset

