import argparse
import os
import pandas as pd

def main(input_path: str):
    df = pd.read_csv(
        input_path,
        sep=";",
        parse_dates=["Timestamp"]
    )

    df["Temperature"] = df["Temperature"].astype(float)

    df["Setpoint"] = (
        df["Setpoint"]
        .ffill()
        .fillna(0)
    )

    df["FanState"] = (
        df["FanState"]
        .map({"on": 1, "off": 0})
        .ffill()
        .fillna(0)
        .astype(int)
    )

    df["OutputState"] = (
        df["output_state"]
        .ffill()
        .fillna("idle")
        .ne("idle")
        .astype(int)
    )

    df["RunningMode"] = (
        df["running_mode"]
        .map({"cool": 1, "heat": 1, "off": 0})
        .ffill()
        .fillna(0)
        .astype(int)
    )

    df = df.drop(
        columns=["Mode", "Occupied", "RentalStatus", "output_state", "running_mode"],
        errors="ignore"
    )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    input_dir, filename = os.path.split(input_path)
    base_dir = os.path.dirname(input_dir)  # Sample_Data
    processed_dir = os.path.join(base_dir, "Processed")

    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, filename)
    df.to_csv(output_path, index=False)

    print(f"Processed file saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HVAC touchcombo data")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to raw CSV file (e.g. Sample_Data/Raw/file.csv)"
    )

    args = parser.parse_args()
    main(args.input_path)