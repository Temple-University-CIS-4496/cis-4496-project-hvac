import argparse
import os
import pandas as pd

def main(input_path: str):
    df = pd.read_csv(
        input_path,
        sep=";",
        parse_dates=["Timestamp"]
    )

    # rename columns to snake_case for consistency
    df = df.rename(columns={
        "Temperature": "outside_temp",
        "outsideMinTemp": "outside_temp_min",
        "outsideMaxTemp": "outside_temp_max",
        "outsideHumidity": "outside_humidity"
    })

    # cast all numeric columns to float
    for col in ["outside_temp", "outside_temp_min", "outside_temp_max", "outside_humidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill forward then backward for outsideMinTemp
    df["outside_temp_min"] = df["outside_temp_min"].ffill().bfill()

    # sort and reset index
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # save to Processed/
    input_dir, filename = os.path.split(input_path)
    base_dir = os.path.dirname(input_dir)
    processed_dir = os.path.join(base_dir, "Processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess outdoor weather data")
    parser.add_argument("input_path", type=str, help="Path to raw CSV file")
    args = parser.parse_args()
    main(args.input_path)