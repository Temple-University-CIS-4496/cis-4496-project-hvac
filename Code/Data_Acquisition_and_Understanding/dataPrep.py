#!/usr/bin/env python
#
# file: dataPrep.py
#
#------------------------------------------------------------------------------

# import system modules
#
import os
import glob
import gc
import argparse

# import external modules
#
import pandas as pd
import numpy as np

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def run(input_dir, outdoor_filename, output_dir):
    """
    method: run

    arguments:
      input_dir: directory containing raw indoor csv files
      outdoor_filename: filename or full path for the weather data
      output_dir: directory where processed results will be stored

    return:
      boolean status of the operation

    description:
      Main execution logic for the data preparation pipeline.
    """

    # validate input directory
    #
    if not os.path.isdir(input_dir):
        print("Error: Input directory not found.")
        return False

    # determine weather path: absolute/relative or filename inside input_dir
    #
    if os.sep in outdoor_filename or outdoor_filename.startswith('.'):
        outdoor_path = outdoor_filename
    else:
        outdoor_path = os.path.join(input_dir, outdoor_filename)

    if not os.path.exists(outdoor_path):
        print("Error: Weather file missing at: %s" % outdoor_path)
        return False

    # prepare output directory
    #
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load and clean outdoor weather data
    #
    outdoor_df = pd.read_csv(
        outdoor_path, sep=';', parse_dates=["Timestamp"], low_memory=False
    )
    outdoor_df = outdoor_df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
    outdoor_df = outdoor_df.rename(columns={"Temperature": "Outdoor_Temperature"})

    # cast weather columns to numeric types
    #
    env_cols = [
        "Outdoor_Temperature", "outsideMinTemp",
        "outsideMaxTemp", "outsideHumidity"
    ]
    cols_to_cast = outdoor_df.columns.intersection(env_cols)
    outdoor_df[cols_to_cast] = outdoor_df[cols_to_cast].apply(
        pd.to_numeric, errors='coerce'
    )

    # gather list of indoor data files (excluding weather file)
    #
    indoor_csvs = [
        f for f in glob.glob(os.path.join(input_dir, "*.csv"))
        if os.path.abspath(f) != os.path.abspath(outdoor_path)
    ]

    # iterate through indoor files
    #
    for csv_path in indoor_csvs:
        filename = os.path.basename(csv_path)
        print("Processing: %s" % filename)

        # load raw data
        #
        df = pd.read_csv(csv_path, sep=";", parse_dates=["Timestamp"], low_memory=False)

        # map to standard internal column names as established in original script
        #
        df = df.rename(columns={
            "output_state": "OutputState", "fan_state": "FanState",
            "running_mode": "RunningMode", "temp": "Temperature", "set_point": "Setpoint"
        })

        # strip rows before the first valid indoor temperature reading
        #
        first_idx = df["Temperature"].first_valid_index()
        df = df.loc[first_idx:].copy()


        # Temperature to float
        #
        df["Temperature"] = df["Temperature"].astype(float)

        # Setpoint ffill and default
        #
        df["Setpoint"] = df["Setpoint"].ffill().fillna(0)

        # FanState mapping (on:1, off:0)
        #
        df["FanState"] = pd.to_numeric(df["FanState"], errors="coerce").ffill().fillna(0).astype(int)

        # OutputState mapping (idle:0, active:1)
        #
        df["OutputState"] = df["OutputState"].ffill().fillna("idle")
        df["OutputState"] = df["OutputState"].astype(str).str.lower().ne("idle").astype(int)

        # RunningMode logic
        #
        df["RunningMode"] = pd.to_numeric(df["RunningMode"], errors="coerce").ffill().fillna(0).astype(int)
        
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df["timestamp_diff"] = df["Timestamp"].diff()

        # check for intervals where fan was active
        #
        df["runtime_change"] = df["timestamp_diff"].where(
            (df["FanState"] == 1) & (df["FanState"].shift(1) == 1)
        )
        df["CumulativeRuntime"] = df["runtime_change"].dt.total_seconds().cumsum()

        # cost: (hours * rate) * user multiplier of 3
        #
        df["CumulativeCost"] = (
            df["CumulativeRuntime"] / 3600.0 * 0.17 * 3.0
        )

        # extract date components
        #
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Day"] = df["Timestamp"].dt.day

        # map numeric running mode back to string labels for dummy creation
        #
        mode_labels = {0: "off", 1: "heat", 2: "cool"}
        df["RunningMode_Str"] = df["RunningMode"].map(mode_labels)
        df = pd.get_dummies(df, columns=["RunningMode_Str"], prefix="RunningMode", dtype=int)

        # isolate indoor timestamps for final alignment
        #
        indoor_timestamps = df[['Timestamp']].copy()
        
        # merge onto timeline
        #
        merged = pd.merge(df, outdoor_df, on="Timestamp", how="outer")
        merged = merged.sort_values("Timestamp").set_index("Timestamp")

        # interpolate the outdoor temperature exactly at indoor logs
        # bfill handles timestamps preceding first weather entry
        #
        merged["Outdoor_Temperature"] = (
            merged["Outdoor_Temperature"].interpolate(method="time").bfill()
        )

        # fill discrete environmental data (Min/Max/Humidity)
        #
        step_cols = ["outsideMinTemp", "outsideMaxTemp", "outsideHumidity"]
        cols_to_fill = merged.columns.intersection(step_cols)
        merged[cols_to_fill] = merged[cols_to_fill].ffill().bfill()

        # filter back to original indoor timestamps via left join
        #
        df = pd.merge(indoor_timestamps, merged.reset_index(), on="Timestamp", how="left")

        # drop intermediate and original raw columns as per script logic
        #
        df = df.drop(
            columns=[
                "Mode", "Occupied", "RentalStatus", "output_state", "running_mode",
                "timestamp_diff", "runtime_change", "fan_state", "temp", "set_point"
            ],
            errors="ignore"
        )

        # save results
        #
        output_path = os.path.join(output_dir, "processed_%s" % filename)
        df.to_csv(output_path, index=False)

        # release memory resources
        #
        del df, merged, indoor_timestamps
        gc.collect()

    print("Success. All files saved to: %s" % output_dir)

    # exit gracefully
    #
    return True

#
# end of function

#------------------------------------------------------------------------------
#
# main program starts here
#
#------------------------------------------------------------------------------


# begin gracefully
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HVAC Data Preparation Tool")

    parser.add_argument(
        "input_dir",
        type=str,
        help="input directory with raw csv files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory to save processed csv files"
    )

    parser.add_argument(
        "--outdoor_filename",
        type=str,
        default="outdoorweather.csv",
        help="filename or path of the weather data file"
    )

    args = parser.parse_args()
    run(args.input_dir, args.outdoor_filename, args.output_dir)

#
# end of file
