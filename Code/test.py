import pandas as pd

df = pd.read_csv("./Sample_Data/Raw/touchcombo_00915395.csv", sep=";")

# print(df['running_mode'].value_counts())
# print(df['Temperature'].value_counts())
# print(df['Mode'].value_counts())
# print(df['Occupied'].value_counts())
# print(df['FanState'].value_counts())
# print(df['output_state'].value_counts())
print(df['RentalStatus'].value_counts())

# print(df.head())
# Temperature = numerical float(convert to fahrenheit to save on data)
# Setpoint = numerical float
# Mode = (cool, heat) has the mode, but is probably useless?
# Occupied = (true, false) convert it to 0, 1
# FanState = (off, on) convert it to 0, 1
# output_state = (idle, cool_low, pre_cool, post_cool, cool_med, post_heat, heat_low, pre_heat, heat_med, cool_high, heat_high, refresh_heat, refresh_cool)
#               potentiall change it to smaller words (e.g. idle=i, cool_low=cl, heat_low=hl, etc.)
# running mode = (off, heat, cool, NaN). Once the running_mode is at a setting (off, cool, or heat), all the NaN values after are set to 
#               either of those values until the next reoccurance
# RentalStatus = (Sold, Unsold) indicates whether the house is sold or not (can be dropped)