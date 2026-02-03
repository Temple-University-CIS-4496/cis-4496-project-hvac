import pandas as pd

df = pd.read_csv("./Sample_Data/Raw/touchcombo_00915395.csv", sep=";")

df = df.drop('Mode', axis=1)

print(df.head())