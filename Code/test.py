import pandas as pd

df = pd.read_csv("./Sample_Data/Raw/house_1_data.csv", sep=";")

print(df.head())