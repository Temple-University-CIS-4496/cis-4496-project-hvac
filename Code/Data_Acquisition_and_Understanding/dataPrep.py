import pandas as pd

df = pd.read_csv("./Sample_Data/Raw/touchcombo_00915382.csv", sep=";")

# drop "Mode"
df = df.drop('Mode', axis=1)

# print(df[215:230])

# forward-fill running_mode to propagate the last known state
df['running_mode'] = df['running_mode'].fillna(method='ffill')

# fills the first row NaN
df['running_mode'] = df['running_mode'].fillna('off')

print(df[215:230])