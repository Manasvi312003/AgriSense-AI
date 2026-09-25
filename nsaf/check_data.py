import pandas as pd

df = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique crop labels:")
print(df["label"].nunique())

print("\nCrop distribution:")
print(df["label"].value_counts())