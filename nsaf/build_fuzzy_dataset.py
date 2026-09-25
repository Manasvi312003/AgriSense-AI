import pandas as pd


def triangular_membership(x, left, center, right):
    if left > center or center > right:
        raise ValueError("Invalid fuzzy boundaries")

    if x <= left or x >= right:
        return 0.0

    if x == center:
        return 1.0

    if x < center:
        return (x - left) / (center - left)

    return (right - x) / (right - center)


# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


fuzzy_rows = []

for _, row in df.iterrows():

    fuzzy_vector = []

    for feature in features:

        value = row[feature]

        minimum = df[feature].min()
        q25 = df[feature].quantile(0.25)
        median = df[feature].median()
        q75 = df[feature].quantile(0.75)
        maximum = df[feature].max()

        low = triangular_membership(
            value, minimum, minimum, median
        )

        medium = triangular_membership(
            value, q25, median, q75
        )

        high = triangular_membership(
            value, median, maximum, maximum
        )

        fuzzy_vector.extend([low, medium, high])

    fuzzy_rows.append(fuzzy_vector)


fuzzy_df = pd.DataFrame(fuzzy_rows)

fuzzy_columns = []

for feature in features:
    fuzzy_columns.extend([
        f"{feature}_low",
        f"{feature}_medium",
        f"{feature}_high"
    ])

fuzzy_df.columns = fuzzy_columns

print("Original dataset shape:", df.shape)
print("Fuzzy dataset shape:", fuzzy_df.shape)
print("\nFirst fuzzy row:")
print(fuzzy_df.iloc[0].values)

# Add crop labels
fuzzy_df["label"] = df["label"].values

print("\nFinal fuzzy dataset shape:", fuzzy_df.shape)

print("\nFinal columns:")
print(fuzzy_df.columns.tolist())

output_path = "data/fuzzy_agriculture.csv"

fuzzy_df.to_csv(output_path, index=False)

print("\nSaved fuzzy dataset to:", output_path)