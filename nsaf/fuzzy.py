import numpy as np


def triangular_membership(x, left, center, right):
    if left > center or center > right:
        raise ValueError("Invalid fuzzy boundaries")

    if x == center:
        return 1.0

    if x <= left:
        return 0.0 if left != center else 1.0

    if x >= right:
        return 0.0 if right != center else 1.0

    if x < center:
        return (x - left) / (center - left)

    return (right - x) / (right - center)

print("Test 1:", triangular_membership(25, 15, 25, 35))
print("Test 2:", triangular_membership(20, 15, 25, 35))
print("Test 3:", triangular_membership(30, 15, 25, 35))
print("Test 4:", triangular_membership(10, 15, 25, 35))
print("Test 5:", triangular_membership(40, 15, 25, 35))

import pandas as pd

df = pd.read_csv("data/Crop_recommendation.csv")

print("\nN feature fuzzy test:")

n_value = df["N"].iloc[0]

low = triangular_membership(n_value, 0, 0, 37)
medium = triangular_membership(n_value, 21, 37, 84.25)
high = triangular_membership(n_value, 37, 84.25, 140)

print("N value:", n_value)
print("Low:", low)
print("Medium:", medium)
print("High:", high)

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

print("\nFuzzy representation of first row:")

for feature in features:
    value = df[feature].iloc[0]

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

    print(
        feature,
        "=", round(value, 2),
        "→ Low:", round(low, 3),
        "Medium:", round(medium, 3),
        "High:", round(high, 3)
    )

    fuzzy_vector = []

for feature in features:
    value = df[feature].iloc[0]

    minimum = df[feature].min()
    q25 = df[feature].quantile(0.25)
    median = df[feature].median()
    q75 = df[feature].quantile(0.75)
    maximum = df[feature].max()

    low = triangular_membership(value, minimum, minimum, median)
    medium = triangular_membership(value, q25, median, q75)
    high = triangular_membership(value, median, maximum, maximum)

    fuzzy_vector.extend([low, medium, high])

print("\nFuzzy Evidence Vector:")
print(fuzzy_vector)

print("\nVector length:", len(fuzzy_vector))