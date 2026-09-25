import numpy as np
import pandas as pd
import pickle

from nsaf_core import NSAFCore


# ============================================================
# Load trained NSAF model
# ============================================================

with open("data/nsaf_crop_model.pkl", "rb") as f:
    saved_model = pickle.load(f)

print("NSAF model loaded successfully!")


# ============================================================
# Load saved components
# ============================================================

prototypes = saved_model["prototypes"]
classes = saved_model["classes"]
fuzzy_boundaries = saved_model["fuzzy_boundaries"]
features = saved_model["features"]

print("Number of classes:", len(classes))
print("Number of features:", len(features))


# ============================================================
# Create NSAF model and load learned parameters
# ============================================================

model = NSAFCore(
    input_size=21,
    learning_rate=0.01
)

model.W = saved_model["W"]
model.context_strength = saved_model["context_strength"]


# ============================================================
# Triangular Membership Function
# ============================================================

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


# ============================================================
# Create fuzzy vector
# ============================================================

def create_fuzzy_vector(input_data):

    fuzzy_vector = []

    for feature in features:

        value = input_data[feature]

        boundaries = fuzzy_boundaries[feature]

        minimum = boundaries["minimum"]
        q25 = boundaries["q25"]
        median = boundaries["median"]
        q75 = boundaries["q75"]
        maximum = boundaries["maximum"]

        low = triangular_membership(
            value,
            minimum,
            minimum,
            median
        )

        medium = triangular_membership(
            value,
            q25,
            median,
            q75
        )

        high = triangular_membership(
            value,
            median,
            maximum,
            maximum
        )

        fuzzy_vector.extend([
            low,
            medium,
            high
        ])

    return np.array(fuzzy_vector, dtype=float)


# ============================================================
# Crop Recommendation
# ============================================================

def recommend_crop(input_data):

    # Raw input → Fuzzy vector
    fuzzy_vector = create_fuzzy_vector(input_data)

    # Fuzzy vector → NSAF representation
    representation = model.forward(fuzzy_vector)

    # Distance from each learned prototype
    distances = np.sum(
        (prototypes - representation) ** 2,
        axis=1
    )

    # Nearest prototype
    predicted_class = np.argmin(distances)

    predicted_crop = classes[predicted_class]

    return predicted_crop, distances


# ============================================================
# Test Input
# ============================================================

sample_input = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.94
}


# ============================================================
# Run Recommendation
# ============================================================

crop, distances = recommend_crop(sample_input)


print("\n===================================")
print("       NSAF CROP RECOMMENDATION")
print("===================================")

print("\nInput:")

for key, value in sample_input.items():
    print(f"{key}: {value}")

print("\nRecommended Crop:", crop)

print("\nTop 5 predictions:")

top_indices = np.argsort(distances)[:5]

for rank, index in enumerate(top_indices, start=1):

    print(
        f"{rank}. {classes[index]} "
        f"(distance: {distances[index]:.4f})"
    )

print("\n===================================")