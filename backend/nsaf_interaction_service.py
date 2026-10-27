
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "nsaf_interaction_prototype_model.pkl"


with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

W = model_data["W"]
context_strength = model_data["context_strength"]
head = model_data["head"]
bias = model_data["bias"]
prototypes = model_data["prototypes"]

classes = model_data["classes"]
features = model_data["features"]
fuzzy_boundaries = model_data["fuzzy_boundaries"]


def triangular_membership(x, left, center, right):
    if left > center or center > right:
        raise ValueError("Invalid fuzzy boundaries")

    if left == center == right:
        return 1.0 if x == center else 0.0

    if x == center:
        return 1.0

    if left == center:
        if x < center:
            return 1.0
        if x >= right:
            return 0.0
        return (right - x) / (right - center)

    if center == right:
        if x > center:
            return 1.0
        if x <= left:
            return 0.0
        return (x - left) / (center - left)

    if x <= left or x >= right:
        return 0.0

    if x < center:
        return (x - left) / (center - left)

    return (right - x) / (right - center)


def create_fuzzy_vector(input_data):
    vector = []

    for feature in features:
        value = float(input_data[feature])
        bounds = fuzzy_boundaries[feature]

        minimum = bounds["minimum"]
        q25 = bounds["q25"]
        median = bounds["median"]
        q75 = bounds["q75"]
        maximum = bounds["maximum"]

        low = triangular_membership(
            value, minimum, minimum, median
        )
        medium = triangular_membership(
            value, q25, median, q75
        )
        high = triangular_membership(
            value, median, maximum, maximum
        )

        vector.extend([low, medium, high])

    return np.asarray(vector, dtype=float)


def predict_crop(input_data, top_n=5):
    missing = set(features) - set(input_data)

    if missing:
        raise ValueError(
            f"Missing input features: {sorted(missing)}"
        )

    x = create_fuzzy_vector(input_data)

    # Improved NSAF representation
    context = np.tanh(x @ W)
    representation = x + context_strength * context

    # Squared Euclidean distance to every crop prototype
    distances = np.sum(
        (prototypes - representation) ** 2,
        axis=1,
    )

    # Smaller distance means a closer prototype
    top_indices = np.argsort(distances)[:top_n]

    return {
        "recommended_crop": classes[int(top_indices[0])],
        "top_predictions": [
            {
                "crop": classes[int(i)],
                "distance": round(float(distances[i]), 4),
            }
            for i in top_indices
        ],
    }
