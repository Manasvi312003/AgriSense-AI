import os
import pickle
import numpy as np


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "nsaf_crop_model.pkl"
)


# Load trained NSAF model
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)


prototypes = model_data["prototypes"]
classes = model_data["classes"]
W = model_data["W"]
context_strength = model_data["context_strength"]
fuzzy_boundaries = model_data["fuzzy_boundaries"]


def triangular_membership(x, left, center, right):
    if x == center:
        return 1.0

    if x <= left:
        return 0.0 if left != center else 1.0

    if x >= right:
        return 0.0 if right != center else 1.0

    if x < center:
        return (x - left) / (center - left)

    return (right - x) / (right - center)


def create_fuzzy_vector(input_data):
    vector = []

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    for feature in features:
        value = float(input_data[feature])
        boundaries = fuzzy_boundaries[feature]

        minimum = boundaries["minimum"]
        q25 = boundaries["q25"]
        median = boundaries["median"]
        q75 = boundaries["q75"]
        maximum = boundaries["maximum"]

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

    return np.array(vector, dtype=float)


def nsaf_forward(x):
    interaction = x @ W
    context = np.tanh(interaction)

    representation = x + (
        context_strength * context
    )

    return representation


def recommend_crop(input_data, top_n=5):

    # 1. Convert raw input → fuzzy 21-D vector
    fuzzy_vector = create_fuzzy_vector(input_data)

    # 2. NSAF transformation
    representation = nsaf_forward(fuzzy_vector)

    # 3. Calculate distance from every crop prototype
    distances = np.sum(
        (prototypes - representation) ** 2,
        axis=1
    )

    # 4. Sort crops by distance
    sorted_indices = np.argsort(distances)

    recommendations = []

    for index in sorted_indices[:top_n]:
        recommendations.append({
            "crop": classes[index],
            "distance": round(float(distances[index]), 4)
        })

    return {
        "recommended_crop": classes[sorted_indices[0]],
        "top_predictions": recommendations
    }