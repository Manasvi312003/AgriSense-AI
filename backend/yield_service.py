import os
import pickle
import numpy as np


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "adaptive_yield_model.pkl"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)


weights = model_data["weights"]
interaction_matrix = model_data["interaction_matrix"]
feature_memory = model_data["feature_memory"]
bias = model_data["bias"]

X_min = model_data["X_min"]
X_range = model_data["X_range"]

y_min = model_data["y_min"]
y_range = model_data["y_range"]

features = model_data["features"]

target_unit = model_data.get(
    "target_unit",
    "t/ha"
)


print("Adaptive Yield Service loaded successfully!")
print("Features:", features)
print("Target unit:", target_unit)


# =========================================================
# ADAPTIVE YIELD FORWARD PASS
# =========================================================

def yield_forward(x):

    # Feature interaction
    interaction = x @ interaction_matrix

    # Bounded nonlinear context
    context = np.tanh(interaction)

    # Adaptive feature-memory gating
    memory_context = (
        feature_memory * context
    )

    # Weighted adaptive representation
    output = np.sum(
        weights * (
            x + memory_context
        )
    )

    # Learned bias
    output += bias

    return output


# =========================================================
# PREDICT YIELD
# =========================================================

def predict_yield(input_data):

    # -----------------------------------------------------
    # 1. Seven model features
    # -----------------------------------------------------

    values = np.array([
        float(input_data["N"]),
        float(input_data["P"]),
        float(input_data["K"]),
        float(input_data["temperature"]),
        float(input_data["humidity"]),
        float(input_data["ph"]),
        float(input_data["rainfall"])
    ], dtype=float)


    # -----------------------------------------------------
    # 2. Min-Max normalization
    # -----------------------------------------------------

    scaled_input = (
        values - X_min
    ) / X_range


    # -----------------------------------------------------
    # 3. Adaptive forward pass
    # -----------------------------------------------------

    prediction_scaled = yield_forward(
        scaled_input
    )


    # -----------------------------------------------------
    # 4. Convert to original yield scale
    # -----------------------------------------------------

    predicted_yield = (
        prediction_scaled * y_range
    ) + y_min


    # -----------------------------------------------------
    # 5. Prevent negative yield
    # -----------------------------------------------------

    predicted_yield = max(
        float(predicted_yield),
        0.0
    )


    # -----------------------------------------------------
    # 6. Area-based total production
    # -----------------------------------------------------

    area = float(
        input_data["area"]
    )

    total_production = (
        predicted_yield * area
    )


    # -----------------------------------------------------
    # 7. Return complete result
    # -----------------------------------------------------

    return {

        "predicted_yield": round(
            predicted_yield,
            3
        ),

        "yield_unit": "t/ha",

        "area": round(
            area,
            3
        ),

        "area_unit": "ha",

        "estimated_production": round(
            total_production,
            3
        ),

        "production_unit": "tonnes"
    }