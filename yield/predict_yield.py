import pickle
import numpy as np

from yield_core import AdaptiveYieldLearner


# =========================================================
# 1. LOAD SAVED MODEL
# =========================================================

model_path = "data/adaptive_yield_model.pkl"

with open(model_path, "rb") as f:
    saved_model = pickle.load(f)

print("Adaptive Yield model loaded successfully!")


# =========================================================
# 2. RESTORE OUR CUSTOM MODEL
# =========================================================

model = AdaptiveYieldLearner(
    input_size=8,
    learning_rate=0.001
)

model.weights = saved_model["weights"]
model.interaction_matrix = saved_model["interaction_matrix"]
model.feature_memory = saved_model["feature_memory"]
model.bias = saved_model["bias"]

X_min = saved_model["X_min"]
X_range = saved_model["X_range"]

y_min = saved_model["y_min"]
y_range = saved_model["y_range"]

features = saved_model["features"]


# =========================================================
# 3. USER INPUT
# =========================================================

print("\n====================================")
print("     AGRISENSE AI - YIELD PREDICTION")
print("====================================")

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))
area = float(input("Area: "))


# =========================================================
# 4. CREATE INPUT VECTOR
# =========================================================

input_data = np.array([
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    area
], dtype=float)


# =========================================================
# 5. NORMALIZE USING TRAINING PARAMETERS
# =========================================================

input_scaled = (
    input_data - X_min
) / X_range


# =========================================================
# 6. OUR ADAPTIVE MODEL PREDICTION
# =========================================================

prediction_scaled = model.forward(input_scaled)


# =========================================================
# 7. CONVERT BACK TO ORIGINAL YIELD SCALE
# =========================================================

prediction = (
    prediction_scaled * y_range
) + y_min


# Prevent negative prediction
prediction = max(
    float(prediction),
    0.0
)


# =========================================================
# 8. RESULT
# =========================================================

print("\n====================================")
print("         YIELD PREDICTION")
print("====================================")

print(f"Predicted Yield: {prediction:.3f}")

print("\nInput Features:")
for name, value in zip(features, input_data):
    print(f"{name}: {value}")

print("\nPrediction completed successfully!")