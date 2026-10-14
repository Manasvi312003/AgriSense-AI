# import pickle
# import numpy as np

# from yield_core import AdaptiveYieldLearner


# # =========================================================
# # 1. LOAD SAVED MODEL
# # =========================================================

# model_path = "data/adaptive_yield_model.pkl"

# with open(model_path, "rb") as f:
#     saved_model = pickle.load(f)

# print("Adaptive Yield model loaded successfully!")


# # =========================================================
# # 2. RESTORE OUR CUSTOM MODEL
# # =========================================================

# model = AdaptiveYieldLearner(
#     input_size=8,
#     learning_rate=0.001
# )

# model.weights = saved_model["weights"]
# model.interaction_matrix = saved_model["interaction_matrix"]
# model.feature_memory = saved_model["feature_memory"]
# model.bias = saved_model["bias"]

# X_min = saved_model["X_min"]
# X_range = saved_model["X_range"]

# y_min = saved_model["y_min"]
# y_range = saved_model["y_range"]

# features = saved_model["features"]


# # =========================================================
# # 3. USER INPUT
# # =========================================================

# print("\n====================================")
# print("     AGRISENSE AI - YIELD PREDICTION")
# print("====================================")

# N = float(input("Nitrogen (N): "))
# P = float(input("Phosphorus (P): "))
# K = float(input("Potassium (K): "))
# temperature = float(input("Temperature: "))
# humidity = float(input("Humidity: "))
# ph = float(input("pH: "))
# rainfall = float(input("Rainfall: "))
# area = float(input("Area: "))


# # =========================================================
# # 4. CREATE INPUT VECTOR
# # =========================================================

# input_data = np.array([
#     N,
#     P,
#     K,
#     temperature,
#     humidity,
#     ph,
#     rainfall,
#     area
# ], dtype=float)


# # =========================================================
# # 5. NORMALIZE USING TRAINING PARAMETERS
# # =========================================================

# input_scaled = (
#     input_data - X_min
# ) / X_range


# # =========================================================
# # 6. OUR ADAPTIVE MODEL PREDICTION
# # =========================================================

# prediction_scaled = model.forward(input_scaled)


# # =========================================================
# # 7. CONVERT BACK TO ORIGINAL YIELD SCALE
# # =========================================================

# prediction = (
#     prediction_scaled * y_range
# ) + y_min


# # Prevent negative prediction
# prediction = max(
#     float(prediction),
#     0.0
# )


# # =========================================================
# # 8. RESULT
# # =========================================================

# print("\n====================================")
# print("         YIELD PREDICTION")
# print("====================================")

# print(f"Predicted Yield: {prediction:.3f}")

# print("\nInput Features:")
# for name, value in zip(features, input_data):
#     print(f"{name}: {value}")

# print("\nPrediction completed successfully!")


#-----------------------------
#-----------------------------

import pickle
import numpy as np
from yield_core import AdaptiveYieldLearner

# 1. LOAD SAVED MODEL AND METADATA
model_path = "data/adaptive_yield_model.pkl"
with open(model_path, "rb") as f:
    saved_model = pickle.load(f)

base_features = saved_model["base_features"]
crop_columns = saved_model["crop_columns"]
total_input_size = len(base_features) + len(crop_columns)

# 2. RESTORE CUSTOM ARCHITECTURE
model = AdaptiveYieldLearner(input_size=total_input_size, learning_rate=0.001)
model.weights = saved_model["weights"]
model.interaction_matrix = saved_model["interaction_matrix"]
model.feature_memory = saved_model["feature_memory"]
model.bias = saved_model["bias"]

X_min = saved_model["X_min"]
X_range = saved_model["X_range"]
y_min = saved_model["y_min"]
y_range = saved_model["y_range"]

# 3. INTERACTIVE USER INTERFACE
print("\n====================================")
print("     AGRISENSE AI - YIELD PREDICTION")
print("====================================")

# Gather numeric environmental variables
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

# Extract cleaner crop names from string headers (e.g., 'crop_rice' -> 'rice')
clean_crop_names = [col.replace("crop_", "") for col in crop_columns]
print(f"\nSupported Crops: {', '.join(clean_crop_names)}")
selected_crop = input("Enter Crop Type: ").strip().lower()

if selected_crop not in clean_crop_names:
    print(f"\n[Warning]: '{selected_crop}' wasn't seen in training. Defaulting to baseline parameters.")

# 4. DYNAMICALLY RECONSTRUCT ONE-HOT ENCODED INPUT ARRAY
env_vector = [N, P, K, temperature, humidity, ph, rainfall]
crop_vector = [1.0 if col == f"crop_{selected_crop}" else 0.0 for col in crop_columns]

input_data = np.array(env_vector + crop_vector, dtype=float)

# 5. SCALE, EXECUTE FORWARD PASS AND DE-NORMALIZE
input_scaled = (input_data - X_min) / X_range
prediction_scaled = model.forward(input_scaled)
prediction = max(float((prediction_scaled * y_range) + y_min), 0.0)

# 6. OUTPUT METRICS
print("\n====================================")
print("         PREDICTION RESULTS")
print("====================================")
print(f"Target Crop    : {selected_crop.upper()}")
print(f"Predicted Yield: {prediction:.3f} tonnes per hectare (t/ha)")
print("====================================")
