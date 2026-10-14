# import os
# import pickle
# import numpy as np


# # =========================================================
# # MODEL PATH
# # =========================================================

# MODEL_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     "data",
#     "adaptive_yield_model.pkl"
# )


# # =========================================================
# # LOAD TRAINED MODEL
# # =========================================================

# with open(MODEL_PATH, "rb") as f:
#     model_data = pickle.load(f)


# weights = model_data["weights"]
# interaction_matrix = model_data["interaction_matrix"]
# feature_memory = model_data["feature_memory"]
# bias = model_data["bias"]

# X_min = model_data["X_min"]
# X_range = model_data["X_range"]

# y_min = model_data["y_min"]
# y_range = model_data["y_range"]

# features = model_data["features"]

# target_unit = model_data.get(
#     "target_unit",
#     "t/ha"
# )


# print("Adaptive Yield Service loaded successfully!")
# print("Features:", features)
# print("Target unit:", target_unit)


# # =========================================================
# # ADAPTIVE YIELD FORWARD PASS
# # =========================================================

# def yield_forward(x):

#     # Feature interaction
#     interaction = x @ interaction_matrix

#     # Bounded nonlinear context
#     context = np.tanh(interaction)

#     # Adaptive feature-memory gating
#     memory_context = (
#         feature_memory * context
#     )

#     # Weighted adaptive representation
#     output = np.sum(
#         weights * (
#             x + memory_context
#         )
#     )

#     # Learned bias
#     output += bias

#     return output


# # =========================================================
# # PREDICT YIELD
# # =========================================================

# def predict_yield(input_data):

#     # -----------------------------------------------------
#     # 1. Seven model features
#     # -----------------------------------------------------

#     values = np.array([
#         float(input_data["N"]),
#         float(input_data["P"]),
#         float(input_data["K"]),
#         float(input_data["temperature"]),
#         float(input_data["humidity"]),
#         float(input_data["ph"]),
#         float(input_data["rainfall"])
#     ], dtype=float)


#     # -----------------------------------------------------
#     # 2. Min-Max normalization
#     # -----------------------------------------------------

#     scaled_input = (
#         values - X_min
#     ) / X_range


#     # -----------------------------------------------------
#     # 3. Adaptive forward pass
#     # -----------------------------------------------------

#     prediction_scaled = yield_forward(
#         scaled_input
#     )


#     # -----------------------------------------------------
#     # 4. Convert to original yield scale
#     # -----------------------------------------------------

#     predicted_yield = (
#         prediction_scaled * y_range
#     ) + y_min


#     # -----------------------------------------------------
#     # 5. Prevent negative yield
#     # -----------------------------------------------------

#     predicted_yield = max(
#         float(predicted_yield),
#         0.0
#     )


#     # -----------------------------------------------------
#     # 6. Area-based total production
#     # -----------------------------------------------------

#     area = float(
#         input_data["area"]
#     )

#     total_production = (
#         predicted_yield * area
#     )


#     # -----------------------------------------------------
#     # 7. Return complete result
#     # -----------------------------------------------------

#     return {

#         "predicted_yield": round(
#             predicted_yield,
#             3
#         ),

#         "yield_unit": "t/ha",

#         "area": round(
#             area,
#             3
#         ),

#         "area_unit": "ha",

#         "estimated_production": round(
#             total_production,
#             3
#         ),

#         "production_unit": "tonnes"
#     }

#-----------------
#-----------------

import pickle
import numpy as np
import os
from yield_model.yield_core import AdaptiveYieldLearner

# Model load karne ka logic setup
MODEL_PATH = "data/adaptive_yield_model.pkl"
yield_model = None
saved_metadata = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        saved_metadata = pickle.load(f)
    
    # Custom architecture re-initialize dimensions ke sath
    base_features = saved_metadata["base_features"]
    crop_columns = saved_metadata["crop_columns"]
    total_inputs = len(base_features) + len(crop_columns)
    
    yield_model = AdaptiveYieldLearner(input_size=total_inputs, learning_rate=0.001)
    yield_model.weights = saved_metadata["weights"]
    yield_model.interaction_matrix = saved_metadata["interaction_matrix"]
    yield_model.feature_memory = saved_metadata["feature_memory"]
    yield_model.bias = saved_metadata["bias"]

def predict_yield(input_dict: dict) -> dict:
    """
    Input schema holds: N, P, K, temperature, humidity, ph, rainfall, area, AND crop (optional/required)
    """
    if yield_model is None or saved_metadata is None:
        return {"predicted_yield": 0.0, "status": "Model not loaded"}
    
    try:
        X_min = saved_metadata["X_min"]
        X_range = saved_metadata["X_range"]
        y_min = saved_metadata["y_min"]
        y_range = saved_metadata["y_range"]
        crop_columns = saved_metadata["crop_columns"]
        
        # Agar dictionary me direct crop nahi hai (jaise pure yield prediction tab se call hua)
        # To default crop 'rice' le lo ya fallback options run karo
        selected_crop = input_dict.get("crop", "rice").strip().lower()
        area_multiplier = float(input_dict.get("area", 1.0))
        
        # Environmental and One-Hot crop vector compilation
        env_vector = [
            float(input_dict["N"]), float(input_dict["P"]), float(input_dict["K"]),
            float(input_dict["temperature"]), float(input_dict["humidity"]),
            float(input_dict["ph"]), float(input_dict["rainfall"])
        ]
        crop_vector = [1.0 if col == f"crop_{selected_crop}" else 0.0 for col in crop_columns]
        
        input_data = np.array(env_vector + crop_vector, dtype=float)
        
        # Scaling aur inference engine run
        input_scaled = (input_data - X_min) / X_range
        prediction_scaled = yield_model.forward(input_scaled)
        
        # Convert back to actual tonnes/hectare scale
        yield_per_hectare = max(float((prediction_scaled * y_range) + y_min), 0.0)
        
        # Total yield = Yield per Hectare * Total Farm Area
        total_predicted_yield = yield_per_hectare * area_multiplier
        
        return {
          "predicted_yield": round(yield_per_hectare, 3),          # Ye per hectare yield hai
            "estimated_production": round(total_predicted_yield, 3), # Ye total farm ka yield hai
            "area": area_multiplier,
            "yield_unit": "t/ha",
            "area_unit": "ha",
            "production_unit": "tonnes",
            "status": "success"
        }
    except Exception as e:
        return {"predicted_yield": 0.0, "status": f"Inference Error: {str(e)}"}
