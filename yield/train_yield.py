import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from yield_core import AdaptiveYieldLearner


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("data/yield_agriculture.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. FEATURES
# =========================================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

target = "yield"

X = df[features].values.astype(float)
y = df[target].values.astype(float)

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 4. INPUT NORMALIZATION
# =========================================================

X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

X_range = X_max - X_min

X_range[X_range == 0] = 1.0

X_train_scaled = (
    X_train - X_min
) / X_range

X_test_scaled = (
    X_test - X_min
) / X_range


# =========================================================
# 5. TARGET NORMALIZATION
# =========================================================

y_min = y_train.min()
y_max = y_train.max()

y_range = y_max - y_min

if y_range == 0:
    y_range = 1.0

y_train_scaled = (
    y_train - y_min
) / y_range


# =========================================================
# 6. CREATE ADAPTIVE YIELD LEARNER
# =========================================================

model = AdaptiveYieldLearner(
    input_size=7,
    learning_rate=0.001
)

print("\nAdaptive Yield Learner created.")
print("Input features:", model.input_size)


# =========================================================
# 7. TRAINING
# =========================================================

epochs = 30

print("\nStarting Adaptive Yield Learning...")
print("------------------------------------")

for epoch in range(epochs):

    total_error = 0.0

    indices = np.random.permutation(
        len(X_train_scaled)
    )

    for i in indices:

        x = X_train_scaled[i]

        target_value = y_train_scaled[i]

        prediction, error = model.update(
            x,
            target_value
        )

        total_error += error ** 2

    mse = total_error / len(X_train_scaled)

    rmse = np.sqrt(mse)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"- Training RMSE: {rmse:.6f}"
    )


print("\nTraining completed successfully!")


# =========================================================
# 8. TESTING
# =========================================================

predictions_scaled = []

for x in X_test_scaled:

    prediction = model.forward(x)

    predictions_scaled.append(
        prediction
    )

predictions_scaled = np.array(
    predictions_scaled
)


# =========================================================
# 9. ORIGINAL SCALE
# =========================================================

predictions = (
    predictions_scaled * y_range
) + y_min

predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# 10. EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# 11. RESULTS
# =========================================================

print("\n====================================")
print("ADAPTIVE YIELD PREDICTION RESULTS")
print("====================================")

print("Target Unit: tonnes/hectare (t/ha)")
print("Test samples:", len(X_test))

print(f"MAE  : {mae:.4f} t/ha")
print(f"RMSE : {rmse:.4f} t/ha")
print(f"R²   : {r2:.4f}")


# =========================================================
# 12. SAMPLE PREDICTIONS
# =========================================================

print("\nFirst 10 Predictions")
print("--------------------")

for i in range(min(10, len(y_test))):

    print(
        f"{i + 1:02d}. "
        f"Actual: {y_test[i]:.3f} t/ha "
        f"| Predicted: {predictions[i]:.3f} t/ha "
        f"| Error: "
        f"{abs(y_test[i] - predictions[i]):.3f}"
    )


# =========================================================
# 13. LEARNED PARAMETERS
# =========================================================

print("\nLearned Model Information")
print("-------------------------")

print(
    "Weights shape:",
    model.weights.shape
)

print(
    "Interaction matrix shape:",
    model.interaction_matrix.shape
)

print(
    "Feature memory shape:",
    model.feature_memory.shape
)

print(
    "Final bias:",
    model.bias
)

print(
    "Feature memory:",
    np.round(
        model.feature_memory,
        4
    )
)


# =========================================================
# 14. SAVE MODEL
# =========================================================

model_data = {

    "weights": model.weights,

    "interaction_matrix":
        model.interaction_matrix,

    "feature_memory":
        model.feature_memory,

    "bias":
        model.bias,

    "X_min":
        X_min,

    "X_range":
        X_range,

    "y_min":
        y_min,

    "y_range":
        y_range,

    "features":
        features,

    "target":
        "yield",

    "target_unit":
        "t/ha"
}


MODEL_PATH = "data/adaptive_yield_model.pkl"

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_data,
        f
    )


print("\n====================================")
print("ADAPTIVE YIELD MODEL SAVED")
print("====================================")

print("Model file:", MODEL_PATH)
print("Features:", features)
print("Target: yield")
print("Target unit: t/ha")
print("Model saved successfully!")