import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow imports such as: from backend.nsaf_service import ...
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Add yield model directory
YIELD_DIR = os.path.join(ROOT, "yield")

if YIELD_DIR not in sys.path:
    sys.path.insert(0, YIELD_DIR)

CROP_DATASET = os.path.join(ROOT, "data", "Crop_recommendation.csv")
YIELD_DATASET = os.path.join(ROOT, "data", "yield_agriculture.csv")

RESULT_DIR = os.path.join(ROOT, "comparison", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# PROPOSED MODEL IMPORTS
# =========================================================

from backend.nsaf_service import recommend_crop
from yield_core import AdaptiveYieldLearner


# =========================================================
# CROP: PROPOSED NSAF MODEL
# =========================================================

def evaluate_nsaf():
    print("\n" + "=" * 60)
    print("PROPOSED MODEL - NSAF CROP RECOMMENDATION")
    print("=" * 60)

    df = pd.read_csv(CROP_DATASET)

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    X = df[features]
    y = df["label"]

    # EXACT SAME HOLD-OUT CONFIGURATION AS compare_models.py
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    predictions = []

    for _, row in X_test.iterrows():

        input_data = {
            "N": float(row["N"]),
            "P": float(row["P"]),
            "K": float(row["K"]),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "ph": float(row["ph"]),
            "rainfall": float(row["rainfall"]),
        }

        result = recommend_crop(input_data)
        predictions.append(result["recommended_crop"])

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    result = pd.DataFrame([{
        "model": "NSAF (Proposed)",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }])

    print("\nNSAF Results")
    print("-" * 40)
    print(f"Test samples : {len(y_test)}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1:.4f}")

    return result


# =========================================================
# YIELD: PROPOSED ADAPTIVE YIELD MODEL
# =========================================================

def evaluate_adaptive_yield():
    print("\n" + "=" * 60)
    print("PROPOSED MODEL - ADAPTIVE YIELD PREDICTION")
    print("=" * 60)

    df = pd.read_csv(YIELD_DATASET)

    # IMPORTANT:
    # yield_agriculture.csv contains 7 input features.
    # This comparison therefore uses the same 7-feature
    # AdaptiveYieldLearner setup as comparison/compare_models.py.
    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    X = df[features].values.astype(float)
    y = df["yield"].values.astype(float)

    # SAME HOLD-OUT CONFIGURATION AS compare_models.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Normalize using TRAINING data only.
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X_train_scaled = (X_train - X_min) / X_range
    X_test_scaled = (X_test - X_min) / X_range

    y_min = y_train.min()
    y_max = y_train.max()
    y_range = y_max - y_min

    if y_range == 0:
        y_range = 1.0

    y_train_scaled = (y_train - y_min) / y_range

    # SAME MODEL + HYPERPARAMETERS AS compare_models.py
    model = AdaptiveYieldLearner(
        input_size=7,
        learning_rate=0.001,
    )

    epochs = 30

    for _ in range(epochs):
        for i in np.random.permutation(len(X_train_scaled)):
            model.update(
                X_train_scaled[i],
                y_train_scaled[i],
            )

    predictions_scaled = np.array([
        model.forward(x)
        for x in X_test_scaled
    ])

    predictions = (
        predictions_scaled * y_range
    ) + y_min

    predictions = np.maximum(
        predictions,
        0,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    result = pd.DataFrame([{
        "model": "Adaptive Yield (Proposed)",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }])

    print("\nAdaptive Yield Results")
    print("-" * 40)
    print(f"Test samples : {len(y_test)}")
    print(f"MAE          : {mae:.4f} t/ha")
    print(f"RMSE         : {rmse:.4f} t/ha")
    print(f"R2           : {r2:.4f}")

    return result


# =========================================================
# MERGE BASELINE + PROPOSED RESULTS
# =========================================================

def create_final_comparison(crop_proposed, yield_proposed):

    crop_baseline_path = os.path.join(
        RESULT_DIR,
        "crop_baseline_results.csv",
    )

    yield_baseline_path = os.path.join(
        RESULT_DIR,
        "yield_baseline_results.csv",
    )

    if not os.path.exists(crop_baseline_path):
        raise FileNotFoundError(
            "Run comparison/compare_models.py first."
        )

    if not os.path.exists(yield_baseline_path):
        raise FileNotFoundError(
            "Run comparison/compare_models.py first."
        )

    crop_baseline = pd.read_csv(crop_baseline_path)
    yield_baseline = pd.read_csv(yield_baseline_path)

    crop_final = pd.concat(
        [crop_baseline, crop_proposed],
        ignore_index=True,
    )

    yield_final = pd.concat(
        [yield_baseline, yield_proposed],
        ignore_index=True,
    )

    crop_output = os.path.join(
        RESULT_DIR,
        "crop_final_comparison.csv",
    )

    yield_output = os.path.join(
        RESULT_DIR,
        "yield_final_comparison.csv",
    )

    crop_final.to_csv(crop_output, index=False)
    yield_final.to_csv(yield_output, index=False)

    print("\n" + "=" * 60)
    print("FINAL CROP COMPARISON")
    print("=" * 60)
    print(crop_final.to_string(index=False))

    print("\n" + "=" * 60)
    print("FINAL YIELD COMPARISON")
    print("=" * 60)
    print(yield_final.to_string(index=False))

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print("Crop  :", crop_output)
    print("Yield :", yield_output)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    nsaf_results = evaluate_nsaf()

    adaptive_yield_results = evaluate_adaptive_yield()

    create_final_comparison(
        nsaf_results,
        adaptive_yield_results,
    )

    print("\nComparison completed successfully!")
