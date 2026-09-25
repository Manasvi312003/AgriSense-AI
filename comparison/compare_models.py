import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PATHS
# =========================================================

ROOT = os.path.dirname(os.path.dirname(__file__))

CROP_DATASET = os.path.join(
    ROOT,
    "data",
    "Crop_recommendation.csv"
)

YIELD_DATASET = os.path.join(
    ROOT,
    "data",
    "yield_agriculture.csv"
)

RESULT_DIR = os.path.join(
    ROOT,
    "comparison",
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# CROP RECOMMENDATION
# =========================================================

def compare_crop_models():

    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION MODEL COMPARISON")
    print("=" * 60)

    df = pd.read_csv(CROP_DATASET)

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    X = df[features]
    y = df["label"]

    # -----------------------------------------------------
    # SAME SPLIT FOR ALL MODELS
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5
            ))
        ])
    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        results.append({
            "model": name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f}, "
            f"F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "crop_baseline_results.csv"
        ),
        index=False
    )

    return results_df


# =========================================================
# YIELD PREDICTION
# =========================================================

def compare_yield_models():

    print("\n" + "=" * 60)
    print("YIELD PREDICTION MODEL COMPARISON")
    print("=" * 60)

    df = pd.read_csv(YIELD_DATASET)

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    X = df[features]
    y = df["yield"]

    # -----------------------------------------------------
    # SAME SPLIT FOR ALL MODELS
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    models = {

        "Random Forest Regressor":
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ),

        "Linear Regression":
            LinearRegression()
    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

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

        results.append({
            "model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

        print(
            f"{name}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "yield_baseline_results.csv"
        ),
        index=False
    )

    return results_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    crop_results = compare_crop_models()

    yield_results = compare_yield_models()

    print("\n")
    print("=" * 60)
    print("CROP BASELINE RESULTS")
    print("=" * 60)

    print(crop_results.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("YIELD BASELINE RESULTS")
    print("=" * 60)

    print(yield_results.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("RESULT FILES SAVED")
    print("=" * 60)

    print(
        "Crop:",
        "comparison/results/crop_baseline_results.csv"
    )

    print(
        "Yield:",
        "comparison/results/yield_baseline_results.csv"
    )