import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from recommend_crop import recommend_crop, features


# ============================================================
# Load original dataset
# ============================================================

df = pd.read_csv("data/Crop_recommendation.csv")


# ============================================================
# Same train-test split used during NSAF training
# ============================================================

X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n===================================")
print(" NSAF RECOMMENDATION VALIDATION")
print("===================================")

print("Test samples:", len(X_test))


# ============================================================
# Test every unseen sample
# ============================================================

predictions = []

for index, row in X_test.iterrows():

    input_data = {
        feature: float(row[feature])
        for feature in features
    }

    predicted_crop, distances = recommend_crop(input_data)

    predictions.append(predicted_crop)


# ============================================================
# Accuracy
# ============================================================

correct = sum(
    actual == predicted
    for actual, predicted in zip(y_test, predictions)
)

total = len(y_test)

accuracy = correct / total


print("\nCorrect predictions:", correct)
print("Incorrect predictions:", total - correct)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Accuracy (%): {accuracy * 100:.2f}%")


# ============================================================
# Classification Report
# ============================================================

print("\nNSAF VALIDATION CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix Shape:", cm.shape)
print("Total predictions:", cm.sum())
print("Correct predictions:", np.trace(cm))
print(
    "Incorrect predictions:",
    cm.sum() - np.trace(cm)
)


# ============================================================
# Show some actual vs predicted results
# ============================================================

print("\nFirst 20 Predictions")
print("====================")

for i, (actual, predicted) in enumerate(
    zip(y_test.values[:20], predictions[:20]),
    start=1
):

    status = "✓" if actual == predicted else "✗"

    print(
        f"{i:02d}. Actual: {actual:<12} "
        f"Predicted: {predicted:<12} {status}"
    )


print("\n===================================")
print(" Validation completed.")
print("===================================")