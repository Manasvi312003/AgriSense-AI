import pandas as pd
from sklearn.model_selection import train_test_split


# Load fuzzy dataset
df = pd.read_csv("data/fuzzy_agriculture.csv")

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

print("\nNumber of classes:", y.nunique())
print("Classes:")
print(sorted(y.unique()))

from nsaf_core import NSAFCore
import numpy as np


# Create NSAF model
model = NSAFCore(
    input_size=21,
    learning_rate=0.01
)

# Convert training data to numpy
X_train_np = X_train.values.astype(float)

# Test NSAF on first training sample
sample = X_train_np[0]

representation = model.forward(sample)

print("\nNSAF input shape:", sample.shape)
print("NSAF representation shape:", representation.shape)
print("First 5 NSAF values:", representation[:5])

# ============================================================
# NSAF Adaptive Prototype Learning
# ============================================================

import numpy as np


# Convert crop names into numeric class IDs
classes = sorted(y_train.unique())

class_to_id = {
    crop: i for i, crop in enumerate(classes)
}

y_train_encoded = np.array([
    class_to_id[crop]
    for crop in y_train
])


# Create one learnable prototype for every crop
num_classes = len(classes)

prototypes = np.zeros(
    (num_classes, 21),
    dtype=float
)


# Prototype learning rate
prototype_lr = 0.05


def predict_from_prototypes(representation, prototypes):
    """
    Predict crop using similarity with learned prototypes.
    """

    distances = np.sum(
        (prototypes - representation) ** 2,
        axis=1
    )

    predicted_class = np.argmin(distances)

    return predicted_class, distances


# ============================================================
# Training
# ============================================================

epochs = 10

for epoch in range(epochs):

    correct = 0

    for i in range(len(X_train_np)):

        x = X_train_np[i]

        # NSAF forward pass
        representation = model.forward(x)

        # Current prediction
        predicted_class, distances = predict_from_prototypes(
            representation,
            prototypes
        )

        actual_class = y_train_encoded[i]

        if predicted_class == actual_class:
            correct += 1

        # ----------------------------------------------------
        # Adaptive prototype update
        # ----------------------------------------------------

        error = 1.0 if predicted_class != actual_class else 0.0

        # Move correct class prototype toward representation
        prototypes[actual_class] += (
            prototype_lr *
            (representation - prototypes[actual_class])
        )

        # If wrong, push predicted prototype away
        if predicted_class != actual_class:

            prototypes[predicted_class] -= (
                prototype_lr *
                error *
                (representation - prototypes[predicted_class])
            )

        # ----------------------------------------------------
        # NSAF interaction update
        # ----------------------------------------------------

        model.update(
            x,
            error
        )

    accuracy = correct / len(X_train_np)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Training Accuracy: {accuracy:.4f}"
    )


print("\nNSAF training completed.")

print("Number of learned prototypes:", len(prototypes))
print("Prototype shape:", prototypes.shape)

# ============================================================
# Testing on unseen data
# ============================================================

X_test_np = X_test.values.astype(float)

y_test_encoded = np.array([
    class_to_id[crop]
    for crop in y_test
])

test_correct = 0

predictions = []

for i in range(len(X_test_np)):

    x = X_test_np[i]

    # Generate NSAF representation
    representation = model.forward(x)

    # Predict using learned prototypes
    predicted_class, distances = predict_from_prototypes(
        representation,
        prototypes
    )

    predictions.append(predicted_class)

    if predicted_class == y_test_encoded[i]:
        test_correct += 1


test_accuracy = test_correct / len(X_test_np)

print("\nNSAF TEST RESULTS")
print("-----------------")
print("Test samples:", len(X_test_np))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

# ============================================================
# Detailed Evaluation
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Classification report
print("\nNSAF CLASSIFICATION REPORT")
print("==========================")

print(
    classification_report(
        y_test_encoded,
        predictions,
        target_names=classes,
        zero_division=0
    )
)


# Confusion matrix
cm = confusion_matrix(
    y_test_encoded,
    predictions
)

print("\nConfusion Matrix Shape:", cm.shape)
print("Total predictions:", cm.sum())
print("Correct predictions:", np.trace(cm))
print("Incorrect predictions:", cm.sum() - np.trace(cm))

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("NSAF Crop Recommendation - Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")

plt.xticks(
    range(len(classes)),
    classes,
    rotation=90
)

plt.yticks(
    range(len(classes)),
    classes
)

plt.colorbar()

plt.tight_layout()

plt.show()