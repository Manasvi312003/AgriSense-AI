# import random
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
# )

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)


# # Load fuzzy dataset
# df = pd.read_csv("data/fuzzy_agriculture.csv")

# # Separate features and target
# X = df.drop("label", axis=1)
# y = df["label"]

# print("Feature shape:", X.shape)
# print("Target shape:", y.shape)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# print("\nTraining samples:", X_train.shape)
# print("Testing samples:", X_test.shape)

# print("\nNumber of classes:", y.nunique())
# print("Classes:")
# print(sorted(y.unique()))

# from nsaf_core import NSAFCore


# # Create NSAF model
# model = NSAFCore(
#     input_size=21,
#     learning_rate=0.01
# )

# # Convert training data to numpy
# X_train_np = X_train.values.astype(float)

# # Test NSAF on first training sample
# sample = X_train_np[0]

# representation = model.forward(sample)

# print("\nNSAF input shape:", sample.shape)
# print("NSAF representation shape:", representation.shape)
# print("First 5 NSAF values:", representation[:5])

# # ============================================================
# # NSAF Adaptive Prototype Learning
# # ============================================================

# # Convert crop names into numeric class IDs
# classes = sorted(y_train.unique())

# class_to_id = {
#     crop: i for i, crop in enumerate(classes)
# }

# y_train_encoded = np.array([
#     class_to_id[crop]
#     for crop in y_train
# ])


# # Create one learnable prototype for every crop
# num_classes = len(classes)

# # prototypes = np.zeros(
# #     (num_classes, 21),
# #     dtype=float
# # )

# # Build initial representations from training samples
# initial_representations = np.array([
#     model.forward(x) for x in X_train_np
# ])

# # Initialize each crop prototype using its training samples
# prototypes = np.zeros((num_classes, 21), dtype=float)

# for class_id in range(num_classes):
#     class_representations = initial_representations[
#         y_train_encoded == class_id
#     ]

#     prototypes[class_id] = class_representations.mean(axis=0)

# # Prototype learning rate
# prototype_lr = 0.05


# def predict_from_prototypes(representation, prototypes):
#     """
#     Predict crop using similarity with learned prototypes.
#     """

#     distances = np.sum(
#         (prototypes - representation) ** 2,
#         axis=1
#     )

#     predicted_class = np.argmin(distances)

#     return predicted_class, distances


# # ============================================================
# # Training
# # ============================================================

# epochs = 10

# for epoch in range(epochs):

#     correct = 0

#     for i in range(len(X_train_np)):

#         x = X_train_np[i]

#         # NSAF forward pass
#         representation = model.forward(x)

#         # Current prediction
#         predicted_class, distances = predict_from_prototypes(
#             representation,
#             prototypes
#         )

#         actual_class = y_train_encoded[i]

#         if predicted_class == actual_class:
#             correct += 1

#         # ----------------------------------------------------
#         # Adaptive prototype update
#         # ----------------------------------------------------

#         error = 1.0 if predicted_class != actual_class else 0.0

#         # Move correct class prototype toward representation
#         prototypes[actual_class] += (
#             prototype_lr *
#             (representation - prototypes[actual_class])
#         )

#         # If wrong, push predicted prototype away
#         if predicted_class != actual_class:

#             prototypes[predicted_class] -= (
#                 prototype_lr *
#                 error *
#                 (representation - prototypes[predicted_class])
#             )

#         # ----------------------------------------------------
#         # NSAF interaction update
#         # ----------------------------------------------------

#         # model.update(
#         #     x,
#         #     error
#         # )

#     accuracy = correct / len(X_train_np)

#     print(
#         f"Epoch {epoch + 1}/{epochs} "
#         f"- Training Accuracy: {accuracy:.4f}"
#     )


# print("\nNSAF training completed.")

# print("Number of learned prototypes:", len(prototypes))
# print("Prototype shape:", prototypes.shape)

# # ============================================================
# # Testing on unseen data
# # ============================================================

# X_test_np = X_test.values.astype(float)

# y_test_encoded = np.array([
#     class_to_id[crop]
#     for crop in y_test
# ])

# test_correct = 0

# predictions = []

# for i in range(len(X_test_np)):

#     x = X_test_np[i]

#     # Generate NSAF representation
#     representation = model.forward(x)

#     # Predict using learned prototypes
#     predicted_class, distances = predict_from_prototypes(
#         representation,
#         prototypes
#     )

#     predictions.append(predicted_class)

#     if predicted_class == y_test_encoded[i]:
#         test_correct += 1


# # test_accuracy = test_correct / len(X_test_np)

# # print("\nNSAF TEST RESULTS")
# # print("-----------------")
# # print("Test samples:", len(X_test_np))
# # print(f"Test Accuracy: {test_accuracy:.4f}")
# # print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

# #new test accuracy calculation using sklearn
# test_accuracy = accuracy_score(
#     y_test_encoded,
#     predictions
# )

# print("\nNSAF TEST RESULTS")
# print("-----------------")
# print("Test samples:", len(X_test_np))
# print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")
# # ============================================================
# # Detailed Evaluation
# # ============================================================

# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix
# )

# # Classification report
# print("\nNSAF CLASSIFICATION REPORT")
# print("==========================")

# print(
#     classification_report(
#         y_test_encoded,
#         predictions,
#         target_names=classes,
#         zero_division=0
#     )
# )


# # Confusion matrix
# cm = confusion_matrix(
#     y_test_encoded,
#     predictions
# )

# print("\nConfusion Matrix Shape:", cm.shape)
# print("Total predictions:", cm.sum())
# print("Correct predictions:", np.trace(cm))
# print("Incorrect predictions:", cm.sum() - np.trace(cm))

# import matplotlib.pyplot as plt


# plt.figure(figsize=(12, 10))

# plt.imshow(cm)

# plt.title("NSAF Crop Recommendation - Confusion Matrix")
# plt.xlabel("Predicted Crop")
# plt.ylabel("Actual Crop")

# plt.xticks(
#     range(len(classes)),
#     classes,
#     rotation=90
# )

# plt.yticks(
#     range(len(classes)),
#     classes
# )

# plt.colorbar()

# plt.tight_layout()

# plt.show()

#..........................
#..........................

# new code added in the recent edit:


import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from nsaf_core import NSAFCore


# ============================================================
# Configuration
# ============================================================

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25  # 25% of remaining 80% = 20% overall
EPOCHS = 10
PROTOTYPE_LR = 0.05
MODEL_LR = 0.01

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "data", "Crop_recommendation.csv")

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


# ============================================================
# Fuzzy Membership Function
# Keep this identical to the inference implementation.
# ============================================================

def triangular_membership(x, left, center, right):
    if left > center or center > right:
        raise ValueError("Invalid fuzzy boundaries")

    # All boundaries are identical
    if left == center == right:
        return 1.0 if x == center else 0.0

    # Peak of the membership function
    if x == center:
        return 1.0

    # Left shoulder: peak at the left boundary
    if left == center:
        if x < center:
            return 1.0
        if x >= right:
            return 0.0
        return (right - x) / (right - center)

    # Right shoulder: peak at the right boundary
    if center == right:
        if x > center:
            return 1.0
        if x <= left:
            return 0.0
        return (x - left) / (center - left)

    # Ordinary triangular membership
    if x <= left or x >= right:
        return 0.0

    if x < center:
        return (x - left) / (center - left)

    return (right - x) / (right - center)


# ============================================================
# Fit Fuzzy Boundaries on Training Data ONLY
# ============================================================

def fit_fuzzy_boundaries(train_df):
    boundaries = {}

    for feature in FEATURES:
        values = train_df[feature]

        boundaries[feature] = {
            "minimum": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "maximum": float(values.max()),
        }

    return boundaries


# ============================================================
# Convert Raw Numerical Features to a 21-D Fuzzy Vector
# ============================================================

def create_fuzzy_vector(input_data, fuzzy_boundaries):
    vector = []

    for feature in FEATURES:
        value = float(input_data[feature])
        bounds = fuzzy_boundaries[feature]

        minimum = bounds["minimum"]
        q25 = bounds["q25"]
        median = bounds["median"]
        q75 = bounds["q75"]
        maximum = bounds["maximum"]

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

    return np.asarray(vector, dtype=float)


def transform_dataframe(df, fuzzy_boundaries):
    rows = [
        create_fuzzy_vector(row, fuzzy_boundaries)
        for _, row in df[FEATURES].iterrows()
    ]

    fuzzy_columns = []

    for feature in FEATURES:
        fuzzy_columns.extend([
            f"{feature}_low",
            f"{feature}_medium",
            f"{feature}_high",
        ])

    return pd.DataFrame(
        rows,
        columns=fuzzy_columns,
        index=df.index,
    )


# ============================================================
# Prototype Prediction
# ============================================================

def predict_from_prototypes(representation, prototypes):
    distances = np.sum(
        (prototypes - representation) ** 2,
        axis=1,
    )

    predicted_class = int(np.argmin(distances))
    return predicted_class, distances


# ============================================================
# Main Training Pipeline
# ============================================================

def train_model():
    random.seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + ["label"]
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns."
        )

    # --------------------------------------------------------
    # 1. Split raw data: 80% development, 20% final test
    # --------------------------------------------------------

    development_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    # Split the development data into 60% train and 20% validation
    train_df, validation_df = train_test_split(
        development_df,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=development_df["label"],
    )

    print("Dataset samples:", len(df))
    print("Training samples:", len(train_df))
    print("Validation samples:", len(validation_df))
    print("Test samples:", len(test_df))
    print("Number of crop classes:", df["label"].nunique())

    # --------------------------------------------------------
    # 2. Fit fuzzy boundaries using training data only
    # --------------------------------------------------------

    fuzzy_boundaries = fit_fuzzy_boundaries(train_df)

    X_train = transform_dataframe(
        train_df, fuzzy_boundaries
    ).to_numpy(dtype=float)

    X_validation = transform_dataframe(
        validation_df, fuzzy_boundaries
    ).to_numpy(dtype=float)

    X_test = transform_dataframe(
        test_df, fuzzy_boundaries
    ).to_numpy(dtype=float)

    classes = sorted(train_df["label"].unique())
    class_to_id = {
        crop: index for index, crop in enumerate(classes)
    }

    # Check that every split contains only known training classes
    unknown_labels = (
        set(validation_df["label"])
        | set(test_df["label"])
    ) - set(classes)

    if unknown_labels:
        raise ValueError(
            f"Unknown classes outside training data: {unknown_labels}"
        )

    y_train = np.array([
        class_to_id[label] for label in train_df["label"]
    ])

    y_validation = np.array([
        class_to_id[label] for label in validation_df["label"]
    ])

    y_test = np.array([
        class_to_id[label] for label in test_df["label"]
    ])

    print("\nFuzzy training shape:", X_train.shape)
    print("Fuzzy validation shape:", X_validation.shape)
    print("Fuzzy test shape:", X_test.shape)

    # --------------------------------------------------------
    # 3. Create NSAF model
    # --------------------------------------------------------

    model = NSAFCore(
        input_size=21,
        learning_rate=MODEL_LR,
    )

    # --------------------------------------------------------
    # 4. Initialize prototypes using training representations
    # --------------------------------------------------------

    initial_representations = np.array([
        model.forward(x) for x in X_train
    ])

    num_classes = len(classes)

    prototypes = np.zeros(
        (num_classes, 21),
        dtype=float,
    )

    for class_id in range(num_classes):
        class_representations = initial_representations[
            y_train == class_id
        ]

        prototypes[class_id] = (
            class_representations.mean(axis=0)
        )

    # --------------------------------------------------------
    # 5. Prototype learning
    # Keep model.update() DISABLED for this baseline.
    # This matches the user's better observed configuration.
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        for index in rng.permutation(len(X_train)):
            x = X_train[index]
            actual_class = y_train[index]

            representation = model.forward(x)

            predicted_class, _ = predict_from_prototypes(
                representation,
                prototypes,
            )

            # Move the correct class prototype towards the sample
            prototypes[actual_class] += (
                PROTOTYPE_LR
                * (representation - prototypes[actual_class])
            )

            # Push the incorrect predicted prototype away
            if predicted_class != actual_class:
                prototypes[predicted_class] -= (
                    PROTOTYPE_LR
                    * (representation - prototypes[predicted_class])
                )

            # Intentionally disabled:
            # model.update(x, error)
            #
            # The user's experiments showed better accuracy
            # when the interaction matrix was not updated.

        train_predictions = [
            predict_from_prototypes(
                model.forward(x), prototypes
            )[0]
            for x in X_train
        ]

        validation_predictions = [
            predict_from_prototypes(
                model.forward(x), prototypes
            )[0]
            for x in X_validation
        ]

        train_accuracy = accuracy_score(
            y_train, train_predictions
        )

        validation_accuracy = accuracy_score(
            y_validation, validation_predictions
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Accuracy: {train_accuracy:.4f} "
            f"| Validation Accuracy: {validation_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # 6. Final test evaluation
    # Do not use these results to tune hyperparameters.
    # --------------------------------------------------------

    test_predictions = [
        predict_from_prototypes(
            model.forward(x), prototypes
        )[0]
        for x in X_test
    ]

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    print("\nNSAF FINAL TEST RESULTS")
    print("-----------------------")
    print("Test samples:", len(X_test))
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=np.arange(len(classes)),
    )

    print("Confusion matrix shape:", cm.shape)
    print("Correct predictions:", int(np.trace(cm)))
    print("Incorrect predictions:", int(cm.sum() - np.trace(cm)))

    # Return everything required by the separate saving script
    return {
        "model": model,
        "prototypes": prototypes,
        "classes": classes,
        "fuzzy_boundaries": fuzzy_boundaries,
        "features": FEATURES,
        "test_accuracy": float(test_accuracy),
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
    }


if __name__ == "__main__":
    train_model()
