
import pickle
from pathlib import Path
from pyexpat import model

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from train_nsaf import (
    DATASET_PATH,
    FEATURES,
    SEED,
    TEST_SIZE,
    fit_fuzzy_boundaries,
    transform_dataframe,
)

from experiment_nsaf_interaction import TrainableNSAF


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "nsaf_interaction_prototype_model.pkl"


def main():
    # Recreate the same development/test split
    df = pd.read_csv(DATASET_PATH)

    development_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    # Fit preprocessing on development data only
    boundaries = fit_fuzzy_boundaries(development_df)

    X_dev = transform_dataframe(
        development_df, boundaries
    ).to_numpy(dtype=float)

    X_test = transform_dataframe(
        test_df, boundaries
    ).to_numpy(dtype=float)

    classes = sorted(development_df["label"].unique())
    class_to_id = {
        name: index for index, name in enumerate(classes)
    }

    y_dev = np.array([
        class_to_id[name] for name in development_df["label"]
    ])

    y_test = np.array([
        class_to_id[name] for name in test_df["label"]
    ])

    # Fixed selected configuration
    weight_lr = 0.001
    context_lr = 0.0001
    head_lr = 0.01
    weight_decay = 0.001
    epochs = 30

    rng = np.random.default_rng(SEED)

    model = TrainableNSAF(
        input_size=X_dev.shape[1],
        num_classes=len(classes),
        seed=SEED,
    )

    # Train on all development samples
    for _ in range(epochs):
        for index in rng.permutation(len(X_dev)):
            x = X_dev[index]
            target = y_dev[index]

            representation, context, logits = model.forward(x)

            logits = logits - np.max(logits)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()

            d_logits = probabilities.copy()
            d_logits[target] -= 1.0

            old_head = model.head.copy()
            d_representation = old_head.T @ d_logits

            model.head -= head_lr * np.outer(
                d_logits, representation
            )
            model.bias -= head_lr * d_logits

            d_context = (
                d_representation * model.context_strength
            )
            d_interaction = d_context * (1.0 - context**2)

            d_W = np.outer(x, d_interaction)
            d_W += weight_decay * model.W
            model.W -= weight_lr * d_W

            d_strength = d_representation * context
            model.context_strength -= (
                context_lr * d_strength
            )
            model.context_strength = np.clip(
                model.context_strength, 0.1, 5.0
            )
 
# --------------------------------------------------------
# 1. Evaluate original classification-head accuracy
# --------------------------------------------------------

    head_predictions = []

    for x in X_test:
        _, _, logits = model.forward(x)
        head_predictions.append(int(np.argmax(logits)))

    head_test_accuracy = accuracy_score(y_test, head_predictions)


# --------------------------------------------------------
# 2. Create one prototype per crop in the learned
#    representation space
# --------------------------------------------------------

    dev_representations = np.array([
        model.forward(x)[0]
        for x in X_dev
    ])

    num_classes = len(classes)

    prototypes = np.zeros(
        (num_classes, dev_representations.shape[1]),
        dtype=float,
    )

    for class_id in range(num_classes):
        class_representations = dev_representations[
            y_dev == class_id
        ]

        prototypes[class_id] = class_representations.mean(axis=0)


    # --------------------------------------------------------
    # 3. Evaluate prototype-based distance accuracy
    # --------------------------------------------------------

    distance_predictions = []
    test_distances = []

    for x in X_test:
        representation, _, _ = model.forward(x)

        distances = np.sum(
            (prototypes - representation) ** 2,
            axis=1,
        )

        predicted_class = int(np.argmin(distances))

        distance_predictions.append(predicted_class)
        test_distances.append(distances)

    distance_test_accuracy = accuracy_score(
        y_test,
        distance_predictions,
    )

    distance_test_precision = precision_score(
        y_test,
        distance_predictions,
        average="macro",
        zero_division=0,
    )

    distance_test_recall = recall_score(
        y_test,
        distance_predictions,
        average="macro",
        zero_division=0,
    )

    distance_test_f1 = f1_score(
        y_test,
        distance_predictions,
        average="macro",
        zero_division=0,
    )

    print("\nPrototype-Distance Model Metrics")
    print(f"Accuracy:  {distance_test_accuracy * 100:.2f}%")
    print(f"Precision: {distance_test_precision * 100:.2f}%")
    print(f"Recall:    {distance_test_recall * 100:.2f}%")
    print(f"F1 Score:  {distance_test_f1 * 100:.2f}%")

    
    # --------------------------------------------------------
    # 4. Save model and prototypes
    # --------------------------------------------------------

    model_data = {
        "W": model.W.copy(),
        "context_strength": model.context_strength.copy(),
        "head": model.head.copy(),
        "bias": model.bias.copy(),
        "prototypes": prototypes,
        "classes": classes,
        "features": FEATURES,
        "fuzzy_boundaries": boundaries,
        "config": {
            "weight_lr": weight_lr,
            "context_lr": context_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "seed": SEED,
        },
        "test_accuracy": float(head_test_accuracy),
        "prototype_test_accuracy": float(distance_test_accuracy),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print("\nImproved NSAF prototype model saved successfully.")
    print("Model path:", MODEL_PATH)
    print("Number of classes:", len(classes))
    print("Prototype shape:", prototypes.shape)
    print("Test samples:", len(test_df))
    print(f"Classification-head accuracy: {head_test_accuracy:.2%}")
    print(f"Prototype-distance accuracy: {distance_test_accuracy:.2%}")
    

if __name__ == "__main__":
    main()
