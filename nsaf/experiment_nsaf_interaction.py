
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from train_nsaf import (
    DATASET_PATH,
    FEATURES,
    SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    fit_fuzzy_boundaries,
    transform_dataframe,
)

# Experiment configuration
PROTOTYPE_LR = 0.05
EPOCHS = 30
WEIGHT_LRS = [0.00001, 0.0001, 0.001]
CONTEXT_LRS = [0.00001, 0.0001]


def softmax(logits):
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


class TrainableNSAF:
    def __init__(self, input_size, num_classes, seed=42):
        rng = np.random.default_rng(seed)

        self.W = rng.normal(0, 0.01, (input_size, input_size))
        self.context_strength = np.ones(input_size)

        # Linear classification head over NSAF representation
        self.head = rng.normal(
            0, 0.01, (num_classes, input_size)
        )
        self.bias = np.zeros(num_classes)

    def forward(self, x):
        interaction = x @ self.W
        context = np.tanh(interaction)
        representation = x + self.context_strength * context
        logits = self.head @ representation + self.bias
        return representation, context, logits



def train_and_evaluate(
    X_train, y_train, X_val, y_val, num_classes,
    weight_lr, context_lr, seed
):
    rng = np.random.default_rng(seed)
    model = TrainableNSAF(X_train.shape[1], num_classes, seed)

    head_lr = 0.01
    weight_decay = 0.001

    for epoch in range(EPOCHS):
        for index in rng.permutation(len(X_train)):
            x = X_train[index]
            target = y_train[index]

            representation, context, logits = model.forward(x)
            probabilities = softmax(logits)

            d_logits = probabilities.copy()
            d_logits[target] -= 1.0

            old_head = model.head.copy()
            d_representation = old_head.T @ d_logits

            model.head -= head_lr * np.outer(
                d_logits, representation
            )
            model.bias -= head_lr * d_logits

            d_context = d_representation * model.context_strength
            d_interaction = d_context * (1.0 - context**2)

            d_W = np.outer(x, d_interaction)
            d_W += weight_decay * model.W
            model.W -= weight_lr * d_W

            d_strength = d_representation * context
            model.context_strength -= context_lr * d_strength
            model.context_strength = np.clip(
                model.context_strength, 0.1, 5.0
            )

    predictions = []

    for x in X_val:
        _, _, logits = model.forward(x)
        predictions.append(int(np.argmax(logits)))

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    return accuracy, macro_f1



def main():
    df = pd.read_csv(DATASET_PATH)

    development_df, _ = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    train_df, validation_df = train_test_split(
        development_df,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=development_df["label"],
    )

    boundaries = fit_fuzzy_boundaries(train_df)

    X_train = transform_dataframe(
        train_df, boundaries
    ).to_numpy(dtype=float)

    X_val = transform_dataframe(
        validation_df, boundaries
    ).to_numpy(dtype=float)

    classes = sorted(train_df["label"].unique())
    class_to_id = {
        name: i for i, name in enumerate(classes)
    }

    y_train = np.array([
        class_to_id[name] for name in train_df["label"]
    ])
    y_val = np.array([
        class_to_id[name] for name in validation_df["label"]
    ])

    # results = []

    # for weight_lr in WEIGHT_LRS:
    #     for context_lr in CONTEXT_LRS:
    #         accuracy, macro_f1 = train_and_evaluate(
    #             X_train, y_train, X_val, y_val,
    #             len(classes), weight_lr, context_lr
    #         )

    #         result = {
    #             "weight_lr": weight_lr,
    #             "context_lr": context_lr,
    #             "validation_accuracy": accuracy,
    #             "macro_f1": macro_f1,
    #         }
    #         results.append(result)
    #         print(result)

    # results.sort(
    #     key=lambda item: (
    #         item["validation_accuracy"],
    #         item["macro_f1"],
    #     ),
    #     reverse=True,
    # )

    # print("\nTOP INTERACTION-LEARNING CONFIGURATIONS")
    # for result in results[:3]:
    #     print(result)


    seeds = [42, 123, 2026]
    results = []

    for seed in seeds:
        accuracy, macro_f1 = train_and_evaluate(
            X_train,
            y_train,
            X_val,
            y_val,
            len(classes),
            weight_lr=0.001,
            context_lr=0.0001,
            seed=seed,
        )

        result = {
            "seed": seed,
            "validation_accuracy": accuracy,
            "macro_f1": macro_f1,
        }

        results.append(result)
        print(result)

    accuracies = [
        result["validation_accuracy"] for result in results
    ]
    macro_f1_scores = [
        result["macro_f1"] for result in results
    ]

    print("\nREPRODUCIBILITY SUMMARY")
    print("-----------------------")
    print(f"Mean validation accuracy: {np.mean(accuracies):.4%}")
    print(f"Std validation accuracy:  {np.std(accuracies):.4%}")
    print(f"Mean macro F1:            {np.mean(macro_f1_scores):.4f}")
    print(f"Accuracy range:           {min(accuracies):.4%} - {max(accuracies):.4%}")
    print("\nFinal test set was not evaluated.")


if __name__ == "__main__":
    main()
