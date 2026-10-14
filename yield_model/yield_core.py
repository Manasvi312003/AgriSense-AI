import numpy as np


class AdaptiveYieldLearner:

    def __init__(self, input_size=8, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate

        # Adaptive feature weights
        self.weights = np.random.randn(input_size) * 0.01

        # Learnable feature interaction matrix
        self.interaction_matrix = (
            np.random.randn(input_size, input_size) * 0.005
        )

        # Adaptive memory for feature importance
        self.feature_memory = np.ones(input_size)

        # Learnable bias
        self.bias = 0.0

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):

        x = np.asarray(x, dtype=float)

        if len(x) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"but received {len(x)}"
            )

        # Adaptive feature representation
        adaptive_x = x * self.feature_memory

        # Feature interaction
        interaction = adaptive_x @ self.interaction_matrix

        # Non-linear context
        context = np.tanh(interaction)

        # Combined representation
        representation = adaptive_x + context

        # Yield prediction
        prediction = (
            np.dot(self.weights, representation)
            + self.bias
        )

        return prediction

    # =====================================================
    # ADAPTIVE LEARNING
    # =====================================================

    def update(self, x, target):

        x = np.asarray(x, dtype=float)

        # Current prediction
        prediction = self.forward(x)

        # Prediction error
        error = target - prediction

        # Adaptive representation
        adaptive_x = x * self.feature_memory

        # -------------------------------------------------
        # Update feature weights
        # -------------------------------------------------

        self.weights += (
            self.learning_rate
            * error
            * adaptive_x
        )

        # -------------------------------------------------
        # Update bias
        # -------------------------------------------------

        self.bias += (
            self.learning_rate
            * error
        )

        # -------------------------------------------------
        # Learn feature interactions
        # -------------------------------------------------

        interaction_signal = np.outer(
            adaptive_x,
            adaptive_x
        )

        self.interaction_matrix += (
            self.learning_rate
            * error
            * interaction_signal
            * 0.01
        )

        # -------------------------------------------------
        # Adaptive feature memory
        # -------------------------------------------------

        importance_signal = np.abs(x)

        self.feature_memory += (
            self.learning_rate
            * error
            * importance_signal
            * 0.001
        )

        # Keep memory stable
        self.feature_memory = np.clip(
            self.feature_memory,
            0.1,
            5.0
        )

        return prediction, error


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    model = AdaptiveYieldLearner(
        input_size=8,
        learning_rate=0.001
    )

    # N, P, K, temperature, humidity, pH, rainfall, area
    sample = np.array([
        90,
        42,
        43,
        20.88,
        82.0,
        6.5,
        202.94,
        1.5
    ])

    prediction = model.forward(sample)

    print("Adaptive Yield Learner Test")
    print("---------------------------")
    print("Input shape:", sample.shape)
    print("Output:", prediction)
    print("Weights shape:", model.weights.shape)
    print("Interaction matrix shape:", model.interaction_matrix.shape)
    print("Feature memory shape:", model.feature_memory.shape)