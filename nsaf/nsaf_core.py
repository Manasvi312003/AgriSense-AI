import numpy as np


class NSAFCore:

    def __init__(self, input_size=21, learning_rate=0.01):
        self.input_size = input_size
        self.learning_rate = learning_rate

        # Learnable interaction matrix
        self.W = np.random.randn(input_size, input_size) * 0.01

        # Adaptive context strength
        self.context_strength = np.ones(input_size)

    def forward(self, x):
        """
        Create context-aware fuzzy representation.
        """

        interaction = x @ self.W

        context = np.tanh(interaction)

        representation = x + (
            self.context_strength * context
        )

        return representation

    def update(self, x, error):
        """
        Adaptive interaction learning rule.
        """

        interaction_signal = np.outer(x, x)

        self.W += (
            self.learning_rate
            * error
            * interaction_signal
        )

        context_signal = np.abs(x)

        self.context_strength += (
            self.learning_rate
            * error
            * context_signal
        )

        self.context_strength = np.clip(
            self.context_strength,
            0.1,
            5.0
        )

# Simple NSAF test

x = np.random.rand(21)

model = NSAFCore(
    input_size=21,
    learning_rate=0.01
)

output = model.forward(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("First 5 output values:", output[:5])