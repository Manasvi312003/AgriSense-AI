import numpy as np
import pandas as pd

np.random.seed(42)

crops = [
    "apple", "banana", "blackgram", "chickpea",
    "coconut", "coffee", "cotton", "grapes",
    "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon",
    "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
]

rows = []

for _ in range(5000):

    crop = np.random.choice(crops)

    # Agricultural conditions
    N = np.random.uniform(20, 140)
    P = np.random.uniform(5, 145)
    K = np.random.uniform(5, 205)

    temperature = np.random.uniform(10, 40)
    humidity = np.random.uniform(30, 100)
    ph = np.random.uniform(4.5, 9.5)
    rainfall = np.random.uniform(20, 300)

    # ---------------------------------------------------------
    # SYNTHETIC YIELD GENERATION
    # Target = yield per hectare (t/ha)
    # ---------------------------------------------------------

    yield_value = (
        0.025 * N
        + 0.018 * P
        + 0.015 * K
        + 0.035 * rainfall
        + 0.08 * humidity
        - 0.12 * abs(temperature - 25)
        - 0.8 * abs(ph - 6.5)
        + np.random.normal(0, 1.5)
    )

    # Crop-specific effect
    crop_effect = {
        "rice": 2.0,
        "maize": 1.8,
        "cotton": 1.5,
        "jute": 1.3,
        "coffee": 1.6,
        "banana": 2.5,
        "apple": 2.0,
        "grapes": 1.8,
        "mango": 2.1,
        "coconut": 2.3,
        "chickpea": 1.2,
        "lentil": 1.1,
        "blackgram": 1.0,
        "kidneybeans": 1.2,
        "mungbean": 1.1,
        "mothbeans": 1.0,
        "muskmelon": 1.7,
        "orange": 1.8,
        "papaya": 2.2,
        "pigeonpeas": 1.2,
        "pomegranate": 1.7
    }

    yield_value += crop_effect.get(crop, 1.0)

    # Minimum yield
    yield_value = max(yield_value, 0.1)

    rows.append([
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        crop,
        yield_value
    ])


columns = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "crop",
    "yield"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv(
    "data/yield_agriculture.csv",
    index=False
)

print("Yield dataset generated successfully!")
print("Target unit: tonnes per hectare (t/ha)")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nYield statistics:")
print(df["yield"].describe())