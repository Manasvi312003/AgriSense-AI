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




# import numpy as np
# import pandas as pd
# import os

# np.random.seed(42)

# # 1. Define distinct agricultural envelopes for each crop type
# # Format: {crop: (min_N, max_N, min_P, max_P, min_K, max_K, ideal_temp, ideal_ph, ideal_rain, base_yield_t_per_ha)}
# crop_profiles = {
#     # Cereals & Grains (High Nitrogen, moderate rain)
#     "rice":        (60, 140, 30, 60,  30, 60,  27.0, 6.0, 200.0, 4.5),
#     "maize":       (60, 120, 40, 70,  30, 60,  24.0, 6.5, 100.0, 5.0),
#     # Fiber & Cash Crops 
#     "cotton":      (40, 100, 30, 60,  30, 60,  26.0, 6.5, 80.0,  2.5),
#     "jute":        (40, 90,  20, 50,  30, 70,  28.0, 6.2, 180.0, 3.0),
#     "coffee":      (60, 110, 20, 50,  60, 120, 22.0, 5.8, 160.0, 2.0),
#     # Legumes & Pulses (Low Nitrogen fixers, highly sensitive to flooding)
#     "chickpea":    (5,  30,  30, 60,  20, 50,  20.0, 7.0, 40.0,  1.5),
#     "lentil":      (5,  25,  25, 50,  20, 40,  18.0, 6.8, 45.0,  1.2),
#     "blackgram":   (10, 30,  30, 60,  20, 50,  28.0, 7.2, 70.0,  1.0),
#     "mungbean":    (10, 30,  25, 50,  20, 50,  29.0, 7.0, 65.0,  0.9),
#     "kidneybeans": (15, 35,  30, 60,  30, 60,  21.0, 6.5, 80.0,  1.4),
#     "mothbeans":   (5,  25,  20, 40,  20, 45,  32.0, 7.5, 35.0,  0.7),
#     "pigeonpeas":  (10, 35,  25, 55,  20, 50,  26.0, 6.7, 85.0,  1.3),
#     # Fruits (High Potassium requirements, highly sensitive to extreme temps/pH)
#     "banana":      (80, 130, 60, 90,  110, 150, 27.0, 6.0, 170.0, 35.0),
#     "apple":       (10, 40,  10, 40,  120, 160, 15.0, 6.2, 90.0,  25.0),
#     "grapes":      (20, 50,  20, 40,  100, 140, 23.0, 6.8, 60.0,  18.0),
#     "mango":       (30, 70,  20, 40,  40, 90,   27.0, 6.0, 110.0, 12.0),
#     "coconut":     (60, 100, 40, 70,  120, 170, 27.0, 5.8, 220.0, 15.0),
#     "orange":      (30, 70,  15, 45,  80, 130,  23.0, 6.3, 100.0, 20.0),
#     "papaya":      (40, 80,  40, 70,  70, 120,  26.0, 6.5, 130.0, 40.0),
#     "pomegranate": (20, 50,  10, 30,  60, 100,  25.0, 7.2, 50.0,  10.0),
#     "muskmelon":   (20, 60,  15, 35,  70, 110,  29.0, 6.7, 55.0,  22.0),
#     "watermelon":  (30, 70,  20, 40,  60, 100,  28.0, 6.5, 60.0,  30.0)
# }

# crops = list(crop_profiles.keys())
# rows = []

# for _ in range(5000):
#     crop = np.random.choice(crops)
#     profile = crop_profiles[crop]
    
#     # 2. Sample NPK within realistic ranges for this specific crop class
#     N = np.random.uniform(profile[0] * 0.7, profile[1] * 1.3)
#     P = np.random.uniform(profile[2] * 0.7, profile[3] * 1.3)
#     K = np.random.uniform(profile[4] * 0.7, profile[5] * 1.3)
    
#     # Sample environmental bounds with variance
#     temperature = np.random.uniform(profile[6] - 8, profile[6] + 8)
#     ph = np.random.uniform(4.8, 8.5)
    
#     # Rainfall variations (ensuring context-appropriate distributions)
#     if crop in ["rice", "jute", "coconut", "banana"]:
#         rainfall = np.random.uniform(120, 350)
#         humidity = np.random.uniform(70, 98)
#     elif crop in ["chickpea", "lentil", "mothbeans", "pomegranate"]:
#         rainfall = np.random.uniform(20, 90)
#         humidity = np.random.uniform(20, 60)
#     else:
#         rainfall = np.random.uniform(60, 180)
#         humidity = np.random.uniform(50, 85)

#     # 3. Apply Multiplicative Biological Penalty Functions
#     ideal_temp, ideal_ph, ideal_rain, base_yield = profile[6], profile[7], profile[8], profile[9]
    
#     # Non-linear quadratic penalties for environmental deviations
#     temp_penalty = np.exp(-0.02 * ((temperature - ideal_temp) ** 2))
#     ph_penalty = np.exp(-0.4 * ((ph - ideal_ph) ** 2))
#     rain_penalty = np.exp(-0.0001 * ((rainfall - ideal_rain) ** 2))
    
#     # Nutrient balance checks (Yield drops if you completely starve the crop of its target macro)
#     opt_N = (profile[0] + profile[1]) / 2
#     opt_K = (profile[4] + profile[5]) / 2
#     n_factor = 1.0 - 0.3 * max(0, (opt_N - N) / opt_N)
#     k_factor = 1.0 - 0.3 * max(0, (opt_K - K) / opt_K)
    
#     # Calculate final yield using calculated biological limits
#     yield_value = base_yield * temp_penalty * ph_penalty * rain_penalty * n_factor * k_factor
    
#     # Inject Gaussian ambient noise representing pests/untracked variations
#     yield_value += np.random.normal(0, base_yield * 0.08)
    
#     # Lower floor boundary control based on realistic minimal yields
#     yield_value = max(yield_value, base_yield * 0.15)

#     rows.append([N, P, K, temperature, humidity, ph, rainfall, crop, yield_value])

# # 4. Export & Construct Dataframe
# columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop", "yield"]
# df = pd.DataFrame(rows, columns=columns)

# os.makedirs("data", exist_ok=True)
# df.to_csv("data/yield_agriculture.csv", index=False)

# print("Advanced Crop-Specific Yield Dataset generated successfully!")
# print("Shape:", df.shape)
# print("\nCrop Distribution Verification:\n", df["crop"].value_counts().head(4))
# print("\nRealistic Yield Descriptions (T/Ha) per Category:\n", df.groupby("crop")["yield"].mean().round(2))
