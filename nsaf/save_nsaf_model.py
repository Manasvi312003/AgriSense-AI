import pickle
import pandas as pd

# Trained NSAF objects
from train_nsaf import model, prototypes, classes

# Original dataset se fuzzy boundaries save karenge
df = pd.read_csv("data/Crop_recommendation.csv")

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

fuzzy_boundaries = {}

for feature in features:

    fuzzy_boundaries[feature] = {
        "minimum": float(df[feature].min()),
        "q25": float(df[feature].quantile(0.25)),
        "median": float(df[feature].median()),
        "q75": float(df[feature].quantile(0.75)),
        "maximum": float(df[feature].max())
    }


# Complete NSAF model
nsaf_model = {

    # Learned prototypes
    "prototypes": prototypes,

    # Crop names
    "classes": classes,

    # Learned NSAF interaction matrix
    "W": model.W,

    # Learned adaptive context strength
    "context_strength": model.context_strength,

    # Fuzzy membership boundaries
    "fuzzy_boundaries": fuzzy_boundaries,

    # Input features
    "features": features
}


# Save complete model
with open("data/nsaf_crop_model.pkl", "wb") as f:
    pickle.dump(nsaf_model, f)


print("\nComplete NSAF model saved successfully!")

print("Prototypes:", prototypes.shape)
print("W matrix:", model.W.shape)
print("Context strength:", model.context_strength.shape)
print("Classes:", len(classes))

print("\nModel file:")
print("data/nsaf_crop_model.pkl")