
from backend.nsaf_interaction_service import predict_crop

sample_input = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.9,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
}

result = predict_crop(sample_input, top_n=5)

print("Improved NSAF prediction:")
print(result)
