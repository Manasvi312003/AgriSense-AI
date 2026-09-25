from database.database import engine, Base
from database.models import (
    User,
    FarmRecord,
    Prediction,
    CropRecommendation
)


print("Creating AgriSense AI database tables...")

Base.metadata.create_all(
    bind=engine
)

print("All tables created successfully!")