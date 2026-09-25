from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# PostgreSQL connection
DATABASE_URL = (
    "postgresql+psycopg2://postgres:Qwerty%23%40123@localhost:5432/agrisense_db"
)


# Create database engine
engine = create_engine(
    DATABASE_URL,
    echo=False
)


# Create session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Base class for database models
Base = declarative_base()


# Database dependency
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()