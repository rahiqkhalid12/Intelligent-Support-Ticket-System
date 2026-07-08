from database import engine, Base
import models  # Import models so SQLAlchemy knows about the tables

print("Creating database...")

Base.metadata.create_all(bind=engine)

print("Database created successfully!")