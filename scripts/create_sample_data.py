from src.database import initialize_database


if __name__ == "__main__":
    initialize_database()
    print("Sample orders are ready in business.db; existing tables were preserved.")
