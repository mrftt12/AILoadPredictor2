import os
import pandas as pd
from utils.create_sample_data import create_hourly_load_dataset, create_daily_load_dataset
from utils.db_manager import init_db, store_sample_data
import sqlalchemy

def setup_database():
    """Initialize the database and load sample data"""
    print("Setting up the database...")
    
    # Initialize database tables
    try:
        init_db()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return False
    
    # Create sample datasets
    print("Generating sample datasets...")
    hourly_df = create_hourly_load_dataset()
    daily_df = create_daily_load_dataset()
    
    # Store in the database
    try:
        print("Storing hourly load data in the database...")
        store_sample_data(hourly_df)
        print(f"Stored {len(hourly_df)} rows of hourly load data")
    except Exception as e:
        print(f"Error storing hourly load data: {e}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save sample datasets to CSV
    hourly_df.to_csv('data/sample_hourly_load.csv', index=False)
    daily_df.to_csv('data/sample_daily_load.csv', index=False)
    
    print("Database setup complete!")
    print(f"Sample data files saved to: data/sample_hourly_load.csv and data/sample_daily_load.csv")
    
    return True

if __name__ == "__main__":
    setup_database()