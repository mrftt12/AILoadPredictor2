import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_hourly_load_dataset(start_date='2023-01-01', periods=8760, output_path='data/sample_hourly_load.csv'):
    """
    Create a sample hourly load dataset with realistic patterns.
    
    Args:
        start_date: Start date for the dataset
        periods: Number of hours (default is 8760, about 1 year)
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame containing the sample data
    """
    # Create date range with hourly frequency
    dates = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Create base load following a seasonal pattern (higher in winter and summer)
    t = np.arange(periods)
    yearly_seasonality = 10000 + 3000 * np.sin(2 * np.pi * t / (365 * 24))  # Yearly cycle
    
    # Add daily pattern (higher during day, lower at night)
    daily_pattern = 2000 * np.sin(2 * np.pi * (t % 24) / 24)
    
    # Add weekly pattern (lower on weekends)
    weekday = pd.Series(dates).dt.dayofweek
    is_weekend = (weekday >= 5).astype(int)
    weekly_pattern = -1500 * is_weekend
    
    # Add noise
    noise = np.random.normal(0, 500, periods)
    
    # Combine all patterns
    load = yearly_seasonality + daily_pattern + weekly_pattern + noise
    
    # Ensure no negative values
    load = np.maximum(load, 0)
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'load': load,
        'temperature': 20 + 15 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 3, periods),
        'humidity': 50 + 20 * np.sin(2 * np.pi * (t + 1000) / (365 * 24)) + np.random.normal(0, 5, periods),
        'is_holiday': np.random.choice([0, 1], size=periods, p=[0.97, 0.03]),
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

def create_daily_load_dataset(start_date='2023-01-01', periods=365, output_path='data/sample_daily_load.csv'):
    """
    Create a sample daily load dataset.
    
    Args:
        start_date: Start date for the dataset
        periods: Number of days (default is 365, about 1 year)
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame containing the sample data
    """
    # Create date range with daily frequency
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Create base load with seasonal pattern
    t = np.arange(periods)
    base_load = 240000 + 70000 * np.sin(2 * np.pi * t / 365)  # Annual cycle
    
    # Add weekly pattern
    weekday = pd.Series(dates).dt.dayofweek
    is_weekend = (weekday >= 5).astype(int)
    weekly_pattern = -30000 * is_weekend
    
    # Add noise
    noise = np.random.normal(0, 10000, periods)
    
    # Combine all patterns
    load = base_load + weekly_pattern + noise
    
    # Ensure no negative values
    load = np.maximum(load, 0)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'load': load,
        'temperature_max': 25 + 15 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, periods),
        'temperature_min': 15 + 15 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, periods),
        'is_holiday': np.random.choice([0, 1], size=periods, p=[0.97, 0.03]),
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    # Create sample datasets
    hourly_df = create_hourly_load_dataset()
    daily_df = create_daily_load_dataset()
    
    print(f"Created hourly dataset with shape: {hourly_df.shape}")
    print(f"Created daily dataset with shape: {daily_df.shape}")