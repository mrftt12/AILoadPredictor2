import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import io
import requests
from datetime import datetime, timedelta

def load_csv_data(file_path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path_or_buffer: Path to CSV file or file-like object
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        data = pd.read_csv(file_path_or_buffer)
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")

def load_data_from_url(url: str) -> pd.DataFrame:
    """
    Fetch data from a URL and load it into a dataframe.
    
    Args:
        url: URL pointing to a CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = pd.read_csv(io.StringIO(response.text))
        return data
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching data from URL: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing data from URL: {str(e)}")

def validate_data_for_forecasting(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate if the data is suitable for time series forecasting.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if dataframe is empty
    if data.empty:
        return False, "Dataset is empty"
    
    # Check if there's a potential timestamp column
    timestamp_col = None
    for col in data.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        return False, "No timestamp column found. Expected column with 'date' or 'time' in the name."
    
    # Try to convert timestamp to datetime
    try:
        pd.to_datetime(data[timestamp_col])
    except Exception as e:
        return False, f"Failed to convert '{timestamp_col}' to datetime: {str(e)}"
    
    # Check if there's at least one numeric column for forecasting
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return False, "No numeric columns found for forecasting"
    
    # Check for sufficient data points
    if len(data) < 30:
        return False, "Insufficient data points for reliable forecasting (less than 30)"
    
    return True, None

def preprocess_for_forecasting(data: pd.DataFrame, timestamp_col: str, 
                             target_col: str, freq: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocess data for time series forecasting.
    
    Args:
        data: Input dataframe
        timestamp_col: Column name containing timestamps
        target_col: Column name containing target values (load)
        freq: Desired frequency ('H', 'D', 'W', 'M')
        
    Returns:
        Preprocessed dataframe
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp
    df = df.sort_values(by=timestamp_col)
    
    # Set timestamp as index
    df = df.set_index(timestamp_col)
    
    # Resample if frequency is specified
    if freq is not None:
        df = df.resample(freq).mean()
    
    # Handle missing values
    df[target_col] = df[target_col].interpolate(method='linear')
    
    # Reset index to get timestamp back as column
    df = df.reset_index()
    
    return df

def generate_time_features(data: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Generate time-based features from the timestamp column.
    
    Args:
        data: Input dataframe
        timestamp_col: Column name containing timestamps
        
    Returns:
        Dataframe with additional time-based features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure timestamp column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract time components
    df['hour'] = df[timestamp_col].dt.hour
    df['day'] = df[timestamp_col].dt.day
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    
    # Create cyclical features for hour, day of week, and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Flag for weekend
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def split_data(data: pd.DataFrame, train_size: float = 0.8, 
               val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: Input dataframe
        train_size: Proportion for training set
        val_size: Proportion for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure proportions are valid
    if train_size + val_size >= 1.0:
        val_size = 0.1
        train_size = 0.8
    
    # Calculate split indices
    n = len(data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split data
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def calculate_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for forecast evaluation.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
    
    # Ensure inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(actual, predicted) * 100  # Convert to percentage
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Scaled Error (MASE)
    if len(actual) > 1:
        # Calculate naive forecast errors (one-step ahead)
        naive_errors = np.abs(np.diff(actual))
        naive_mae = np.mean(naive_errors)
        
        # Calculate MASE
        if naive_mae != 0:
            mase = mae / naive_mae
        else:
            mase = np.nan
    else:
        mase = np.nan
    
    return {
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "mase": mase
    }

def infer_data_frequency(data: pd.DataFrame, timestamp_col: str) -> str:
    """
    Infer the frequency of time series data.
    
    Args:
        data: Input dataframe
        timestamp_col: Column name containing timestamps
        
    Returns:
        String representing frequency ('H', 'D', 'W', 'M', etc.)
    """
    # Ensure timestamp column is datetime
    dates = pd.to_datetime(data[timestamp_col])
    
    # Try to infer frequency
    freq = pd.infer_freq(dates)
    
    if freq is not None:
        return freq
    
    # If frequency can't be inferred, try to determine from time differences
    if len(dates) > 1:
        # Calculate time differences
        time_diff = dates.iloc[1] - dates.iloc[0]
        
        # Determine frequency based on time difference
        if time_diff.total_seconds() < 3600:
            return 'min'  # Minutes
        elif time_diff.total_seconds() < 86400:
            return 'H'    # Hours
        elif time_diff.total_seconds() < 604800:
            return 'D'    # Days
        elif time_diff.total_seconds() < 2592000:
            return 'W'    # Weeks
        else:
            return 'M'    # Months
    
    # Default to daily if frequency can't be determined
    return 'D'
