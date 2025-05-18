import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import io
import requests
from datetime import datetime

class DataProcessingAgent:
    """
    Handles all aspects of data ingestion and preparation.
    Responsible for fetching data, cleaning, preprocessing, and preparing it for analysis and modeling.
    """
    
    def __init__(self):
        """Initialize the data processing agent."""
        pass
    
    def ingest_from_file(self, file) -> pd.DataFrame:
        """
        Ingest data from an uploaded file.
        
        Args:
            file: The uploaded file object
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            data = pd.read_csv(file)
            return data
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    def ingest_from_url(self, url: str) -> pd.DataFrame:
        """
        Ingest data from a URL (GitHub, Kaggle, etc.).
        
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
    
    def process(self, data: pd.DataFrame, timestamp_col: str, target_col: str, 
                freq: str, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process the data for time series analysis.
        
        Args:
            data: Input DataFrame
            timestamp_col: Name of the column containing timestamps
            target_col: Name of the column containing target values (load)
            freq: Desired frequency ('H', 'D', 'W', 'M')
            feature_cols: List of additional feature columns to include
            
        Returns:
            Processed DataFrame suitable for time series analysis and modeling
        """
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()
        
        # Process timestamp column
        df = self._process_timestamp(df, timestamp_col, freq)
        
        # Check for and handle missing values
        df = self._handle_missing_values(df, target_col)
        
        # Prepare final dataset with selected columns
        selected_cols = [timestamp_col, target_col]
        if feature_cols:
            selected_cols.extend(feature_cols)
            
        df = df[selected_cols].copy()
        
        # Generate time-based features
        df = self._generate_time_features(df, timestamp_col)
        
        return df
    
    def _process_timestamp(self, df: pd.DataFrame, timestamp_col: str, freq: str) -> pd.DataFrame:
        """
        Process the timestamp column and ensure consistent frequency.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of the column containing timestamps
            freq: Desired frequency
            
        Returns:
            DataFrame with processed timestamp column
        """
        # Ensure timestamp column is in datetime format
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            raise ValueError(f"Error converting {timestamp_col} to datetime: {str(e)}")
        
        # Sort by timestamp
        df = df.sort_values(by=timestamp_col)
        
        # Set timestamp as index temporarily for resampling
        df = df.set_index(timestamp_col)
        
        # Resample to the desired frequency if needed
        try:
            df = df.resample(freq).mean()
            
            # Reset index to get timestamp back as a column
            df = df.reset_index()
        except Exception as e:
            # If resampling fails, just reset the index and continue
            df = df.reset_index()
            print(f"Warning: Could not resample to frequency {freq}: {str(e)}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Detect and handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Name of the column containing target values
            
        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values in target column
        if df[target_col].isna().sum() > 0:
            # Interpolate missing values in target column
            df[target_col] = df[target_col].interpolate(method='linear')
            
            # Fill any remaining NAs (at the start/end)
            df[target_col] = df[target_col].fillna(method='bfill').fillna(method='ffill')
        
        # For other columns, use forward fill then backward fill
        for col in df.columns:
            if col != target_col and df[col].isna().sum() > 0:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _generate_time_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Generate time-based features from the timestamp column.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of the column containing timestamps
            
        Returns:
            DataFrame with additional time-based features
        """
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
