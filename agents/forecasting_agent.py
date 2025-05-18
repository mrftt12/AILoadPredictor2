import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os
from datetime import datetime, timedelta

class ForecastingAgent:
    """
    Generates future load forecasts using the selected model.
    Responsible for loading the deployed model and generating forecasts
    for the specified horizon and granularity.
    """
    
    def __init__(self):
        """Initialize the forecasting agent."""
        pass
    
    def predict(self, model: Dict[str, Any], data: pd.DataFrame, 
               horizon: int, conf_int: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts using the deployed model.
        
        Args:
            model: Dictionary containing the deployed model and metadata
            data: Processed dataframe
            horizon: Number of periods to forecast
            conf_int: Confidence interval (between 0 and 1)
            
        Returns:
            Dictionary containing forecast results and confidence intervals
        """
        model_name = model.get("model_name")
        
        if model_name == "LSTM":
            return self._forecast_with_lstm(model, data, horizon, conf_int)
        elif model_name == "LightGBM":
            return self._forecast_with_lightgbm(model, data, horizon, conf_int)
        elif model_name == "Prophet":
            return self._forecast_with_prophet(model, data, horizon, conf_int)
        elif model_name in ["ARIMA", "SARIMA"]:
            return self._forecast_with_statsmodels(model, data, horizon, conf_int)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def _forecast_with_lstm(self, model_info: Dict[str, Any], data: pd.DataFrame, 
                           horizon: int, conf_int: float) -> Dict[str, Any]:
        """
        Generate forecasts using an LSTM model.
        
        Args:
            model_info: Dictionary containing model and metadata
            data: Processed dataframe
            horizon: Number of periods to forecast
            conf_int: Confidence interval
            
        Returns:
            Dictionary containing forecast results
        """
        # Extract model and metadata
        model = model_info["model"]
        scaler = model_info["scaler"]
        feature_names = model_info["feature_names"]
        timesteps = model_info.get("timesteps", 24)
        
        # Check if we have enough data
        if len(data) < timesteps:
            raise ValueError(f"Not enough data for LSTM forecasting. Need at least {timesteps} datapoints.")
        
        # Prepare features for forecasting
        X = data[feature_names].values
        X_scaled = scaler.transform(X)
        
        # Store original data for later
        original_data = data.copy()
        
        # Get the last known sequence
        last_sequence = X_scaled[-timesteps:]
        
        # Initialize arrays to store forecasts and intervals
        forecasts = []
        upper_intervals = []
        lower_intervals = []
        
        # Generate forecasts iteratively
        for i in range(horizon):
            # Reshape the sequence for LSTM input [samples, timesteps, features]
            X_pred = last_sequence.reshape(1, timesteps, X_scaled.shape[1])
            
            # Generate prediction for the next step
            y_pred = model.predict(X_pred, verbose=0)[0][0]
            forecasts.append(y_pred)
            
            # Generate confidence intervals (using prediction + standard error)
            # For LSTM, we'll approximate using a fixed percentage of the prediction
            error_margin = y_pred * (1 - conf_int)
            upper_intervals.append(y_pred + error_margin)
            lower_intervals.append(max(0, y_pred - error_margin))  # Ensure non-negative loads
            
            # Update sequence for next prediction (remove oldest, add newest prediction)
            # For simplicity, we'll assume the feature values remain the same as the last known values
            # except for any time-based features that might need updating
            next_features = X_scaled[-1].copy()  # Use the last row of features as a template
            
            # Shift the sequence
            last_sequence = np.vstack([last_sequence[1:], next_features])
        
        # Create forecast dates
        last_date = original_data.index[-1] if isinstance(original_data.index, pd.DatetimeIndex) else pd.to_datetime(original_data.iloc[-1].name)
        forecast_dates = self._generate_future_dates(last_date, horizon)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': lower_intervals,
            'upper_bound': upper_intervals
        })
        
        return {
            "predictions": forecast_df,
            "conf_intervals": {
                "lower": lower_intervals,
                "upper": upper_intervals,
                "level": conf_int
            },
            "metrics": None  # No metrics for future forecasts
        }
    
    def _forecast_with_lightgbm(self, model_info: Dict[str, Any], data: pd.DataFrame, 
                              horizon: int, conf_int: float) -> Dict[str, Any]:
        """
        Generate forecasts using a LightGBM model.
        
        Args:
            model_info: Dictionary containing model and metadata
            data: Processed dataframe
            horizon: Number of periods to forecast
            conf_int: Confidence interval
            
        Returns:
            Dictionary containing forecast results
        """
        # Extract model and metadata
        model = model_info["model"]
        scaler = model_info["scaler"]
        feature_names = model_info["feature_names"]
        
        # Store original data for later
        original_data = data.copy()
        
        # Prepare features for initial forecasting
        X = data[feature_names].values
        X_scaled = scaler.transform(X)
        
        # Initialize arrays to store forecasts and intervals
        forecasts = []
        upper_intervals = []
        lower_intervals = []
        
        # For LightGBM, we need to create future feature values
        # This is challenging as we need to extrapolate time-based features
        
        # Get the last date in the dataset
        if isinstance(original_data.index, pd.DatetimeIndex):
            last_date = original_data.index[-1]
        else:
            timestamp_col = [col for col in original_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if timestamp_col:
                last_date = pd.to_datetime(original_data[timestamp_col[0]].iloc[-1])
            else:
                # If no timestamp column is found, use the last row's index
                last_date = pd.to_datetime(original_data.iloc[-1].name)
        
        # Generate future dates
        forecast_dates = self._generate_future_dates(last_date, horizon)
        
        # For each future date, generate the feature values and predict
        for i, future_date in enumerate(forecast_dates):
            # Create future features based on time components
            next_features = X_scaled[-1].copy()  # Use the last row as template
            
            # Update any time-based features if they exist in the feature names
            time_features = [f for f in feature_names if any(t in f.lower() for t in 
                                                        ['hour', 'day', 'week', 'month', 'year'])]
            
            # If time features exist, update them
            if time_features:
                # Create a features dict for the future date
                future_features = {
                    'hour': future_date.hour,
                    'day': future_date.day,
                    'day_of_week': future_date.dayofweek,
                    'month': future_date.month,
                    'year': future_date.year,
                    'hour_sin': np.sin(2 * np.pi * future_date.hour / 24),
                    'hour_cos': np.cos(2 * np.pi * future_date.hour / 24),
                    'day_of_week_sin': np.sin(2 * np.pi * future_date.dayofweek / 7),
                    'day_of_week_cos': np.cos(2 * np.pi * future_date.dayofweek / 7),
                    'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                    'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                    'is_weekend': 1 if future_date.dayofweek >= 5 else 0
                }
                
                # Update next_features with time components
                for j, feature in enumerate(feature_names):
                    if feature in future_features:
                        # Find the index of this feature in the scaler
                        if isinstance(next_features, np.ndarray):
                            next_features[j] = future_features[feature]
                        else:
                            next_features.iloc[j] = future_features[feature]
            
            # Reshape for prediction
            if isinstance(next_features, np.ndarray):
                X_pred = next_features.reshape(1, -1)
            else:
                X_pred = next_features.values.reshape(1, -1)
            
            # Generate prediction
            y_pred = model.predict(X_pred)[0]
            forecasts.append(y_pred)
            
            # Generate confidence intervals
            # For LightGBM, we'll approximate using a fixed percentage of the prediction
            error_margin = y_pred * (1 - conf_int)
            upper_intervals.append(y_pred + error_margin)
            lower_intervals.append(max(0, y_pred - error_margin))  # Ensure non-negative loads
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': lower_intervals,
            'upper_bound': upper_intervals
        })
        
        return {
            "predictions": forecast_df,
            "conf_intervals": {
                "lower": lower_intervals,
                "upper": upper_intervals,
                "level": conf_int
            },
            "metrics": None  # No metrics for future forecasts
        }
    
    def _forecast_with_prophet(self, model_info: Dict[str, Any], data: pd.DataFrame, 
                             horizon: int, conf_int: float) -> Dict[str, Any]:
        """
        Generate forecasts using a Prophet model.
        
        Args:
            model_info: Dictionary containing model and metadata
            data: Processed dataframe
            horizon: Number of periods to forecast
            conf_int: Confidence interval
            
        Returns:
            Dictionary containing forecast results
        """
        # Extract the model
        model = model_info["model"]
        
        # Determine frequency from data
        freq = self._infer_frequency(data)
        
        # Make future dataframe for prediction
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract the forecasted values
        forecasts = forecast['yhat'].tail(horizon).values
        lower_intervals = forecast['yhat_lower'].tail(horizon).values
        upper_intervals = forecast['yhat_upper'].tail(horizon).values
        forecast_dates = forecast['ds'].tail(horizon).values
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': lower_intervals,
            'upper_bound': upper_intervals
        })
        
        return {
            "predictions": forecast_df,
            "conf_intervals": {
                "lower": lower_intervals,
                "upper": upper_intervals,
                "level": conf_int
            },
            "metrics": None  # No metrics for future forecasts
        }
    
    def _forecast_with_statsmodels(self, model_info: Dict[str, Any], data: pd.DataFrame, 
                                 horizon: int, conf_int: float) -> Dict[str, Any]:
        """
        Generate forecasts using a statsmodels model (ARIMA/SARIMA).
        
        Args:
            model_info: Dictionary containing model and metadata
            data: Processed dataframe
            horizon: Number of periods to forecast
            conf_int: Confidence interval
            
        Returns:
            Dictionary containing forecast results
        """
        # Extract the model
        model = model_info["model"]
        model_name = model_info["model_name"]
        
        # Generate forecasts
        forecast_result = model.get_forecast(steps=horizon, alpha=1-conf_int)
        
        # Extract forecast values and confidence intervals
        forecasts = forecast_result.predicted_mean
        conf_intervals = forecast_result.conf_int()
        lower_intervals = conf_intervals.iloc[:, 0].values
        upper_intervals = conf_intervals.iloc[:, 1].values
        
        # Create forecast dates
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.iloc[-1].name)
        forecast_dates = self._generate_future_dates(last_date, horizon)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': lower_intervals,
            'upper_bound': upper_intervals
        })
        
        return {
            "predictions": forecast_df,
            "conf_intervals": {
                "lower": lower_intervals,
                "upper": upper_intervals,
                "level": conf_int
            },
            "metrics": None  # No metrics for future forecasts
        }
    
    def _generate_future_dates(self, last_date: pd.Timestamp, horizon: int) -> List[pd.Timestamp]:
        """
        Generate future dates based on the last date and inferred frequency.
        
        Args:
            last_date: The last date in the dataset
            horizon: Number of periods to forecast
            
        Returns:
            List of future dates
        """
        # Convert to timestamp if not already
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.to_datetime(last_date)
        
        # Infer frequency based on time of day
        if last_date.hour != 0:
            # Assume hourly data
            freq = 'H'
        elif last_date.day != 1:
            # Assume daily data
            freq = 'D'
        elif last_date.month != 1:
            # Assume monthly data
            freq = 'MS'
        else:
            # Default to daily
            freq = 'D'
        
        # Generate future dates
        future_dates = [last_date + pd.Timedelta(i+1, freq) for i in range(horizon)]
        
        return future_dates
    
    def _infer_frequency(self, data: pd.DataFrame) -> str:
        """
        Infer the frequency of the time series data.
        
        Args:
            data: Input dataframe
            
        Returns:
            String representing the frequency ('H', 'D', 'W', 'M', etc.)
        """
        # Check if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            # Try to infer frequency from the index
            freq = pd.infer_freq(data.index)
            
            if freq is not None:
                return freq
            
            # If frequency can't be inferred, try to determine from time differences
            if len(data) > 1:
                # Calculate time differences
                time_diff = data.index[1] - data.index[0]
                
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
        
        # Look for timestamp columns
        timestamp_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if timestamp_cols:
            # Use the first timestamp column
            dates = pd.to_datetime(data[timestamp_cols[0]])
            
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
