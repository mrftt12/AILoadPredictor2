import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

class ModelVerificationAgent:
    """
    Evaluates and compares the performance of trained models.
    Responsible for testing models, computing metrics, generating evaluation plots,
    and selecting the best model.
    """
    
    def __init__(self):
        """Initialize the model verification agent."""
        pass
    
    def evaluate(self, data: pd.DataFrame, target_col: str, timestamp_col: str,
                models: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
        """
        Evaluate trained models and select the best one.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            models: Dictionary of trained models and metadata
            metrics: List of metrics to use for evaluation
            
        Returns:
            Dictionary containing evaluation results, plots, and best model selection
        """
        # Create dictionaries to store results
        evaluation_metrics = {}
        model_plots = {}
        
        # Set up comparison data for plotting
        comparison_data = []
        
        # Split into train and test for final evaluation
        train_size = int(len(data) * 0.8)  # Use 80% for consistency
        test_data = data.iloc[train_size:].copy()
        
        # Evaluate each model
        for model_name, model_info in models.items():
            print(f"Evaluating {model_name}...")
            
            # Generate predictions
            y_true, y_pred = self._generate_predictions(
                model_name=model_name,
                model_info=model_info,
                test_data=test_data,
                target_col=target_col,
                timestamp_col=timestamp_col
            )
            
            # Calculate metrics
            model_metrics = self._calculate_metrics(y_true, y_pred)
            evaluation_metrics[model_name] = model_metrics
            
            # Create evaluation plots
            model_plots[model_name] = self._create_evaluation_plots(
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                test_data=test_data,
                timestamp_col=timestamp_col
            )
            
            # Add to comparison data
            for metric, value in model_metrics.items():
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': value
                })
        
        # Create comparison plot
        comparison_plot = self._create_comparison_plot(comparison_data)
        
        # Select best model
        best_model = self._select_best_model(evaluation_metrics)
        
        # Combine results
        results = {
            "metrics": evaluation_metrics,
            "model_plots": model_plots,
            "comparison_plot": comparison_plot,
            "best_model": best_model
        }
        
        return results
    
    def _generate_predictions(self, model_name: str, model_info: Dict[str, Any],
                             test_data: pd.DataFrame, target_col: str,
                             timestamp_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using the specified model.
        
        Args:
            model_name: Name of the model
            model_info: Dictionary containing model and metadata
            test_data: Test dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            
        Returns:
            Tuple of (true values, predicted values)
        """
        y_true = test_data[target_col].values
        
        if model_name == "LSTM":
            # Prepare data for LSTM prediction
            feature_cols = model_info["feature_names"]
            scaler = model_info["scaler"]
            timesteps = model_info.get("timesteps", 24)
            model = model_info["model"]
            
            # Scale features
            X_test = test_data[feature_cols].values
            X_test_scaled = scaler.transform(X_test)
            
            # Create sequences for LSTM
            X_test_lstm = []
            for i in range(len(X_test_scaled) - timesteps):
                X_test_lstm.append(X_test_scaled[i:(i + timesteps)])
            
            X_test_lstm = np.array(X_test_lstm)
            
            # Generate predictions
            if len(X_test_lstm) > 0:
                y_pred = model.predict(X_test_lstm, verbose=0).flatten()
                y_true = y_true[timesteps:]  # Adjust true values to match predictions
            else:
                # If not enough data for sequences
                y_pred = np.array([])
            
        elif model_name == "LightGBM":
            # Prepare data for LightGBM prediction
            feature_cols = model_info["feature_names"]
            scaler = model_info["scaler"]
            model = model_info["model"]
            
            # Scale features
            X_test = test_data[feature_cols].values
            X_test_scaled = scaler.transform(X_test)
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            
        elif model_name == "Prophet":
            # Prepare data for Prophet prediction
            model = model_info["model"]
            
            # Create future dataframe
            future = pd.DataFrame({
                'ds': test_data[timestamp_col]
            })
            
            # Generate predictions
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
            
        elif model_name in ["ARIMA", "SARIMA"]:
            # For ARIMA and SARIMA, simply forecast
            model = model_info["model"]
            
            # Generate predictions
            y_pred = model.forecast(steps=len(test_data))
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Ensure y_true and y_pred have the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        return y_true, y_pred
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "mape": float('inf'),
                "rmse": float('inf'),
                "r2": -float('inf'),
                "mae": float('inf')
            }
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # to percentage
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            "mape": mape,
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        }
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str, test_data: pd.DataFrame,
                               timestamp_col: str) -> Dict[str, go.Figure]:
        """
        Create evaluation plots for a model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            test_data: Test dataframe
            timestamp_col: Column name containing timestamps
            
        Returns:
            Dictionary of Plotly figures
        """
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            # Create empty plots with message
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Insufficient data for evaluation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            return {
                "actual_vs_predicted": empty_fig.copy(),
                "residuals": empty_fig.copy()
            }
        
        # Limit to the common length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Get timestamps for plotting
        timestamps = test_data[timestamp_col].values[:min_len]
        
        # Create actual vs predicted plot
        actual_vs_pred_fig = go.Figure()
        
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        actual_vs_pred_fig.update_layout(
            title=f'{model_name} - Actual vs Predicted',
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title='Legend',
            template='plotly_white'
        )
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create residuals plot
        residuals_fig = go.Figure()
        
        residuals_fig.add_trace(go.Scatter(
            x=timestamps,
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='green')
        ))
        
        residuals_fig.add_shape(
            type='line',
            x0=timestamps[0],
            y0=0,
            x1=timestamps[-1],
            y1=0,
            line=dict(color='black', dash='dash')
        )
        
        residuals_fig.update_layout(
            title=f'{model_name} - Residuals',
            xaxis_title='Time',
            yaxis_title='Residual',
            template='plotly_white'
        )
        
        return {
            "actual_vs_predicted": actual_vs_pred_fig,
            "residuals": residuals_fig
        }
    
    def _create_comparison_plot(self, comparison_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a comparison plot of model metrics.
        
        Args:
            comparison_data: List of dictionaries with model, metric, and value
            
        Returns:
            Plotly figure with comparison plot
        """
        # Convert to DataFrame for plotting
        df_comparison = pd.DataFrame(comparison_data)
        
        # Focus on MAPE and R2
        df_filtered = df_comparison[df_comparison['Metric'].isin(['mape', 'r2'])]
        
        # Create the comparison plot
        fig = px.bar(
            df_filtered,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Metric Value', 'Model': 'Model Name'},
            height=500
        )
        
        # Customize for R2 and MAPE (lower MAPE and higher R2 is better)
        fig.update_layout(
            template='plotly_white',
            legend_title='Metric'
        )
        
        return fig
    
    def _select_best_model(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Select the best model based on multiple metrics.
        
        Args:
            metrics: Dictionary of model metrics
            
        Returns:
            Name of the best model
        """
        # Create a score for each model based on MAPE (50% weight) and R2 (50% weight)
        # Lower MAPE and higher R2 are better
        scores = {}
        
        for model_name, model_metrics in metrics.items():
            # Get MAPE and R2 values
            mape = model_metrics.get('mape', float('inf'))
            r2 = model_metrics.get('r2', -float('inf'))
            
            # Handle invalid values
            if np.isnan(mape) or np.isinf(mape):
                mape = float('inf')
            if np.isnan(r2) or np.isinf(r2):
                r2 = -float('inf')
            
            # Calculate normalized scores (0-1, higher is better)
            # For MAPE: lower is better, so we invert: 1 / (1 + MAPE)
            # For R2: higher is better, so we normalize: (R2 + 1) / 2 [assuming R2 range: -1 to 1]
            mape_score = 1 / (1 + mape/100)  # Convert MAPE to 0-1 scale
            r2_score_norm = (r2 + 1) / 2 if r2 >= -1 else 0  # Convert R2 to 0-1 scale
            
            # Combined score (50% MAPE, 50% R2)
            combined_score = 0.5 * mape_score + 0.5 * r2_score_norm
            
            scores[model_name] = combined_score
        
        # Select the model with the highest score
        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to the first model if no scores available
            best_model = list(metrics.keys())[0] if metrics else None
        
        return best_model
