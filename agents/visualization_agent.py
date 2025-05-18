import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

class VisualizationAgent:
    """
    Creates visual representations of data, results, and forecasts.
    Responsible for generating various plots and potentially interactive dashboards
    for easier exploration by the user.
    """
    
    def __init__(self):
        """Initialize the visualization agent."""
        pass
    
    def plot_forecast(self, data: pd.DataFrame, forecasts: pd.DataFrame, 
                     conf_intervals: Optional[Dict[str, Any]] = None, 
                     include_history: bool = True) -> go.Figure:
        """
        Create a plot of forecasted values with optional confidence intervals.
        
        Args:
            data: Original dataframe
            forecasts: Dataframe containing forecasted values
            conf_intervals: Dictionary with confidence interval data
            include_history: Whether to include historical data in the plot
            
        Returns:
            Plotly figure object with the forecast plot
        """
        # Create figure
        fig = go.Figure()
        
        # Determine the column names
        timestamp_col = 'date'
        if timestamp_col not in forecasts.columns:
            timestamp_col = forecasts.columns[0]  # Assume first column is date
        
        forecast_col = 'forecast'
        if forecast_col not in forecasts.columns:
            forecast_col = forecasts.columns[1]  # Assume second column is forecast
        
        # Add historical data if requested
        if include_history:
            # Find timestamp and value columns in original data
            if isinstance(data.index, pd.DatetimeIndex):
                hist_dates = data.index
                value_col = data.columns[0]  # Assume first column is the target
                hist_values = data[value_col]
            else:
                # Look for timestamp column
                timestamp_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                if timestamp_cols:
                    hist_dates = pd.to_datetime(data[timestamp_cols[0]])
                    
                    # Look for value column (could be 'load', 'target', etc.)
                    value_cols = [col for col in data.columns if any(val in col.lower() for val in 
                                                               ['load', 'target', 'value', 'demand'])]
                    if value_cols:
                        value_col = value_cols[0]
                    else:
                        # Use the first numeric column that's not the timestamp
                        value_col = [col for col in data.columns if col != timestamp_cols[0] 
                                   and pd.api.types.is_numeric_dtype(data[col])][0]
                    
                    hist_values = data[value_col]
                else:
                    # If no timestamp column found, we can't include history
                    include_history = False
            
            if include_history:
                fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=hist_values,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecasts[timestamp_col],
            y=forecasts[forecast_col],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals if provided
        if conf_intervals and 'lower_bound' in forecasts.columns and 'upper_bound' in forecasts.columns:
            fig.add_trace(go.Scatter(
                x=forecasts[timestamp_col],
                y=forecasts['upper_bound'],
                mode='lines',
                name=f'Upper Bound ({conf_intervals["level"]*100:.0f}%)',
                line=dict(width=0),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=forecasts[timestamp_col],
                y=forecasts['lower_bound'],
                mode='lines',
                name=f'Lower Bound ({conf_intervals["level"]*100:.0f}%)',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.1)',
                fill='tonexty',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title='Load Forecast',
            xaxis_title='Date/Time',
            yaxis_title='Load',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], 
                             selected_metrics: List[str] = None) -> go.Figure:
        """
        Create a model comparison plot based on selected metrics.
        
        Args:
            metrics: Dictionary of model metrics
            selected_metrics: List of metrics to include in the plot
            
        Returns:
            Plotly figure object with the comparison plot
        """
        # If no metrics specified, use MAPE and R2
        if selected_metrics is None:
            selected_metrics = ['mape', 'r2']
        
        # Create a list of dictionaries for plotting
        plot_data = []
        for model_name, model_metrics in metrics.items():
            for metric in selected_metrics:
                if metric in model_metrics:
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': model_metrics[metric]
                    })
        
        # Convert to dataframe
        df = pd.DataFrame(plot_data)
        
        # Create the bar plot
        fig = px.bar(
            df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Metric Value', 'Model': 'Model Name'},
            height=500
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            legend_title='Metric'
        )
        
        return fig
    
    def plot_residuals(self, actual: np.ndarray, predicted: np.ndarray, 
                      dates: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create a residual plot for model evaluation.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            dates: Optional array of dates for x-axis
            
        Returns:
            Plotly figure object with the residuals plot
        """
        # Calculate residuals
        residuals = actual - predicted
        
        # Create figure
        fig = go.Figure()
        
        # Add residuals line
        if dates is not None:
            x_values = dates
            x_title = 'Date/Time'
        else:
            x_values = np.arange(len(residuals))
            x_title = 'Index'
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='green')
        ))
        
        # Add zero line
        fig.add_shape(
            type='line',
            x0=x_values[0],
            y0=0,
            x1=x_values[-1],
            y1=0,
            line=dict(color='black', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Residuals Plot',
            xaxis_title=x_title,
            yaxis_title='Residual (Actual - Predicted)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_actual_vs_predicted(self, actual: np.ndarray, predicted: np.ndarray, 
                               dates: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create a plot comparing actual and predicted values.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            dates: Optional array of dates for x-axis
            
        Returns:
            Plotly figure object with the actual vs. predicted plot
        """
        # Create figure
        fig = go.Figure()
        
        # Determine x-axis values
        if dates is not None:
            x_values = dates
            x_title = 'Date/Time'
        else:
            x_values = np.arange(len(actual))
            x_title = 'Index'
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=x_values,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=x_values,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Actual vs. Predicted Values',
            xaxis_title=x_title,
            yaxis_title='Value',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_feature_importance(self, model_name: str, feature_names: List[str], 
                              importance: np.ndarray) -> go.Figure:
        """
        Create a feature importance plot.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            importance: Array of importance values
            
        Returns:
            Plotly figure object with the feature importance plot
        """
        # Create a dataframe
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        # Create the bar plot
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance for {model_name}',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
            height=500
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            yaxis=dict(autorange="reversed")  # Highest importance at the top
        )
        
        return fig
    
    def generate_report(self, data: pd.DataFrame, eda_results: Dict[str, Any],
                       model_results: Dict[str, Any], forecast_results: Dict[str, Any],
                       selected_model: str) -> str:
        """
        Generate a comprehensive HTML report for the forecasting process.
        
        Args:
            data: Processed dataframe
            eda_results: Results from EDA
            model_results: Results from model training
            forecast_results: Results from forecasting
            selected_model: Name of the selected model
            
        Returns:
            HTML string containing the complete report
        """
        # Convert Plotly figures to HTML
        time_series_html = eda_results["time_series_plot"].to_html(full_html=False, include_plotlyjs='cdn')
        seasonality_html = eda_results["seasonality_plot"].to_html(full_html=False, include_plotlyjs=False)
        autocorrelation_html = eda_results["autocorrelation_plot"].to_html(full_html=False, include_plotlyjs=False)
        model_comparison_html = model_results["comparison_plot"].to_html(full_html=False, include_plotlyjs=False)
        model_plot_html = model_results["model_plots"][selected_model]["actual_vs_predicted"].to_html(full_html=False, include_plotlyjs=False)
        forecast_plot_html = forecast_results["forecast_plot"].to_html(full_html=False, include_plotlyjs=False)
        
        # Format metrics tables
        eda_stats_html = eda_results["descriptive_stats"].to_html(index=False, classes="table table-striped")
        
        model_metrics_df = pd.DataFrame(model_results["metrics"]).T
        model_metrics_html = model_metrics_df.to_html(classes="table table-striped")
        
        forecast_data_html = forecast_results["forecast_data"].to_html(classes="table table-striped")
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Load Forecasting Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .table {{ margin-bottom: 20px; }}
                .plot {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mt-4 mb-4">Load Forecasting Report</h1>
                
                <div class="section">
                    <h2>1. Data Overview</h2>
                    <p>This report summarizes the load forecasting analysis performed on the provided dataset.</p>
                    <p><strong>Data Shape:</strong> {data.shape[0]} rows, {data.shape[1]} columns</p>
                </div>
                
                <div class="section">
                    <h2>2. Exploratory Data Analysis</h2>
                    
                    <h3>2.1 Time Series Plot</h3>
                    <div class="plot">
                        {time_series_html}
                    </div>
                    
                    <h3>2.2 Descriptive Statistics</h3>
                    <div class="table-responsive">
                        {eda_stats_html}
                    </div>
                    
                    <h3>2.3 Seasonality Decomposition</h3>
                    <div class="plot">
                        {seasonality_html}
                    </div>
                    
                    <h3>2.4 Autocorrelation Analysis</h3>
                    <div class="plot">
                        {autocorrelation_html}
                    </div>
                    
                    <h3>2.5 Key Insights</h3>
                    <pre>{eda_results["insights"]}</pre>
                </div>
                
                <div class="section">
                    <h2>3. Model Evaluation</h2>
                    
                    <h3>3.1 Model Comparison</h3>
                    <div class="plot">
                        {model_comparison_html}
                    </div>
                    
                    <h3>3.2 Performance Metrics</h3>
                    <div class="table-responsive">
                        {model_metrics_html}
                    </div>
                    
                    <h3>3.3 Selected Model: {selected_model}</h3>
                    <div class="plot">
                        {model_plot_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>4. Load Forecast</h2>
                    
                    <h3>4.1 Forecast Plot</h3>
                    <div class="plot">
                        {forecast_plot_html}
                    </div>
                    
                    <h3>4.2 Forecast Data</h3>
                    <div class="table-responsive">
                        {forecast_data_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>5. Conclusions</h2>
                    <p>The load forecasting analysis was performed using various time series models. The {selected_model} model was selected as the best performing model based on evaluation metrics.</p>
                    <p>The forecast provides predictions for future load values, which can be used for planning and resource allocation.</p>
                </div>
                
                <footer class="mt-5 mb-3 text-center text-muted">
                    <p>Generated by the AI Agent-Based Load Forecasting System</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html
