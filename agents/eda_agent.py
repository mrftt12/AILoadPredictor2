import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

class EDAAgent:
    """
    Analyzes the processed data to uncover patterns and insights.
    Responsible for calculating descriptive statistics, generating visualizations,
    identifying trends, seasonality, outliers, and potential correlations.
    """
    
    def __init__(self):
        """Initialize the EDA agent."""
        pass
    
    def analyze(self, data: pd.DataFrame, target_col: str, 
                timestamp_col: str) -> Dict[str, Any]:
        """
        Perform exploratory data analysis on the time series data.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values (load)
            timestamp_col: Column name containing timestamps
            
        Returns:
            Dictionary containing EDA results, visualizations, and insights
        """
        # Set index to timestamp temporarily for time series analysis
        df = data.copy()
        df_ts = df.set_index(timestamp_col)
        
        # Generate descriptive statistics
        descriptive_stats = self._calculate_descriptive_stats(df, target_col)
        
        # Generate time series plot
        time_series_plot = self._create_time_series_plot(df, timestamp_col, target_col)
        
        # Perform seasonality decomposition
        seasonality_plot = self._analyze_seasonality(df_ts, target_col)
        
        # Calculate autocorrelation
        autocorrelation_plot = self._analyze_autocorrelation(df_ts, target_col)
        
        # Generate insights
        insights = self._generate_insights(df, df_ts, target_col, timestamp_col)
        
        # Combine results
        results = {
            "descriptive_stats": descriptive_stats,
            "time_series_plot": time_series_plot,
            "seasonality_plot": seasonality_plot,
            "autocorrelation_plot": autocorrelation_plot,
            "insights": insights
        }
        
        return results
    
    def _calculate_descriptive_stats(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Calculate descriptive statistics for the target variable.
        
        Args:
            df: Input dataframe
            target_col: Column name containing the target values
            
        Returns:
            DataFrame with descriptive statistics
        """
        # Calculate basic statistics
        stats = df[target_col].describe()
        
        # Add additional statistics
        additional_stats = pd.Series({
            'skew': df[target_col].skew(),
            'kurtosis': df[target_col].kurtosis(),
            'median': df[target_col].median(),
            'missing_values': df[target_col].isna().sum(),
            'missing_percentage': (df[target_col].isna().sum() / len(df)) * 100
        })
        
        # Combine into a DataFrame
        stats_df = pd.DataFrame({
            'Statistic': stats.index.tolist() + additional_stats.index.tolist(),
            'Value': stats.values.tolist() + additional_stats.values.tolist()
        })
        
        return stats_df
    
    def _create_time_series_plot(self, df: pd.DataFrame, timestamp_col: str, 
                                target_col: str) -> go.Figure:
        """
        Create a time series plot of the target variable.
        
        Args:
            df: Input dataframe
            timestamp_col: Column name containing timestamps
            target_col: Column name containing the target values
            
        Returns:
            Plotly figure object with the time series plot
        """
        fig = px.line(
            df, 
            x=timestamp_col, 
            y=target_col,
            title=f'Time Series Plot of {target_col}',
            labels={timestamp_col: 'Date/Time', target_col: 'Load'}
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title='Date/Time',
            yaxis_title='Load',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _analyze_seasonality(self, df_ts: pd.DataFrame, target_col: str) -> go.Figure:
        """
        Perform seasonality decomposition on the time series.
        
        Args:
            df_ts: Input dataframe with timestamp index
            target_col: Column name containing the target values
            
        Returns:
            Plotly figure object with the seasonality decomposition plot
        """
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                df_ts[target_col], 
                model='additive', 
                period=self._estimate_seasonality_period(df_ts.index)
            )
            
            # Create subplots
            fig = make_subplots(
                rows=4, 
                cols=1,
                subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.1
            )
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=df_ts.index, y=decomposition.observed, mode='lines', name='Observed'), 
                row=1, col=1
            )
            
            fig.add_trace(go.Scatter(
                x=df_ts.index, y=decomposition.trend, mode='lines', name='Trend'), 
                row=2, col=1
            )
            
            fig.add_trace(go.Scatter(
                x=df_ts.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), 
                row=3, col=1
            )
            
            fig.add_trace(go.Scatter(
                x=df_ts.index, y=decomposition.resid, mode='lines', name='Residual'), 
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800, 
                title_text=f'Seasonality Decomposition of {target_col}',
                template='plotly_white',
                showlegend=False
            )
            
        except Exception as e:
            # If decomposition fails, create a simple message plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not perform seasonality decomposition: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                height=800, 
                title_text=f'Seasonality Decomposition of {target_col} (Failed)',
                template='plotly_white'
            )
        
        return fig
    
    def _analyze_autocorrelation(self, df_ts: pd.DataFrame, target_col: str) -> go.Figure:
        """
        Calculate and plot autocorrelation and partial autocorrelation.
        
        Args:
            df_ts: Input dataframe with timestamp index
            target_col: Column name containing the target values
            
        Returns:
            Plotly figure object with the autocorrelation plots
        """
        try:
            # Calculate ACF and PACF
            acf_values = acf(df_ts[target_col].dropna(), nlags=40)
            pacf_values = pacf(df_ts[target_col].dropna(), nlags=40)
            
            # Create subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=['Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)'],
                vertical_spacing=0.15
            )
            
            # Add ACF bars
            fig.add_trace(go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color='blue'
            ), row=1, col=1)
            
            # Add PACF bars
            fig.add_trace(go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name='PACF',
                marker_color='red'
            ), row=2, col=1)
            
            # Add confidence intervals (approximately ±1.96/sqrt(n))
            ci = 1.96 / np.sqrt(len(df_ts[target_col].dropna()))
            
            for i in range(1, 3):
                fig.add_shape(
                    type='line',
                    xref=f'x{i}',
                    yref=f'y{i}',
                    x0=0,
                    y0=ci,
                    x1=40,
                    y1=ci,
                    line=dict(color='gray', width=1, dash='dash')
                )
                
                fig.add_shape(
                    type=f'line',
                    xref=f'x{i}',
                    yref=f'y{i}',
                    x0=0,
                    y0=-ci,
                    x1=40,
                    y1=-ci,
                    line=dict(color='gray', width=1, dash='dash')
                )
            
            # Update layout
            fig.update_layout(
                height=600, 
                title_text=f'Autocorrelation Analysis of {target_col}',
                template='plotly_white',
                showlegend=False
            )
            
            # Update y-axes
            fig.update_yaxes(title_text='Correlation', range=[-1, 1])
            
            # Update x-axes
            fig.update_xaxes(title_text='Lag')
            
        except Exception as e:
            # If analysis fails, create a simple message plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not perform autocorrelation analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                height=600, 
                title_text=f'Autocorrelation Analysis of {target_col} (Failed)',
                template='plotly_white'
            )
        
        return fig
    
    def _estimate_seasonality_period(self, index) -> int:
        """
        Estimate the seasonality period from the time series index.
        
        Args:
            index: DatetimeIndex of the time series
            
        Returns:
            Estimated seasonality period
        """
        # Get the frequency of the time series
        try:
            # Try to infer frequency
            freq = pd.infer_freq(index)
            
            if freq is None:
                # If frequency cannot be inferred, try to calculate it
                if len(index) > 1:
                    # Calculate the median difference between consecutive timestamps
                    diff = np.median(np.diff(index.astype(np.int64)) / 10**9)
                    
                    # Convert to seconds
                    seconds_diff = diff
                    
                    # Determine frequency based on seconds difference
                    if seconds_diff < 60:  # Less than a minute
                        freq = 'S'  # Seconds
                    elif seconds_diff < 3600:  # Less than an hour
                        freq = 'T'  # Minutes
                    elif seconds_diff < 86400:  # Less than a day
                        freq = 'H'  # Hours
                    elif seconds_diff < 604800:  # Less than a week
                        freq = 'D'  # Days
                    elif seconds_diff < 2592000:  # Less than a month
                        freq = 'W'  # Weeks
                    else:
                        freq = 'M'  # Months
                else:
                    # Default to daily if only one timestamp
                    freq = 'D'
            
            # Set period based on frequency
            if freq in ['S', 'T', 'min']:
                period = 60  # Minute data
            elif freq in ['H']:
                period = 24  # Hourly data
            elif freq in ['D', 'B']:
                period = 7  # Daily data
            elif freq in ['W']:
                period = 52  # Weekly data
            elif freq in ['M', 'MS']:
                period = 12  # Monthly data
            elif freq in ['Q', 'QS']:
                period = 4  # Quarterly data
            else:
                period = 24  # Default to daily (24 hours)
        except Exception as e:
            # Default to 24 (assuming hourly data) if estimation fails
            period = 24
        
        return period
    
    def _generate_insights(self, df: pd.DataFrame, df_ts: pd.DataFrame, 
                          target_col: str, timestamp_col: str) -> str:
        """
        Generate insights based on the EDA.
        
        Args:
            df: Original dataframe
            df_ts: Dataframe with timestamp index
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            
        Returns:
            String with insights
        """
        insights = []
        
        # Target variable statistics
        min_val = df[target_col].min()
        max_val = df[target_col].max()
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        
        insights.append(f"Load Statistics: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Std={std_val:.2f}")
        
        # Time range
        start_date = df[timestamp_col].min()
        end_date = df[timestamp_col].max()
        duration = end_date - start_date
        
        insights.append(f"Time Range: {start_date} to {end_date} ({duration.days} days)")
        
        # Check for missing values
        missing = df[target_col].isna().sum()
        if missing > 0:
            insights.append(f"Missing Values: {missing} ({missing/len(df)*100:.2f}% of data)")
        else:
            insights.append("No missing values detected in the target variable.")
        
        # Check for outliers (z-score > 3)
        z_scores = np.abs((df[target_col] - mean_val) / std_val)
        outliers = df[z_scores > 3]
        
        if len(outliers) > 0:
            insights.append(f"Potential Outliers: {len(outliers)} points ({len(outliers)/len(df)*100:.2f}% of data)")
        else:
            insights.append("No significant outliers detected using z-score method.")
        
        # Time-based patterns
        if 'hour' in df.columns and 'day_of_week' in df.columns:
            # Hourly patterns
            hourly_avg = df.groupby('hour')[target_col].mean()
            peak_hour = hourly_avg.idxmax()
            low_hour = hourly_avg.idxmin()
            
            insights.append(f"Peak Load Hour: {peak_hour}:00, Low Load Hour: {low_hour}:00")
            
            # Day of week patterns
            dow_avg = df.groupby('day_of_week')[target_col].mean()
            peak_day = dow_avg.idxmax()
            low_day = dow_avg.idxmin()
            
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            insights.append(f"Peak Load Day: {days[peak_day]}, Low Load Day: {days[low_day]}")
            
            # Weekend vs weekday
            if 'is_weekend' in df.columns:
                weekend_avg = df[df['is_weekend'] == 1][target_col].mean()
                weekday_avg = df[df['is_weekend'] == 0][target_col].mean()
                
                if weekend_avg < weekday_avg:
                    insights.append(f"Weekend loads are {(weekday_avg-weekend_avg)/weekday_avg*100:.2f}% lower than weekday loads")
                else:
                    insights.append(f"Weekend loads are {(weekend_avg-weekday_avg)/weekday_avg*100:.2f}% higher than weekday loads")
        
        # Trend assessment
        try:
            from scipy import stats
            
            # Simple linear regression to detect trend
            x = np.arange(len(df_ts))
            y = df_ts[target_col].values
            slope, _, r_value, p_value, _ = stats.linregress(x, y)
            
            if p_value < 0.05:
                if slope > 0:
                    insights.append(f"Significant INCREASING trend detected (slope={slope:.4f}, r²={r_value**2:.4f}, p={p_value:.4f})")
                else:
                    insights.append(f"Significant DECREASING trend detected (slope={slope:.4f}, r²={r_value**2:.4f}, p={p_value:.4f})")
            else:
                insights.append(f"No significant trend detected (p={p_value:.4f})")
                
        except Exception as e:
            insights.append("Could not assess trend due to an error in calculation.")
        
        # Join insights with line breaks
        return "\n\n".join(insights)
