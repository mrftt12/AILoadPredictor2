import streamlit as st
import os
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="AI Agent-Based Load Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Ingestion"
    
# Helper functions for direct implementation
def load_data_from_url(url):
    """Load data from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None

def process_data(data, timestamp_col, target_col, freq):
    """Process the data for time series analysis"""
    df = data.copy()
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp
    df = df.sort_values(by=timestamp_col)
    
    # Set timestamp as index for resampling
    df = df.set_index(timestamp_col)
    
    # Resample to desired frequency
    try:
        df = df.resample(freq).mean()
    except Exception as e:
        st.warning(f"Could not resample to frequency {freq}. Using original data.")
    
    # Fill missing values
    df = df.interpolate(method='linear')
    
    # Reset index to get timestamp back as a column
    df = df.reset_index()
    
    # Generate time features
    df['hour'] = df[timestamp_col].dt.hour
    df['day'] = df[timestamp_col].dt.day
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def perform_eda(data, target_col, timestamp_col):
    """Perform exploratory data analysis"""
    # Copy data and set timestamp as index for time series analysis
    df = data.copy()
    df_ts = df.set_index(timestamp_col)
    
    # Descriptive statistics
    stats = df[target_col].describe()
    additional_stats = pd.Series({
        'skew': df[target_col].skew(),
        'kurtosis': df[target_col].kurtosis(),
        'median': df[target_col].median()
    })
    stats_df = pd.DataFrame({
        'Statistic': stats.index.tolist() + additional_stats.index.tolist(),
        'Value': stats.values.tolist() + additional_stats.values.tolist()
    })
    
    # Time series plot
    time_series_plot = px.line(
        df, 
        x=timestamp_col, 
        y=target_col,
        title=f'Time Series Plot of {target_col}'
    )
    time_series_plot.update_layout(height=500)
    
    # Create autocorrelation plot
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    
    # Calculate ACF and PACF values
    from statsmodels.tsa.stattools import acf, pacf
    try:
        acf_values = acf(df_ts[target_col].dropna(), nlags=40)
        pacf_values = pacf(df_ts[target_col].dropna(), nlags=40)
        
        # Create figure
        autocorr_fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=['Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)']
        )
        
        # Add ACF trace
        autocorr_fig.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
            row=1, col=1
        )
        
        # Add PACF trace
        autocorr_fig.add_trace(
            go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
            row=2, col=1
        )
        
        autocorr_fig.update_layout(height=600, showlegend=False)
    except:
        # Fallback if ACF calculation fails
        autocorr_fig = go.Figure()
        autocorr_fig.add_annotation(
            text="Could not calculate autocorrelation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Generate seasonality plot
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(
            df_ts[target_col],
            model='additive',
            period=12  # Default period
        )
        
        seasonality_fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual']
        )
        
        # Add traces
        seasonality_fig.add_trace(
            go.Scatter(x=df_ts.index, y=decomposition.observed, mode='lines'),
            row=1, col=1
        )
        
        seasonality_fig.add_trace(
            go.Scatter(x=df_ts.index, y=decomposition.trend, mode='lines'),
            row=2, col=1
        )
        
        seasonality_fig.add_trace(
            go.Scatter(x=df_ts.index, y=decomposition.seasonal, mode='lines'),
            row=3, col=1
        )
        
        seasonality_fig.add_trace(
            go.Scatter(x=df_ts.index, y=decomposition.resid, mode='lines'),
            row=4, col=1
        )
        
        seasonality_fig.update_layout(height=800, showlegend=False)
    except:
        # Fallback if decomposition fails
        seasonality_fig = go.Figure()
        seasonality_fig.add_annotation(
            text="Could not perform seasonal decomposition",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Generate insights
    insights = []
    
    # Basic stats
    insights.append(f"Average {target_col}: {df[target_col].mean():.2f}")
    insights.append(f"Range: {df[target_col].min():.2f} to {df[target_col].max():.2f}")
    
    # Time patterns if available
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')[target_col].mean()
        peak_hour = hourly_avg.idxmax()
        insights.append(f"Peak hour: {peak_hour}:00")
    
    if 'day_of_week' in df.columns:
        daily_avg = df.groupby('day_of_week')[target_col].mean()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_day = days[daily_avg.idxmax()]
        insights.append(f"Peak day: {peak_day}")
    
    if 'is_weekend' in df.columns:
        weekend_avg = df[df['is_weekend'] == 1][target_col].mean()
        weekday_avg = df[df['is_weekend'] == 0][target_col].mean()
        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        insights.append(f"Weekend vs Weekday difference: {diff_pct:.1f}%")
    
    # Return results
    return {
        "descriptive_stats": stats_df,
        "time_series_plot": time_series_plot,
        "autocorrelation_plot": autocorr_fig,
        "seasonality_plot": seasonality_fig,
        "insights": "\n".join(insights)
    }

def train_models(data, target_col, timestamp_col, models, config):
    """Train models on the data"""
    df = data.copy()
    
    # Extract features (excluding timestamp and target)
    feature_cols = [col for col in df.columns if col not in [timestamp_col, target_col]]
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    train_size = int(len(X_scaled) * config["train_size"])
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Dictionary to store trained models and metrics
    trained_models = {}
    all_metrics = {}
    model_plots = {}
    
    # Train each model
    for model_name in models:
        if model_name == "RandomForest":
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store model
            trained_models[model_name] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols
            }
            
        elif model_name == "LinearRegression":
            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store model
            trained_models[model_name] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols
            }
            
        else:
            # Skip unsupported models
            continue
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        all_metrics[model_name] = {
            "mape": mape,
            "rmse": rmse,
            "r2": r2
        }
        
        # Create evaluation plots
        timestamps = df[timestamp_col].values[train_size:train_size+len(y_pred)]
        
        # Actual vs predicted plot
        actual_vs_pred = go.Figure()
        actual_vs_pred.add_trace(go.Scatter(
            x=timestamps, y=y_test, mode='lines', name='Actual', line=dict(color='blue')
        ))
        actual_vs_pred.add_trace(go.Scatter(
            x=timestamps, y=y_pred, mode='lines', name='Predicted', line=dict(color='red')
        ))
        actual_vs_pred.update_layout(
            title=f'{model_name} - Actual vs Predicted',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500
        )
        
        # Residuals plot
        residuals = y_test - y_pred
        residuals_plot = go.Figure()
        residuals_plot.add_trace(go.Scatter(
            x=timestamps, y=residuals, mode='lines', name='Residuals', line=dict(color='green')
        ))
        residuals_plot.update_layout(
            title=f'{model_name} - Residuals',
            xaxis_title='Time',
            yaxis_title='Residual',
            height=400
        )
        
        # Store plots
        model_plots[model_name] = {
            "actual_vs_predicted": actual_vs_pred,
            "residuals": residuals_plot
        }
    
    # Create comparison plot
    comparison_data = []
    for model_name, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    comparison_plot = px.bar(
        df_comparison,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison'
    )
    
    # Determine best model based on MAPE (lower is better)
    best_model = min(all_metrics.items(), key=lambda x: x[1]["mape"])[0]
    
    # Return results
    return {
        "models": trained_models,
        "metrics": all_metrics,
        "model_plots": model_plots,
        "comparison_plot": comparison_plot,
        "best_model": best_model
    }

def generate_forecasts(data, model_info, horizon):
    """Generate forecasts using the trained model"""
    try:
        # Extract model and metadata
        model = model_info["model"]
        scaler = model_info["scaler"]
        feature_cols = model_info["feature_cols"]
        
        # Determine the frequency of data (daily, hourly, etc.)
        if "timestamp" in data.columns:
            timestamp_col = "timestamp"
        else:
            # Try to find a timestamp column
            timestamp_candidates = [col for col in data.columns 
                                   if any(time_str in col.lower() for time_str in ['time', 'date'])]
            timestamp_col = timestamp_candidates[0] if timestamp_candidates else None
            
        if timestamp_col:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                
            # Determine frequency from timestamps
            timestamps = data[timestamp_col].sort_values()
            time_diff = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (len(timestamps) - 1) / 3600
            
            if time_diff < 1.5:  # Less than 1.5 hours
                freq = pd.Timedelta(hours=1)
            elif time_diff < 36:  # Less than 36 hours
                freq = pd.Timedelta(days=1)
            else:
                freq = pd.Timedelta(days=7)  # Weekly or longer
                
            # Get the last date
            last_date = data[timestamp_col].iloc[-1]
            
            # Generate future dates
            future_dates = [last_date + (i * freq) for i in range(1, horizon+1)]
        else:
            # If no timestamp column, use index
            last_date = pd.to_datetime(data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now()
            future_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]
        
        # Create future features
        future_features = []
        for date in future_dates:
            # Create a feature vector similar to the training data
            features = {}
            
            # Add time-based features
            if 'hour' in feature_cols:
                features['hour'] = date.hour
            if 'day' in feature_cols:
                features['day'] = date.day
            if 'dayofweek' in feature_cols:
                features['dayofweek'] = date.dayofweek
            if 'day_of_week' in feature_cols:
                features['day_of_week'] = date.dayofweek
            if 'month' in feature_cols:
                features['month'] = date.month
            if 'year' in feature_cols:
                features['year'] = date.year
            if 'is_weekend' in feature_cols:
                features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
            if 'dayofyear' in feature_cols:
                features['dayofyear'] = date.dayofyear
            
            # Add cyclical features if they were in the training data
            if 'hour_sin' in feature_cols:
                features['hour_sin'] = np.sin(2 * np.pi * date.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * date.hour / 24)
            
            if 'day_of_week_sin' in feature_cols or 'dayofweek_sin' in feature_cols:
                features['day_of_week_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
                features['day_of_week_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
                features['dayofweek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
                features['dayofweek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
            
            if 'month_sin' in feature_cols:
                features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
            
            # Extract only the features that were used in training
            feature_vector = []
            for col in feature_cols:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    # Default value for missing features
                    feature_vector.append(0)
            
            future_features.append(feature_vector)
        
        # Scale features
        X_future = scaler.transform(future_features)
        
        # Generate predictions
        forecasts = model.predict(X_future)
        
        # Create uncertainty bounds (simple approach)
        if isinstance(forecasts, np.ndarray):
            lower_bound = forecasts * 0.9  # 10% below prediction
            upper_bound = forecasts * 1.1  # 10% above prediction
        else:
            # Handle non-ndarray results (like pandas Series)
            lower_bound = np.array(forecasts) * 0.9
            upper_bound = np.array(forecasts) * 1.1
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecasts,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
    except Exception as e:
        st.error(f"Error in forecast generation: {str(e)}")
        # Return empty dataframe with expected columns
        forecast_df = pd.DataFrame(columns=['date', 'forecast', 'lower_bound', 'upper_bound'])
    
    # Create forecast plot
    historical_data = data.copy()
    if "timestamp" in historical_data.columns:
        timestamp_col = "timestamp"
    else:
        # Try to find a timestamp column
        timestamp_candidates = [col for col in historical_data.columns 
                              if any(time_str in col.lower() for time_str in ['time', 'date'])]
        timestamp_col = timestamp_candidates[0] if timestamp_candidates else historical_data.index.name
    
    target_col = [col for col in historical_data.columns if col not in feature_cols and col != timestamp_col][0]
    
    # Create plot
    fig = go.Figure()
    
    # Add historical data
    if len(historical_data) > 0:
        fig.add_trace(go.Scatter(
            x=historical_data[timestamp_col],
            y=historical_data[target_col],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.1)',
        fill='tonexty',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Load Forecast',
        xaxis_title='Date/Time',
        yaxis_title='Load',
        height=600
    )
    
    # Return results
    return {
        "forecast_data": forecast_df,
        "forecast_plot": fig
    }

# Import database utils
from utils.db_utils import DatabaseHandler

# Initialize database handler if it doesn't exist
if 'db_handler' not in st.session_state:
    st.session_state.db_handler = DatabaseHandler()

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Data Ingestion", "Exploratory Data Analysis", "Model Training", 
         "Model Evaluation", "Forecasting", "Visualization", "Forecast History"]

page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.current_page))
st.session_state.current_page = page

# Header
st.title("AI Agent-Based Load Forecasting System")

# Data Ingestion Page
if page == "Data Ingestion":
    st.header("Data Ingestion")
    
    st.markdown("""
    ## Welcome to the Load Forecasting System
    
    This application helps you forecast electrical load using various machine learning models. 
    Start by uploading your time series data or selecting from predefined datasets.
    """)
    
    data_source = st.radio(
        "Select Data Source",
        ["Upload CSV", "URL (GitHub/Kaggle)"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        # Sample datasets option
        st.subheader("Sample Datasets")
        sample_option = st.selectbox(
            "Load a sample dataset",
            ["None", "Electricity Load Data (Hourly)", "Power Consumption (15min)"]
        )
        
        # Handle sample dataset selection
        if sample_option != "None" and st.button("Load Sample"):
            if sample_option == "Electricity Load Data (Hourly)":
                # Create sample hourly electricity load data
                dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
                np.random.seed(42)
                
                # Create base load with daily and weekly patterns
                n = len(dates)
                base_load = 100 + 50 * np.sin(np.linspace(0, 4*np.pi, n))  # Daily pattern
                weekly_pattern = 20 * np.sin(np.linspace(0, 4*np.pi/7, n))  # Weekly pattern
                random_noise = np.random.normal(0, 10, n)
                
                load = base_load + weekly_pattern + random_noise
                
                # Create dataframe
                data = pd.DataFrame({
                    'timestamp': dates,
                    'load': load,
                    'temp': 20 + 5 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 2, n)
                })
                
                st.session_state.data = data
                st.success(f"Sample electricity load data loaded! Shape: {data.shape}")
                st.dataframe(data.head())
                
            elif sample_option == "Power Consumption (15min)":
                # Create sample 15-min power consumption data
                dates = pd.date_range(start='2023-01-01', end='2023-01-15', freq='15min')
                np.random.seed(43)
                
                n = len(dates)
                # More complex patterns with 15-min variations
                base_load = 500 + 200 * np.sin(np.linspace(0, 8*np.pi, n))  # Daily pattern
                weekly_pattern = 100 * np.sin(np.linspace(0, 4*np.pi/7, n))  # Weekly pattern
                random_noise = np.random.normal(0, 50, n)
                
                consumption = base_load + weekly_pattern + random_noise
                
                # Create dataframe
                data = pd.DataFrame({
                    'timestamp': dates,
                    'power_consumption': consumption,
                    'temperature': 22 + 8 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 3, n),
                    'humidity': 60 + 15 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 5, n)
                })
                
                st.session_state.data = data
                st.success(f"Sample power consumption data loaded! Shape: {data.shape}")
                st.dataframe(data.head())
        
        # Or enter URL
        st.subheader("Load from URL")
        url = st.text_input("Enter URL to CSV file")
        if url and st.button("Fetch Data from URL"):
            try:
                data = load_data_from_url(url)
                st.session_state.data = data
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    if st.session_state.data is not None:
        st.subheader("Configure Data Processing")
        
        # Data configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            timestamp_col = st.selectbox(
                "Select timestamp column",
                st.session_state.data.columns.tolist()
            )
            
            target_col = st.selectbox(
                "Select target column (load values)",
                [col for col in st.session_state.data.columns if col != timestamp_col]
            )
        
        with col2:
            freq = st.selectbox(
                "Select data frequency",
                ["H", "D", "W", "M"],
                format_func=lambda x: {
                    "H": "Hourly", 
                    "D": "Daily", 
                    "W": "Weekly", 
                    "M": "Monthly"
                }[x]
            )
            
            feature_cols = st.multiselect(
                "Select additional feature columns (optional)",
                [col for col in st.session_state.data.columns 
                 if col not in [timestamp_col, target_col]]
            )
        
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                try:
                    processed_data = process_data(
                        data=st.session_state.data,
                        timestamp_col=timestamp_col,
                        target_col=target_col,
                        freq=freq
                    )
                    
                    st.session_state.processed_data = processed_data
                    st.session_state.data_config = {
                        "timestamp_col": timestamp_col,
                        "target_col": target_col,
                        "freq": freq,
                        "feature_cols": feature_cols
                    }
                    
                    st.success("Data processed successfully!")
                    st.dataframe(processed_data.head())
                    
                    # Auto navigate to EDA
                    st.session_state.current_page = "Exploratory Data Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing data: {e}")

# Exploratory Data Analysis Page
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("Please process your data first on the Data Ingestion page.")
    else:
        if st.session_state.eda_results is None or st.button("Run EDA"):
            with st.spinner("Performing exploratory data analysis..."):
                try:
                    eda_results = perform_eda(
                        data=st.session_state.processed_data,
                        target_col=st.session_state.data_config["target_col"],
                        timestamp_col=st.session_state.data_config["timestamp_col"]
                    )
                    st.session_state.eda_results = eda_results
                    st.success("EDA completed successfully!")
                except Exception as e:
                    st.error(f"Error during EDA: {e}")
        
        if st.session_state.eda_results:
            tabs = st.tabs(["Time Series", "Seasonality", "Autocorrelation", "Statistics"])
            
            with tabs[0]:  # Time Series
                st.subheader("Time Series Plot")
                st.plotly_chart(st.session_state.eda_results["time_series_plot"], use_container_width=True)
            
            with tabs[1]:  # Seasonality
                st.subheader("Seasonality Decomposition")
                st.plotly_chart(st.session_state.eda_results["seasonality_plot"], use_container_width=True)
            
            with tabs[2]:  # Autocorrelation
                st.subheader("Autocorrelation")
                st.plotly_chart(st.session_state.eda_results["autocorrelation_plot"], use_container_width=True)
            
            with tabs[3]:  # Statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(st.session_state.eda_results["descriptive_stats"])
                
                st.subheader("EDA Insights")
                st.write(st.session_state.eda_results["insights"])

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please process your data first on the Data Ingestion page.")
    else:
        st.subheader("Configure Models")
        
        # Model selection
        models_to_train = st.multiselect(
            "Select models to train",
            ["RandomForest", "LinearRegression"],
            default=["RandomForest", "LinearRegression"]
        )
        
        # Split configuration
        col1, col2 = st.columns(2)
        with col1:
            train_size = st.slider("Training data percentage", 50, 90, 80)
            test_size = 100 - train_size
            st.write(f"Test data percentage: {test_size}%")
            
        with col2:
            horizon = st.number_input("Forecast horizon (periods)", min_value=1, value=24)
            cv_folds = st.number_input("Cross-validation folds", min_value=1, max_value=10, value=3)
        
        # Hyperparameter tuning
        tuning_method = st.selectbox(
            "Hyperparameter tuning method",
            ["Grid Search", "Random Search", "None"],
            index=1
        )
        
        # Advanced config (optional)
        with st.expander("Advanced Configuration"):
            n_jobs = st.slider("Number of parallel jobs", -1, 8, -1)
            random_state = st.number_input("Random state", value=42)
            max_trials = st.number_input("Maximum tuning trials (for Random Search)", 
                                        min_value=5, max_value=100, value=10)
        
        if st.button("Train Models"):
            if not models_to_train:
                st.error("Please select at least one model to train.")
            else:
                with st.spinner("Training models... This may take a while."):
                    try:
                        # Configure training parameters
                        train_config = {
                            "train_size": train_size/100,
                            "horizon": horizon,
                            "cv_folds": cv_folds,
                            "tuning_method": tuning_method.lower().replace(" ", "_"),
                            "n_jobs": n_jobs,
                            "random_state": random_state,
                            "max_trials": max_trials
                        }
                        
                        # Train models
                        trained_models = train_models(
                            data=st.session_state.processed_data,
                            target_col=st.session_state.data_config["target_col"],
                            timestamp_col=st.session_state.data_config["timestamp_col"],
                            models=models_to_train,
                            config=train_config
                        )
                        
                        st.session_state.trained_models = trained_models
                        st.success(f"Successfully trained {len(models_to_train)} models!")
                        
                        # Auto navigate to Model Evaluation
                        st.session_state.current_page = "Model Evaluation"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during model training: {e}")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("Please train models first on the Model Training page.")
    else:
        # Show evaluation metrics
        st.subheader("Model Performance Metrics")
        
        metrics_df = pd.DataFrame(
            {model: metrics for model, metrics in 
             st.session_state.trained_models["metrics"].items()}
        ).T
        
        st.dataframe(metrics_df)
        
        # Show evaluation plots
        st.subheader("Performance Comparison")
        st.plotly_chart(st.session_state.trained_models["comparison_plot"], use_container_width=True)
        
        # Model selection
        st.subheader("Select Best Model")
        
        available_models = list(st.session_state.trained_models["metrics"].keys())
        
        recommended_model = st.session_state.trained_models["best_model"]
        
        selected_model = st.selectbox(
            "Choose the model to use for forecasting",
            available_models,
            index=available_models.index(recommended_model) if recommended_model in available_models else 0,
            help="The recommended model is highlighted based on MAPE and R-squared performance."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_model} - Actual vs Predicted")
            st.plotly_chart(st.session_state.trained_models["model_plots"][selected_model]["actual_vs_predicted"], 
                           use_container_width=True)
        
        with col2:
            st.subheader(f"{selected_model} - Residuals")
            st.plotly_chart(st.session_state.trained_models["model_plots"][selected_model]["residuals"], 
                           use_container_width=True)
        
        if st.button("Use Selected Model for Forecasting"):
            st.session_state.best_model = selected_model
            st.success(f"Selected {selected_model} for forecasting!")
            
            # Auto navigate to Forecasting
            st.session_state.current_page = "Forecasting"
            st.rerun()

# Forecasting Page
elif page == "Forecasting":
    st.header("Forecasting")
    
    if st.session_state.best_model is None:
        st.warning("Please select a model first on the Model Evaluation page.")
    else:
        st.subheader(f"Generate Forecasts Using {st.session_state.best_model}")
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.number_input(
                "Forecast horizon (periods)",
                min_value=1,
                value=24
            )
            
            confidence_interval = st.slider(
                "Confidence interval (%)",
                min_value=50,
                max_value=99,
                value=95,
                step=5
            )
        
        with col2:
            include_history = st.checkbox("Include historical data in plot", value=True)
            
            if st.session_state.best_model in ["Prophet", "SARIMA", "ARIMA"]:
                seasonality_mode = st.selectbox(
                    "Seasonality mode",
                    ["additive", "multiplicative"],
                    index=0
                )
            else:
                seasonality_mode = "additive"  # Default for models that don't use this parameter
        
        if st.button("Generate Forecasts"):
            with st.spinner("Generating forecasts..."):
                try:
                    # Configure forecast parameters
                    forecast_config = {
                        "horizon": forecast_horizon,
                        "confidence_interval": confidence_interval/100,
                        "include_history": include_history,
                        "seasonality_mode": seasonality_mode
                    }
                    
                    # Generate forecasts
                    forecasts = generate_forecasts(
                        data=st.session_state.processed_data,
                        model_info=st.session_state.trained_models["models"][st.session_state.best_model],
                        horizon=forecast_horizon
                    )
                    
                    st.session_state.forecasts = forecasts
                    st.success(f"Successfully generated forecasts for {forecast_horizon} periods!")
                    
                    # Save forecast to database
                    try:
                        # Save model first
                        model_metrics = st.session_state.trained_models["metrics"].get(st.session_state.best_model, {})
                        model_params = {"name": st.session_state.best_model}
                        
                        model_id = st.session_state.db_handler.save_model(
                            name=f"{st.session_state.best_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            model_type=st.session_state.best_model,
                            metrics=model_metrics,
                            parameters=model_params
                        )
                        
                        if model_id:
                            # Then save forecast
                            forecast_id = st.session_state.db_handler.save_forecast(
                                model_id=model_id,
                                forecast_data=forecasts["forecast_data"],
                                horizon=forecast_horizon
                            )
                            
                            if forecast_id:
                                st.success(f"Forecast saved to database with ID: {forecast_id}")
                    except Exception as e:
                        st.warning(f"Could not save forecast to database: {e}")
                    
                    # Auto navigate to Visualization
                    st.session_state.current_page = "Visualization"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during forecasting: {e}")

# Visualization Page
elif page == "Visualization":
    st.header("Visualization")
    
    if st.session_state.forecasts is None:
        st.warning("Please generate forecasts first on the Forecasting page.")
    else:
        st.subheader("Forecast Results")
        
        # Display forecast plot
        st.plotly_chart(st.session_state.forecasts["forecast_plot"], use_container_width=True)
        
        # Display forecast data
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecasts["forecast_data"])
        
        # Option to save dataset
        with st.expander("Save Dataset Information"):
            dataset_name = st.text_input("Dataset Name", value="Load Dataset")
            dataset_description = st.text_area("Dataset Description", value="Time series data for load forecasting")
            
            if st.button("Save Dataset to Database"):
                try:
                    dataset_id = st.session_state.db_handler.save_dataset(
                        name=dataset_name,
                        description=dataset_description,
                        data=st.session_state.processed_data
                    )
                    
                    if dataset_id:
                        st.success(f"Dataset saved with ID: {dataset_id}")
                    else:
                        st.error("Failed to save dataset to database")
                except Exception as e:
                    st.error(f"Error saving dataset: {e}")
        
        # Allow download of forecast results
        csv = st.session_state.forecasts["forecast_data"].to_csv(index=False)
        st.download_button(
            label="Download Forecast Data as CSV",
            data=csv,
            file_name="load_forecast_results.csv",
            mime="text/csv"
        )
        
        # Show forecast performance metrics if available
        if "performance_metrics" in st.session_state.forecasts:
            st.subheader("Forecast Performance Metrics")
            st.dataframe(st.session_state.forecasts["performance_metrics"])
        
        # Offer to download a report
        st.subheader("Forecast Report")
        
        if st.button("Generate Forecast Report"):
            with st.spinner("Generating report..."):
                try:
                    # Simple report generation
                    report = f"""
                    <h1>Load Forecasting Report</h1>
                    <h2>Model: {st.session_state.best_model}</h2>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <h3>Forecast Summary</h3>
                    <p>Horizon: {len(st.session_state.forecasts['forecast_data'])} periods</p>
                    """
                    
                    st.session_state.report = report
                    st.success("Report generated successfully!")
                    
                    # Provide download link
                    st.download_button(
                        label="Download Full Report (HTML)",
                        data=report,
                        file_name="load_forecasting_report.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {e}")

# Forecast History Page
elif page == "Forecast History":
    st.header("Forecast History")
    st.markdown("""
    This page shows your saved forecasts from the database. You can view past forecasting results,
    compare performance across different models, and track your forecasting accuracy over time.
    """)
    
    # Get saved forecasts from database
    try:
        forecasts_df = st.session_state.db_handler.get_forecast_list()
        
        if forecasts_df is not None and not forecasts_df.empty:
            st.subheader("Saved Forecasts")
            st.dataframe(forecasts_df)
            
            # Select a forecast to view
            if "selected_forecast_id" not in st.session_state:
                st.session_state.selected_forecast_id = None
                
            forecast_ids = forecasts_df["id"].tolist()
            selected_id = st.selectbox(
                "Select a forecast to view details",
                forecast_ids,
                format_func=lambda x: f"Forecast {x}: {forecasts_df[forecasts_df['id'] == x]['model_type'].values[0]} - {forecasts_df[forecasts_df['id'] == x]['created_at'].values[0]}"
            )
            
            if st.button("View Forecast Details") or st.session_state.selected_forecast_id == selected_id:
                st.session_state.selected_forecast_id = selected_id
                
                # Get forecast details
                forecast = st.session_state.db_handler.get_forecast_by_id(selected_id)
                
                if forecast:
                    # Display forecast info
                    st.subheader(f"Forecast #{forecast['id']} Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Model Type:** {forecast['model_type']}")
                        st.markdown(f"**Forecast Date:** {forecast['forecast_date']}")
                    
                    with col2:
                        st.markdown(f"**Horizon:** {forecast['horizon']} periods")
                        st.markdown(f"**Created At:** {forecast['created_at']}")
                    
                    # Model metrics
                    st.subheader("Model Performance")
                    if forecast['model_metrics']:
                        metrics_df = pd.DataFrame({
                            'Metric': list(forecast['model_metrics'].keys()),
                            'Value': list(forecast['model_metrics'].values())
                        })
                        st.dataframe(metrics_df)
                    else:
                        st.info("No model metrics available")
                    
                    # Forecast values
                    st.subheader("Forecast Values")
                    if forecast['forecast_values']:
                        forecast_df = pd.DataFrame(forecast['forecast_values'])
                        st.dataframe(forecast_df)
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(forecast_df['date']),
                            y=forecast_df['forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add confidence intervals if available
                        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                            fig.add_trace(go.Scatter(
                                x=pd.to_datetime(forecast_df['date']),
                                y=forecast_df['upper_bound'],
                                mode='lines',
                                name='Upper Bound',
                                line=dict(width=0),
                                showlegend=True
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=pd.to_datetime(forecast_df['date']),
                                y=forecast_df['lower_bound'],
                                mode='lines',
                                name='Lower Bound',
                                line=dict(width=0),
                                fillcolor='rgba(255, 0, 0, 0.1)',
                                fill='tonexty',
                                showlegend=True
                            ))
                        
                        fig.update_layout(
                            title='Saved Load Forecast',
                            xaxis_title='Date/Time',
                            yaxis_title='Load',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No forecast values available")
                else:
                    st.error("Could not retrieve forecast details")
        else:
            st.info("No forecasts saved yet. Generate and save forecasts to see them here.")
            
        # Get saved models
        models_df = st.session_state.db_handler.get_model_list()
        if models_df is not None and not models_df.empty:
            with st.expander("View Saved Models"):
                st.dataframe(models_df)
        
        # Get saved datasets
        datasets_df = st.session_state.db_handler.get_dataset_list()
        if datasets_df is not None and not datasets_df.empty:
            with st.expander("View Saved Datasets"):
                st.dataframe(datasets_df)
    
    except Exception as e:
        st.error(f"Error accessing forecast history: {e}")

# Add footer
st.markdown("---")
st.markdown("AI Agent-Based Load Forecasting System")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass
