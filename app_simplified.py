import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.db_manager import load_data_from_db, store_sample_data
from utils.data_utils import validate_data_for_forecasting
from utils.create_sample_data import create_hourly_load_dataset, create_daily_load_dataset

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

# Helper functions for the simplified app
def perform_eda(data, target_col, timestamp_col):
    """Perform simplified EDA without agent dependencies"""
    # Make a copy
    df = data.copy()
    
    # Set timestamp as index for time series analysis
    df_ts = df.set_index(timestamp_col)
    
    # Calculate basic statistics
    stats = df[target_col].describe()
    stats_df = pd.DataFrame({
        'Statistic': stats.index.tolist(),
        'Value': stats.values.tolist()
    })
    
    # Create time series plot
    time_series_fig = px.line(
        df, 
        x=timestamp_col, 
        y=target_col,
        title=f'Time Series Plot of {target_col}',
        labels={timestamp_col: 'Date/Time', target_col: 'Load'}
    )
    
    # Add daily/weekly/monthly patterns based on data frequency
    if len(df) > 24:  # Enough data for pattern analysis
        # Determine frequency by checking time differences
        time_diffs = pd.Series(pd.to_datetime(df[timestamp_col])).diff().dropna()
        median_diff = time_diffs.median().total_seconds()
        
        # Add insights based on frequency
        insights = []
        
        if median_diff < 3600:  # Less than an hour - minutes data
            insights.append("Data appears to be in minutes frequency")
        elif median_diff < 86400:  # Less than a day - hourly data
            # Calculate hourly patterns
            df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
            hourly_avg = df.groupby('hour')[target_col].mean()
            peak_hour = hourly_avg.idxmax()
            low_hour = hourly_avg.idxmin()
            insights.append(f"Peak load hour: {peak_hour}:00, low load hour: {low_hour}:00")
            
            # Daily patterns
            df['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
            daily_avg = df.groupby('day_of_week')[target_col].mean()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day = days[daily_avg.idxmax()]
            low_day = days[daily_avg.idxmin()]
            insights.append(f"Peak load day: {peak_day}, low load day: {low_day}")
            
            # Weekend vs weekday
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            weekend_avg = df[df['is_weekend'] == 1][target_col].mean()
            weekday_avg = df[df['is_weekend'] == 0][target_col].mean()
            
            if weekend_avg < weekday_avg:
                insights.append(f"Weekend loads are {(weekday_avg-weekend_avg)/weekday_avg*100:.2f}% lower than weekday loads")
            else:
                insights.append(f"Weekend loads are {(weekend_avg-weekday_avg)/weekday_avg*100:.2f}% higher than weekday loads")
            
            # Hourly pattern plot
            hourly_pattern_fig = px.bar(
                x=hourly_avg.index, 
                y=hourly_avg.values,
                title="Average Load by Hour of Day",
                labels={"x": "Hour of Day", "y": f"Average {target_col}"}
            )
            
            # Daily pattern plot
            daily_pattern_fig = px.bar(
                x=[days[i] for i in daily_avg.index], 
                y=daily_avg.values,
                title="Average Load by Day of Week",
                labels={"x": "Day of Week", "y": f"Average {target_col}"}
            )
        else:  # Daily or lower frequency
            # Monthly patterns
            df['month'] = pd.to_datetime(df[timestamp_col]).dt.month
            monthly_avg = df.groupby('month')[target_col].mean()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            peak_month = months[monthly_avg.idxmax()-1]
            low_month = months[monthly_avg.idxmin()-1]
            insights.append(f"Peak load month: {peak_month}, low load month: {low_month}")
            
            # Monthly pattern plot
            monthly_pattern_fig = px.bar(
                x=[months[i-1] for i in monthly_avg.index], 
                y=monthly_avg.values,
                title="Average Load by Month",
                labels={"x": "Month", "y": f"Average {target_col}"}
            )
    else:
        insights = ["Not enough data points for detailed pattern analysis"]
        hourly_pattern_fig = None
        daily_pattern_fig = None
        monthly_pattern_fig = None
    
    # Combine all results
    eda_results = {
        "descriptive_stats": stats_df,
        "time_series_plot": time_series_fig,
        "hourly_pattern_plot": hourly_pattern_fig if 'hourly_pattern_fig' in locals() else None,
        "daily_pattern_plot": daily_pattern_fig if 'daily_pattern_fig' in locals() else None,
        "monthly_pattern_plot": monthly_pattern_fig if 'monthly_pattern_fig' in locals() else None,
        "insights": "\n".join(insights)
    }
    
    return eda_results

def train_simple_models(data, target_col, timestamp_col, models, config):
    """Train simplified models without MLflow and TensorFlow dependencies"""
    # Make a copy
    df = data.copy()
    
    # Prepare features
    # Add time features if they don't exist
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = pd.to_datetime(df[timestamp_col]).dt.month
    if 'day' not in df.columns:
        df['day'] = pd.to_datetime(df[timestamp_col]).dt.day
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [timestamp_col, target_col]]
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data
    train_size = int(len(df) * config['train_size'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trained_models = {}
    metrics = {}
    model_plots = {}
    
    # Train Random Forest
    if 'RandomForest' in models:
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        metrics['RandomForest'] = {
            'mape': mape,
            'rmse': rmse,
            'r2': r2
        }
        
        # Create plots
        actual_vs_pred_fig = go.Figure()
        
        # Add traces
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        # Update layout
        actual_vs_pred_fig.update_layout(
            title=f'Random Forest - Actual vs Predicted',
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title='Legend',
            template='plotly_white'
        )
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create residuals plot
        residuals_fig = go.Figure()
        residuals_fig.add_trace(go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='green')
        ))
        
        residuals_fig.add_shape(type='line',
            x0=0, y0=0,
            x1=len(residuals), y1=0,
            line=dict(color='black', dash='dash')
        )
        
        residuals_fig.update_layout(
            title=f'Random Forest - Residuals',
            xaxis_title='Time',
            yaxis_title='Residual',
            template='plotly_white'
        )
        
        # Store plots
        model_plots['RandomForest'] = {
            'actual_vs_predicted': actual_vs_pred_fig,
            'residuals': residuals_fig
        }
        
        # Store model
        trained_models['RandomForest'] = {
            'model': rf,
            'scaler': scaler,
            'feature_names': feature_cols
        }
    
    # Train Linear Regression
    if 'LinearRegression' in models:
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = lr.predict(X_test_scaled)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        metrics['LinearRegression'] = {
            'mape': mape,
            'rmse': rmse,
            'r2': r2
        }
        
        # Create plots
        actual_vs_pred_fig = go.Figure()
        
        # Add traces
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        actual_vs_pred_fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        # Update layout
        actual_vs_pred_fig.update_layout(
            title=f'Linear Regression - Actual vs Predicted',
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title='Legend',
            template='plotly_white'
        )
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create residuals plot
        residuals_fig = go.Figure()
        residuals_fig.add_trace(go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='green')
        ))
        
        residuals_fig.add_shape(type='line',
            x0=0, y0=0,
            x1=len(residuals), y1=0,
            line=dict(color='black', dash='dash')
        )
        
        residuals_fig.update_layout(
            title=f'Linear Regression - Residuals',
            xaxis_title='Time',
            yaxis_title='Residual',
            template='plotly_white'
        )
        
        # Store plots
        model_plots['LinearRegression'] = {
            'actual_vs_predicted': actual_vs_pred_fig,
            'residuals': residuals_fig
        }
        
        # Store model
        trained_models['LinearRegression'] = {
            'model': lr,
            'scaler': scaler,
            'feature_names': feature_cols
        }
    
    # Create comparison plot
    comparison_data = []
    for model_name, model_metrics in metrics.items():
        for metric_name, metric_value in model_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': metric_value
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_filtered = df_comparison[df_comparison['Metric'].isin(['mape', 'r2'])]
    
    comparison_plot = px.bar(
        df_filtered,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Value': 'Metric Value', 'Model': 'Model Name'},
        height=500
    )
    
    # Determine best model
    best_model = min(metrics.items(), key=lambda x: x[1]['mape'])[0]
    
    # Return results
    return {
        'models': trained_models,
        'metrics': metrics,
        'best_model': best_model,
        'comparison_plot': comparison_plot,
        'model_plots': model_plots
    }

def generate_forecasts(data, model_name, model_info, config):
    """Generate forecasts using the selected model"""
    # Get model and scaler
    model = model_info['models'][model_name]['model']
    scaler = model_info['models'][model_name]['scaler']
    feature_names = model_info['models'][model_name]['feature_names']
    
    # Prepare data for forecasting
    # Make a copy
    df = data.copy()
    
    # Get the last timestamp
    if isinstance(df.index, pd.DatetimeIndex):
        last_timestamp = df.index[-1]
    else:
        timestamp_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
        last_timestamp = pd.to_datetime(df[timestamp_col].iloc[-1])
    
    # Generate future timestamps
    horizon = config['horizon']
    future_timestamps = []
    
    # Determine frequency
    if len(df) > 1:
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index[1] - df.index[0]
        else:
            time_diff = pd.to_datetime(df[timestamp_col].iloc[1]) - pd.to_datetime(df[timestamp_col].iloc[0])
        
        for i in range(1, horizon + 1):
            future_timestamps.append(last_timestamp + i * time_diff)
    else:
        # Default to hourly if we can't determine
        for i in range(1, horizon + 1):
            future_timestamps.append(last_timestamp + pd.Timedelta(hours=i))
    
    # Create future features
    future_features = []
    for timestamp in future_timestamps:
        # Extract time components
        hour = timestamp.hour
        day = timestamp.day
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        
        # Create cyclical features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Combine features
        feature_values = {
            'hour': hour,
            'day': day,
            'day_of_week': day_of_week,
            'month': month,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_week_sin': day_of_week_sin,
            'day_of_week_cos': day_of_week_cos,
            'month_sin': month_sin,
            'month_cos': month_cos
        }
        
        # Create feature array in the same order as training
        feature_array = []
        for feature in feature_names:
            if feature in feature_values:
                feature_array.append(feature_values[feature])
            else:
                # For unknown features, use the last value from the dataset
                if feature in df.columns:
                    feature_array.append(df[feature].iloc[-1])
                else:
                    feature_array.append(0)  # Default value
        
        future_features.append(feature_array)
    
    # Convert to numpy array
    future_features = np.array(future_features)
    
    # Scale features
    future_features_scaled = scaler.transform(future_features)
    
    # Generate predictions
    predictions = model.predict(future_features_scaled)
    
    # Create dataframe
    forecast_df = pd.DataFrame({
        'date': future_timestamps,
        'forecast': predictions
    })
    
    # Add confidence intervals
    # For simplicity, use a fixed percentage of the prediction
    confidence = config['confidence_interval']
    error_margin = predictions * (1 - confidence)
    
    forecast_df['lower_bound'] = forecast_df['forecast'] - error_margin
    forecast_df['upper_bound'] = forecast_df['forecast'] + error_margin
    
    # Create forecast plot
    if config['include_history']:
        # Get historical data
        if isinstance(df.index, pd.DatetimeIndex):
            hist_dates = df.index
            value_col = df.columns[0]  # Assume first column is the target
            hist_values = df[value_col]
        else:
            timestamp_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
            target_col = [col for col in df.columns if col != timestamp_col][0]  # Simplified
            hist_dates = pd.to_datetime(df[timestamp_col])
            hist_values = df[target_col]
        
        # Create plot with history
        forecast_plot = go.Figure()
        
        # Add historical data
        forecast_plot.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name=f'Upper Bound ({confidence*100:.0f}%)',
            line=dict(width=0),
            showlegend=True
        ))
        
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name=f'Lower Bound ({confidence*100:.0f}%)',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            showlegend=True
        ))
    else:
        # Create plot with just the forecast
        forecast_plot = go.Figure()
        
        # Add forecast
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name=f'Upper Bound ({confidence*100:.0f}%)',
            line=dict(width=0),
            showlegend=True
        ))
        
        forecast_plot.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name=f'Lower Bound ({confidence*100:.0f}%)',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            showlegend=True
        ))
    
    # Update layout
    forecast_plot.update_layout(
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
    
    # Return results
    return {
        'forecast_data': forecast_df,
        'forecast_plot': forecast_plot,
        'conf_intervals': {
            'lower': forecast_df['lower_bound'].values,
            'upper': forecast_df['upper_bound'].values,
            'level': confidence
        }
    }

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Data Ingestion", "Exploratory Data Analysis", "Model Training", 
         "Model Evaluation", "Forecasting", "Visualization"]

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
        ["Upload CSV", "URL (GitHub/Kaggle)", "Sample Data", "Database"]
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
    elif data_source == "URL (GitHub/Kaggle)":
        url = st.text_input("Enter URL to CSV file")
        if url and st.button("Fetch Data"):
            try:
                data = pd.read_csv(url)
                st.session_state.data = data
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    elif data_source == "Sample Data":
        sample_type = st.selectbox(
            "Select sample data type",
            ["Hourly Load Data", "Daily Load Data"]
        )
        
        if st.button("Load Sample Data"):
            try:
                if sample_type == "Hourly Load Data":
                    # Check if the file exists
                    if os.path.exists("data/sample_hourly_load.csv"):
                        data = pd.read_csv("data/sample_hourly_load.csv")
                    else:
                        # Create the data
                        data = create_hourly_load_dataset()
                elif sample_type == "Daily Load Data":
                    # Check if the file exists
                    if os.path.exists("data/sample_daily_load.csv"):
                        data = pd.read_csv("data/sample_daily_load.csv")
                    else:
                        # Create the data
                        data = create_daily_load_dataset()
                
                st.session_state.data = data
                st.success(f"Sample data loaded successfully! Shape: {data.shape}")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
    else:  # Database
        if st.button("Load Data from Database"):
            try:
                data = load_data_from_db()
                if len(data) > 0:
                    st.session_state.data = data
                    st.success(f"Data loaded successfully from database! Shape: {data.shape}")
                    st.dataframe(data.head())
                else:
                    st.warning("No data found in the database. Please load sample data first.")
            except Exception as e:
                st.error(f"Error loading data from database: {e}")
    
    if st.session_state.data is not None:
        st.subheader("Configure Data Processing")
        
        # Data validation
        is_valid, error_msg = validate_data_for_forecasting(st.session_state.data)
        if not is_valid:
            st.warning(f"Data validation issue: {error_msg}")
        
        # Data configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            # Find potential timestamp columns
            timestamp_cols = [col for col in st.session_state.data.columns 
                             if 'date' in col.lower() or 'time' in col.lower()]
            if not timestamp_cols:
                timestamp_cols = st.session_state.data.columns.tolist()
            
            timestamp_col = st.selectbox(
                "Select timestamp column",
                timestamp_cols
            )
            
            # Find potential target columns (numeric columns that aren't the timestamp)
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in numeric_cols if col != timestamp_col and 
                          ('load' in col.lower() or 'demand' in col.lower() or 'consumption' in col.lower())]
            if not target_cols:
                target_cols = [col for col in numeric_cols if col != timestamp_col]
            
            target_col = st.selectbox(
                "Select target column (load values)",
                target_cols
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
                    # Convert timestamp to datetime
                    df = st.session_state.data.copy()
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    
                    # Sort by timestamp
                    df = df.sort_values(by=timestamp_col)
                    
                    # Handle missing values in target column
                    if df[target_col].isna().sum() > 0:
                        df[target_col] = df[target_col].interpolate(method='linear')
                    
                    # Generate time features
                    df['hour'] = df[timestamp_col].dt.hour
                    df['day'] = df[timestamp_col].dt.day
                    df['day_of_week'] = df[timestamp_col].dt.dayofweek
                    df['month'] = df[timestamp_col].dt.month
                    df['year'] = df[timestamp_col].dt.year
                    
                    st.session_state.processed_data = df
                    st.session_state.data_config = {
                        "timestamp_col": timestamp_col,
                        "target_col": target_col,
                        "freq": freq,
                        "feature_cols": feature_cols
                    }
                    
                    st.success("Data processed successfully!")
                    st.dataframe(df.head())
                    
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
            tabs = st.tabs(["Time Series", "Patterns", "Statistics", "Insights"])
            
            with tabs[0]:  # Time Series
                st.subheader("Time Series Plot")
                st.plotly_chart(st.session_state.eda_results["time_series_plot"], use_container_width=True)
            
            with tabs[1]:  # Patterns
                st.subheader("Load Patterns")
                
                # Display hourly patterns if available
                if st.session_state.eda_results["hourly_pattern_plot"] is not None:
                    st.plotly_chart(st.session_state.eda_results["hourly_pattern_plot"], use_container_width=True)
                
                # Display daily patterns if available
                if st.session_state.eda_results["daily_pattern_plot"] is not None:
                    st.plotly_chart(st.session_state.eda_results["daily_pattern_plot"], use_container_width=True)
                
                # Display monthly patterns if available
                if st.session_state.eda_results["monthly_pattern_plot"] is not None:
                    st.plotly_chart(st.session_state.eda_results["monthly_pattern_plot"], use_container_width=True)
            
            with tabs[2]:  # Statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(st.session_state.eda_results["descriptive_stats"])
            
            with tabs[3]:  # Insights
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
        
        # Advanced config (optional)
        with st.expander("Advanced Configuration"):
            n_jobs = st.slider("Number of parallel jobs", -1, 8, -1)
            random_state = st.number_input("Random state", value=42)
        
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
                            "n_jobs": n_jobs,
                            "random_state": random_state
                        }
                        
                        # Train models
                        trained_models = train_simple_models(
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
            
            seasonality_mode = st.selectbox(
                "Seasonality mode",
                ["additive", "multiplicative"],
                index=0
            )
        
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
                        model_name=st.session_state.best_model,
                        model_info=st.session_state.trained_models,
                        config=forecast_config
                    )
                    
                    st.session_state.forecasts = forecasts
                    st.success(f"Successfully generated forecasts for {forecast_horizon} periods!")
                    
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
        
        # Download forecast data
        csv = st.session_state.forecasts["forecast_data"].to_csv(index=False)
        st.download_button(
            label="Download Forecast Data as CSV",
            data=csv,
            file_name="load_forecast.csv",
            mime="text/csv"
        )
        
        # Generate report
        st.subheader("Generate Report")
        
        if st.button("Generate Forecast Report"):
            try:
                report_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Load Forecasting Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #2c3e50; }}
                        .section {{ margin-bottom: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Load Forecasting Report</h1>
                    
                    <div class="section">
                        <h2>Forecast Summary</h2>
                        <p>Model: {st.session_state.best_model}</p>
                        <p>Forecast Horizon: {len(st.session_state.forecasts["forecast_data"])} periods</p>
                        <p>Confidence Interval: {st.session_state.forecasts["conf_intervals"]["level"]*100}%</p>
                    </div>
                    
                    <div class="section">
                        <h2>Model Performance</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>MAPE (%)</td>
                                <td>{st.session_state.trained_models["metrics"][st.session_state.best_model]["mape"]:.2f}</td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>{st.session_state.trained_models["metrics"][st.session_state.best_model]["rmse"]:.2f}</td>
                            </tr>
                            <tr>
                                <td>R²</td>
                                <td>{st.session_state.trained_models["metrics"][st.session_state.best_model]["r2"]:.4f}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Forecast Data</h2>
                        <table>
                            <tr>
                                <th>Date</th>
                                <th>Forecast</th>
                                <th>Lower Bound</th>
                                <th>Upper Bound</th>
                            </tr>
                            {"".join([f"<tr><td>{row['date']}</td><td>{row['forecast']:.2f}</td><td>{row['lower_bound']:.2f}</td><td>{row['upper_bound']:.2f}</td></tr>" for _, row in st.session_state.forecasts["forecast_data"].iterrows()])}
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Conclusion</h2>
                        <p>The forecasting model predicts the load pattern will continue with the same seasonality and trends observed in the historical data.</p>
                        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                </body>
                </html>
                """
                
                # Provide download link
                st.download_button(
                    label="Download Report",
                    data=report_html,
                    file_name="load_forecast_report.html",
                    mime="text/html"
                )
                
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")