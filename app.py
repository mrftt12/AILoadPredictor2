import streamlit as st
import os
import pandas as pd
import numpy as np
from agents.coordinating_agent import CoordinatingAgent

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

# Initialize the coordinating agent
if 'coordinating_agent' not in st.session_state:
    st.session_state.coordinating_agent = CoordinatingAgent()

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
        url = st.text_input("Enter URL to CSV file")
        if url and st.button("Fetch Data"):
            try:
                data = st.session_state.coordinating_agent.fetch_data_from_url(url)
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
                    processed_data = st.session_state.coordinating_agent.process_data(
                        data=st.session_state.data,
                        timestamp_col=timestamp_col,
                        target_col=target_col,
                        freq=freq,
                        feature_cols=feature_cols
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
                    eda_results = st.session_state.coordinating_agent.perform_eda(
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
            ["LSTM", "LightGBM", "Prophet", "ARIMA", "SARIMA"],
            default=["LSTM", "LightGBM", "Prophet"]
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
                        trained_models = st.session_state.coordinating_agent.train_models(
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
                    forecasts = st.session_state.coordinating_agent.generate_forecasts(
                        data=st.session_state.processed_data,
                        model_name=st.session_state.best_model,
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
                    report = st.session_state.coordinating_agent.generate_report(
                        processed_data=st.session_state.processed_data,
                        eda_results=st.session_state.eda_results,
                        model_results=st.session_state.trained_models,
                        forecast_results=st.session_state.forecasts,
                        selected_model=st.session_state.best_model
                    )
                    
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

# Add footer
st.markdown("---")
st.markdown("AI Agent-Based Load Forecasting System")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass
