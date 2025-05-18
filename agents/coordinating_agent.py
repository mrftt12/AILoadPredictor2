import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple

from agents.data_processing_agent import DataProcessingAgent
from agents.eda_agent import EDAAgent
from agents.modeling_agent import ModelingAgent
from agents.model_verification_agent import ModelVerificationAgent
from agents.model_deployment_agent import ModelDeploymentAgent
from agents.forecasting_agent import ForecastingAgent
from agents.visualization_agent import VisualizationAgent

class CoordinatingAgent:
    """
    Coordinates the entire forecasting workflow by orchestrating all specialized agents.
    Serves as the central hub for agent communication and data flow.
    """
    
    def __init__(self):
        """Initialize all specialized agents."""
        self.data_processing_agent = DataProcessingAgent()
        self.eda_agent = EDAAgent()
        self.modeling_agent = ModelingAgent()
        self.model_verification_agent = ModelVerificationAgent()
        self.model_deployment_agent = ModelDeploymentAgent()
        self.forecasting_agent = ForecastingAgent()
        self.visualization_agent = VisualizationAgent()
        
        # For storing the workflow state
        self.state = {
            "data_source": None,
            "processed_data": None,
            "eda_results": None,
            "trained_models": None,
            "selected_model": None,
            "forecasts": None,
            "visuals": None
        }
    
    def fetch_data_from_url(self, url: str) -> pd.DataFrame:
        """Fetch data from a URL using the data processing agent."""
        return self.data_processing_agent.ingest_from_url(url)
    
    def process_data(self, data: pd.DataFrame, timestamp_col: str, 
                    target_col: str, freq: str, feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Process the data using the data processing agent.
        
        Args:
            data: The input dataframe
            timestamp_col: Column name containing timestamps
            target_col: Column name containing the target values (load)
            freq: Desired frequency ('H', 'D', 'W', 'M')
            feature_cols: List of additional feature columns to include
            
        Returns:
            Processed dataframe ready for analysis and modeling
        """
        processed_data = self.data_processing_agent.process(
            data=data, 
            timestamp_col=timestamp_col,
            target_col=target_col,
            freq=freq,
            feature_cols=feature_cols
        )
        
        self.state["processed_data"] = processed_data
        return processed_data
    
    def perform_eda(self, data: pd.DataFrame, target_col: str, timestamp_col: str) -> Dict[str, Any]:
        """
        Perform exploratory data analysis using the EDA agent.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            
        Returns:
            Dictionary containing EDA results, visualizations, and insights
        """
        eda_results = self.eda_agent.analyze(
            data=data,
            target_col=target_col,
            timestamp_col=timestamp_col
        )
        
        self.state["eda_results"] = eda_results
        return eda_results
    
    def train_models(self, data: pd.DataFrame, target_col: str, timestamp_col: str, 
                     models: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train selected models using the modeling agent.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            models: List of model names to train
            config: Configuration dictionary for training
            
        Returns:
            Dictionary containing trained models, metrics, and evaluation results
        """
        # Step 1: Train models
        trained_models = self.modeling_agent.train(
            data=data,
            target_col=target_col,
            timestamp_col=timestamp_col,
            models=models,
            config=config
        )
        
        # Step 2: Verify and compare models
        verification_results = self.model_verification_agent.evaluate(
            data=data,
            target_col=target_col,
            timestamp_col=timestamp_col,
            models=trained_models["models"],
            metrics=["mape", "rmse", "r2"]
        )
        
        # Combine results
        model_results = {
            "models": trained_models["models"],
            "metrics": verification_results["metrics"],
            "best_model": verification_results["best_model"],
            "comparison_plot": verification_results["comparison_plot"],
            "model_plots": verification_results["model_plots"]
        }
        
        self.state["trained_models"] = model_results
        return model_results
    
    def generate_forecasts(self, data: pd.DataFrame, model_name: str, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate forecasts using the selected model via the forecasting agent.
        
        Args:
            data: Processed dataframe
            model_name: Name of the selected model
            config: Configuration dictionary for forecasting
            
        Returns:
            Dictionary containing forecast results and visualizations
        """
        # Step 1: Deploy the model
        model = self.state["trained_models"]["models"][model_name]
        deployed_model = self.model_deployment_agent.deploy(model, model_name)
        
        # Step 2: Generate forecasts
        forecasts = self.forecasting_agent.predict(
            model=deployed_model,
            data=data,
            horizon=config["horizon"],
            conf_int=config["confidence_interval"]
        )
        
        # Step 3: Visualize forecasts
        forecast_plot = self.visualization_agent.plot_forecast(
            data=data,
            forecasts=forecasts["predictions"],
            conf_intervals=forecasts.get("conf_intervals"),
            include_history=config["include_history"]
        )
        
        # Combine results
        forecast_results = {
            "forecast_data": forecasts["predictions"],
            "forecast_plot": forecast_plot,
            "conf_intervals": forecasts.get("conf_intervals"),
            "performance_metrics": forecasts.get("metrics")
        }
        
        self.state["forecasts"] = forecast_results
        return forecast_results
    
    def generate_report(self, processed_data: pd.DataFrame, eda_results: Dict[str, Any],
                       model_results: Dict[str, Any], forecast_results: Dict[str, Any],
                       selected_model: str) -> str:
        """
        Generate a comprehensive HTML report for the forecasting process.
        
        Args:
            processed_data: The processed dataframe
            eda_results: Results from EDA
            model_results: Results from model training and evaluation
            forecast_results: Results from forecasting
            selected_model: Name of the selected model
            
        Returns:
            HTML string containing the complete report
        """
        report_html = self.visualization_agent.generate_report(
            data=processed_data,
            eda_results=eda_results,
            model_results=model_results,
            forecast_results=forecast_results,
            selected_model=selected_model
        )
        
        return report_html
