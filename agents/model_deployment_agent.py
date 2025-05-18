import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pickle
import mlflow
import os

class ModelDeploymentAgent:
    """
    Manages the packaging and serving of the selected model.
    Responsible for retrieving the best model, preparing it for inference,
    and setting up local model serving.
    """
    
    def __init__(self):
        """Initialize the model deployment agent."""
        # Set up MLflow tracking URI if not configured
        if mlflow.get_tracking_uri() is None:
            mlflow.set_tracking_uri("mlruns")
        
        # Create a model directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
    
    def deploy(self, model_info: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Deploy a model for inference.
        
        Args:
            model_info: Dictionary containing model and metadata
            model_name: Name of the model
            
        Returns:
            Dictionary containing the deployed model and metadata
        """
        # Extract the model and necessary metadata
        model = model_info.get("model")
        
        if model is None:
            raise ValueError(f"No model found in model_info for {model_name}")
        
        # Determine deployment method based on model type
        if model_name == "LSTM":
            deployed_model = self._deploy_lstm(model_info)
        elif model_name == "LightGBM":
            deployed_model = self._deploy_lightgbm(model_info)
        elif model_name == "Prophet":
            deployed_model = self._deploy_prophet(model_info)
        elif model_name in ["ARIMA", "SARIMA"]:
            deployed_model = self._deploy_statsmodels(model_info, model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        # Add model name to deployed model info
        deployed_model["model_name"] = model_name
        
        return deployed_model
    
    def _deploy_lstm(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy an LSTM model.
        
        Args:
            model_info: Dictionary containing model and metadata
            
        Returns:
            Dictionary with deployed model and metadata
        """
        model = model_info["model"]
        scaler = model_info["scaler"]
        feature_names = model_info["feature_names"]
        timesteps = model_info.get("timesteps", 24)
        
        # Save model locally
        model_path = os.path.join("models", "lstm_model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the TensorFlow model
        model.save(model_path)
        
        # Save the scaler and metadata
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        
        with open(os.path.join(model_path, "metadata.pkl"), "wb") as f:
            metadata = {
                "feature_names": feature_names,
                "timesteps": timesteps
            }
            pickle.dump(metadata, f)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "timesteps": timesteps,
            "model_path": model_path
        }
    
    def _deploy_lightgbm(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a LightGBM model.
        
        Args:
            model_info: Dictionary containing model and metadata
            
        Returns:
            Dictionary with deployed model and metadata
        """
        model = model_info["model"]
        scaler = model_info["scaler"]
        feature_names = model_info["feature_names"]
        
        # Save model locally
        model_path = os.path.join("models", "lightgbm_model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the LightGBM model
        model_file = os.path.join(model_path, "model.txt")
        model.save_model(model_file)
        
        # Save the scaler and metadata
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        
        with open(os.path.join(model_path, "metadata.pkl"), "wb") as f:
            metadata = {
                "feature_names": feature_names
            }
            pickle.dump(metadata, f)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_path": model_path
        }
    
    def _deploy_prophet(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a Prophet model.
        
        Args:
            model_info: Dictionary containing model and metadata
            
        Returns:
            Dictionary with deployed model and metadata
        """
        model = model_info["model"]
        
        # Save model locally
        model_path = os.path.join("models", "prophet_model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the Prophet model
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        return {
            "model": model,
            "model_path": model_path
        }
    
    def _deploy_statsmodels(self, model_info: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Deploy a statsmodels model (ARIMA/SARIMA).
        
        Args:
            model_info: Dictionary containing model and metadata
            model_name: Name of the model (ARIMA or SARIMA)
            
        Returns:
            Dictionary with deployed model and metadata
        """
        model = model_info["model"]
        
        # Save model locally
        model_path = os.path.join("models", f"{model_name.lower()}_model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the statsmodels model
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        return {
            "model": model,
            "model_path": model_path
        }
    
    def load_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """
        Load a deployed model.
        
        Args:
            model_path: Path to the saved model
            model_name: Name of the model
            
        Returns:
            Dictionary containing the loaded model and metadata
        """
        # Determine loading method based on model type
        if model_name == "LSTM":
            return self._load_lstm(model_path)
        elif model_name == "LightGBM":
            return self._load_lightgbm(model_path)
        elif model_name == "Prophet":
            return self._load_prophet(model_path)
        elif model_name in ["ARIMA", "SARIMA"]:
            return self._load_statsmodels(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def _load_lstm(self, model_path: str) -> Dict[str, Any]:
        """
        Load a deployed LSTM model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary with loaded model and metadata
        """
        import tensorflow as tf
        
        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)
        
        # Load the scaler and metadata
        with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(model_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": metadata["feature_names"],
            "timesteps": metadata["timesteps"],
            "model_path": model_path
        }
    
    def _load_lightgbm(self, model_path: str) -> Dict[str, Any]:
        """
        Load a deployed LightGBM model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary with loaded model and metadata
        """
        import lightgbm as lgb
        
        # Load the LightGBM model
        model_file = os.path.join(model_path, "model.txt")
        model = lgb.Booster(model_file=model_file)
        
        # Load the scaler and metadata
        with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(model_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": metadata["feature_names"],
            "model_path": model_path
        }
    
    def _load_prophet(self, model_path: str) -> Dict[str, Any]:
        """
        Load a deployed Prophet model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary with loaded model and metadata
        """
        # Load the Prophet model
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        return {
            "model": model,
            "model_path": model_path
        }
    
    def _load_statsmodels(self, model_path: str) -> Dict[str, Any]:
        """
        Load a deployed statsmodels model (ARIMA/SARIMA).
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary with loaded model and metadata
        """
        # Load the statsmodels model
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        return {
            "model": model,
            "model_path": model_path
        }
