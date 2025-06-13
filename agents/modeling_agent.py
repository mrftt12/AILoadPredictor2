import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import statsmodels.api as sm
from prophet import Prophet

class ModelingAgent:
    """
    Trains and tunes various forecasting models.
    Responsible for preparing data, training models, implementing hyperparameter tuning,
    and logging experiments.
    """
    
    def __init__(self):
        """Initialize the modeling agent."""
        # Set up MLflow tracking URI if not configured
        if mlflow.get_tracking_uri() is None:
            mlflow.set_tracking_uri("mlruns")
        
        # Define default hyperparameter grids
        self.hyperparameter_grids = {
            "RandomForest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "LightGBM": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "num_leaves": [31, 50, 100],
                "max_depth": [-1, 10, 20]
            },
            "Prophet": {
                "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
                "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
                "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0]
            },
            "ARIMA": {
                "p": [0, 1, 2],
                "d": [0, 1],
                "q": [0, 1, 2]
            },
            "SARIMA": {
                "p": [0, 1, 2],
                "d": [0, 1],
                "q": [0, 1, 2],
                "P": [0, 1],
                "D": [0, 1],
                "Q": [0, 1],
                "s": [12, 24] # seasonal periods
            }
        }
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Suppress warnings during model training
        warnings.filterwarnings("ignore")
    
    def train(self, data: pd.DataFrame, target_col: str, timestamp_col: str,
              models: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train and tune specified models on the provided data.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            models: List of model names to train
            config: Configuration dictionary for training
            
        Returns:
            Dictionary containing trained models and training details
        """
        # Start an MLflow experiment
        experiment_name = f"load_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test, feature_names, scaler = self._prepare_data(
            data, target_col, timestamp_col, config.get("train_size", 0.8)
        )
        
        # Create a dictionary to store trained models
        trained_models = {}
        
        # Train each requested model
        for model_name in models:
            print(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=model_name) as run:
                # Log training configuration
                mlflow.log_params({
                    "model_name": model_name,
                    "train_size": config.get("train_size", 0.8),
                    "features": ", ".join(feature_names),
                    "target": target_col
                })
                
                # Train the specific model
                if model_name == "RandomForest":
                    model, metrics = self._train_random_forest(
                        X_train, y_train, X_test, y_test, 
                        config, feature_names
                    )
                
                elif model_name == "LightGBM":
                    model, metrics = self._train_lightgbm(
                        X_train, y_train, X_test, y_test, 
                        config, feature_names
                    )
                
                elif model_name == "Prophet":
                    model, metrics = self._train_prophet(
                        data, target_col, timestamp_col,
                        config
                    )
                
                elif model_name == "ARIMA":
                    model, metrics = self._train_arima(
                        data, target_col, config
                    )
                
                elif model_name == "SARIMA":
                    model, metrics = self._train_sarima(
                        data, target_col, config
                    )
                
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Add trained model to the results
                trained_models[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "scaler": scaler,
                    "feature_names": feature_names,
                    "run_id": run.info.run_id
                }
        
        # Return trained models
        return {
            "models": trained_models,
            "experiment_id": mlflow.get_experiment_by_name(experiment_name).experiment_id,
            "X_test": X_test,
            "y_test": y_test
        }
    
    def _prepare_data(self, data: pd.DataFrame, target_col: str, timestamp_col: str, 
                    train_size: float) -> Tuple:
        """
        Prepare data for model training.
        
        Args:
            data: Processed dataframe
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            train_size: Proportion of data to use for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
        """
        df = data.copy()
        
        # Remove timestamp column from features
        feature_cols = [col for col in df.columns if col not in [timestamp_col, target_col]]
        
        # Split data into features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=train_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, feature_cols, scaler
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test, config, feature_names):
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            config: Training configuration
            feature_names: Names of features
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Define hyperparameters for tuning
        tuning_method = config.get("tuning_method", "none")
        if tuning_method == "grid_search":
            # Grid search for hyperparameters
            param_grid = self.hyperparameter_grids["RandomForest"]
            
            # Create grid search object
            grid_search = GridSearchCV(
                estimator=RandomForestRegressor(random_state=config.get("random_state", 42)),
                param_grid=param_grid,
                cv=config.get("cv_folds", 3),
                scoring='neg_mean_squared_error',
                n_jobs=config.get("n_jobs", -1)
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            
            # Create model with best parameters
            model = RandomForestRegressor(
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", None),
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                random_state=config.get("random_state", 42)
            )
            
        elif tuning_method == "random_search":
            # Random search for hyperparameters
            import random
            hp_grid = self.hyperparameter_grids["RandomForest"]
            
            # Randomly select parameters
            rand_params = {
                "n_estimators": random.choice(hp_grid["n_estimators"]),
                "max_depth": random.choice(hp_grid["max_depth"]),
                "min_samples_split": random.choice(hp_grid["min_samples_split"]),
                "min_samples_leaf": random.choice(hp_grid["min_samples_leaf"])
            }
            
            # Create model with random parameters
            model = RandomForestRegressor(
                n_estimators=rand_params["n_estimators"],
                max_depth=rand_params["max_depth"],
                min_samples_split=rand_params["min_samples_split"],
                min_samples_leaf=rand_params["min_samples_leaf"],
                random_state=config.get("random_state", 42)
            )
            
            # Set best_params for logging
            best_params = rand_params
            
        else:
            # Default hyperparameters
            best_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
            
            # Create model with default parameters
            model = RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                random_state=config.get("random_state", 42)
            )
        
        # Log hyperparameters
        mlflow.log_params(best_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Save feature importance to a csv
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Return model and metrics
        return model, metrics
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, config, feature_names):
        """
        Train a LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            config: Training configuration
            feature_names: Names of features
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        
        # Define hyperparameters
        tuning_method = config.get("tuning_method", "none")
        if tuning_method == "grid_search":
            # Grid search for hyperparameters
            param_grid = self.hyperparameter_grids["LightGBM"]
            
            # Create parameter combinations
            best_params = None
            best_score = float('inf')
            
            # Perform cross-validation for each parameter combination
            for learning_rate in param_grid["learning_rate"]:
                for n_estimators in param_grid["n_estimators"]:
                    for num_leaves in param_grid["num_leaves"]:
                        for max_depth in param_grid["max_depth"]:
                            params = {
                                'objective': 'regression',
                                'metric': 'rmse',
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'num_leaves': num_leaves,
                                'max_depth': max_depth,
                                'verbose': -1
                            }
                            
                            # Perform cross-validation
                            cv_results = lgb.cv(
                                params,
                                train_data,
                                num_boost_round=100,
                                nfold=config["cv_folds"],
                                stratified=False,
                                shuffle=False,
                                verbose_eval=False
                            )
                            
                            # Get the best score
                            best_cv_score = min(cv_results['rmse-mean'])
                            
                            # Update best parameters if better score
                            if best_cv_score < best_score:
                                best_score = best_cv_score
                                best_params = params
            
            # Use best parameters
            params = best_params
            
        elif tuning_method == "random_search":
            # Random search for hyperparameters
            import random
            hp_grid = self.hyperparameter_grids["LightGBM"]
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': random.choice(hp_grid["learning_rate"]),
                'n_estimators': random.choice(hp_grid["n_estimators"]),
                'num_leaves': random.choice(hp_grid["num_leaves"]),
                'max_depth': random.choice(hp_grid["max_depth"]),
                'verbose': -1
            }
            
        else:
            # Default hyperparameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'n_estimators': 200,
                'num_leaves': 31,
                'max_depth': -1,
                'verbose': -1
            }
        
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[train_data],
            verbose_eval=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Log the model
        mlflow.lightgbm.log_model(model, "lightgbm_model")
        
        # Log feature importance
        if hasattr(model, 'feature_importance'):
            importance = model.feature_importance()
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            # Save feature importance to a csv
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
        
        # Return model and metrics
        return model, metrics
    
    def _train_prophet(self, data, target_col, timestamp_col, config):
        """
        Train a Prophet model.
        
        Args:
            data: DataFrame with timestamp and target columns
            target_col: Column name containing the target values
            timestamp_col: Column name containing timestamps
            config: Training configuration
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Prepare data for Prophet
        df_prophet = data[[timestamp_col, target_col]].copy()
        df_prophet.columns = ['ds', 'y']  # Prophet requires these column names
        
        # Split into train and test sets
        train_size = int(len(df_prophet) * config.get("train_size", 0.8))
        df_train = df_prophet.iloc[:train_size]
        df_test = df_prophet.iloc[train_size:]
        
        # Hyperparameter tuning
        tuning_method = config.get("tuning_method", "none")
        if tuning_method in ["grid_search", "random_search"]:
            param_grid = self.hyperparameter_grids["Prophet"]
            
            if tuning_method == "grid_search":
                # Grid search for hyperparameters
                best_params = None
                best_score = float('inf')
                
                # Iterate through parameter combinations
                for changepoint_prior_scale in param_grid["changepoint_prior_scale"]:
                    for seasonality_prior_scale in param_grid["seasonality_prior_scale"]:
                        for holidays_prior_scale in param_grid["holidays_prior_scale"]:
                            try:
                                # Initialize and fit model
                                model = Prophet(
                                    changepoint_prior_scale=changepoint_prior_scale,
                                    seasonality_prior_scale=seasonality_prior_scale,
                                    holidays_prior_scale=holidays_prior_scale
                                )
                                
                                model.fit(df_train)
                                
                                # Make predictions on the test set
                                future = model.make_future_dataframe(periods=len(df_test))
                                forecast = model.predict(future)
                                
                                # Extract predictions for the test period
                                y_pred = forecast.iloc[-len(df_test):]['yhat'].values
                                y_true = df_test['y'].values
                                
                                # Calculate RMSE
                                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                                
                                # Update best parameters if better score
                                if rmse < best_score:
                                    best_score = rmse
                                    best_params = {
                                        'changepoint_prior_scale': changepoint_prior_scale,
                                        'seasonality_prior_scale': seasonality_prior_scale,
                                        'holidays_prior_scale': holidays_prior_scale
                                    }
                            except Exception as e:
                                print(f"Error during Prophet training: {e}")
                                continue
                
                # Use best parameters
                if best_params:
                    prophet_params = best_params
                else:
                    # Default parameters if grid search fails
                    prophet_params = {
                        'changepoint_prior_scale': 0.05,
                        'seasonality_prior_scale': 10.0,
                        'holidays_prior_scale': 10.0
                    }
                    
            else:  # Random search
                # Random search for hyperparameters
                import random
                hp_grid = self.hyperparameter_grids["Prophet"]
                
                prophet_params = {
                    'changepoint_prior_scale': random.choice(hp_grid["changepoint_prior_scale"]),
                    'seasonality_prior_scale': random.choice(hp_grid["seasonality_prior_scale"]),
                    'holidays_prior_scale': random.choice(hp_grid["holidays_prior_scale"])
                }
                
        else:
            # Default parameters
            prophet_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0
            }
        
        # Log hyperparameters
        mlflow.log_params(prophet_params)
        
        # Initialize and fit model with best parameters
        model = Prophet(
            changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
            seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
            holidays_prior_scale=prophet_params['holidays_prior_scale']
        )
        
        # Add additional seasonalities if enough data
        if len(df_train) >= 48:  # At least 2 days of hourly data
            model.add_seasonality(name='daily', period=24, fourier_order=5)
        
        if len(df_train) >= 24*7:  # At least a week of hourly data
            model.add_seasonality(name='weekly', period=24*7, fourier_order=3)
        
        model.fit(df_train)
        
        # Make predictions on the test set
        future = model.make_future_dataframe(periods=len(df_test))
        forecast = model.predict(future)
        
        # Extract predictions for the test period
        y_pred = forecast.iloc[-len(df_test):]['yhat'].values
        y_true = df_test['y'].values
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        # Save the model as a pickle file (Prophet models can't be logged directly with MLflow)
        import pickle
        with open("prophet_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        mlflow.log_artifact("prophet_model.pkl")
        
        # Return model and metrics
        return model, metrics
    
    def _train_arima(self, data, target_col, config):
        """
        Train an ARIMA model.
        
        Args:
            data: DataFrame with target column
            target_col: Column name containing the target values
            config: Training configuration
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Prepare data for ARIMA
        series = data[target_col].copy()
        
        # Split into train and test
        train_size = int(len(series) * config.get("train_size", 0.8))
        train_data = series.iloc[:train_size]
        test_data = series.iloc[train_size:]
        
        # Hyperparameter tuning
        tuning_method = config.get("tuning_method", "none")
        if tuning_method in ["grid_search", "random_search"]:
            param_grid = self.hyperparameter_grids["ARIMA"]
            
            if tuning_method == "grid_search":
                # Grid search for hyperparameters
                best_params = None
                best_score = float('inf')
                
                # Iterate through parameter combinations
                for p in param_grid["p"]:
                    for d in param_grid["d"]:
                        for q in param_grid["q"]:
                            try:
                                # Fit model
                                model = sm.tsa.ARIMA(train_data, order=(p, d, q))
                                model_fit = model.fit()
                                
                                # Get AIC score
                                aic = model_fit.aic
                                
                                # Update best parameters if better score
                                if aic < best_score:
                                    best_score = aic
                                    best_params = (p, d, q)
                            except Exception as e:
                                print(f"Error during ARIMA training with order ({p},{d},{q}): {e}")
                                continue
                
                # Use best parameters
                if best_params:
                    order = best_params
                else:
                    # Default order if grid search fails
                    order = (1, 1, 1)
                    
            else:  # Random search
                # Random search for hyperparameters
                import random
                hp_grid = self.hyperparameter_grids["ARIMA"]
                
                order = (
                    random.choice(hp_grid["p"]),
                    random.choice(hp_grid["d"]),
                    random.choice(hp_grid["q"])
                )
                
        else:
            # Default order
            order = (1, 1, 1)
        
        # Log hyperparameters
        mlflow.log_params({
            "p": order[0],
            "d": order[1],
            "q": order[2]
        })
        
        # Fit model with best parameters
        model = sm.tsa.ARIMA(train_data, order=order)
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.forecast(steps=len(test_data))
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data.values, predictions)
        
        # Save model summary
        with open("arima_summary.txt", "w") as f:
            f.write(str(model_fit.summary()))
        
        mlflow.log_artifact("arima_summary.txt")
        
        # Save the model as a pickle file
        import pickle
        with open("arima_model.pkl", "wb") as f:
            pickle.dump(model_fit, f)
        
        mlflow.log_artifact("arima_model.pkl")
        
        # Return model and metrics
        return model_fit, metrics
    
    def _train_sarima(self, data, target_col, config):
        """
        Train a SARIMA model.
        
        Args:
            data: DataFrame with target column
            target_col: Column name containing the target values
            config: Training configuration
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Prepare data for SARIMA
        series = data[target_col].copy()
        
        # Split into train and test
        train_size = int(len(series) * config.get("train_size", 0.8))
        train_data = series.iloc[:train_size]
        test_data = series.iloc[train_size:]
        
        # Hyperparameter tuning
        tuning_method = config.get("tuning_method", "none")
        if tuning_method in ["grid_search", "random_search"]:
            param_grid = self.hyperparameter_grids["SARIMA"]
            
            if tuning_method == "grid_search":
                # Grid search for hyperparameters (limited)
                best_params = None
                best_score = float('inf')
                
                # Iterate through a subset of parameter combinations
                for p in param_grid["p"][:2]:  # Limit search space
                    for d in param_grid["d"]:
                        for q in param_grid["q"][:2]:
                            for P in param_grid["P"]:
                                for D in param_grid["D"]:
                                    for Q in param_grid["Q"]:
                                        for s in param_grid["s"]:
                                            try:
                                                # Fit model
                                                model = sm.tsa.SARIMAX(
                                                    train_data, 
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, s)
                                                )
                                                model_fit = model.fit(disp=False)
                                                
                                                # Get AIC score
                                                aic = model_fit.aic
                                                
                                                # Update best parameters if better score
                                                if aic < best_score:
                                                    best_score = aic
                                                    best_params = (p, d, q, P, D, Q, s)
                                            except Exception as e:
                                                print(f"Error during SARIMA training with params: {e}")
                                                continue
                
                # Use best parameters
                if best_params:
                    order = best_params[:3]
                    seasonal_order = best_params[3:6] + (best_params[6],)
                else:
                    # Default orders if grid search fails
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 12)
                    
            else:  # Random search
                # Random search for hyperparameters
                import random
                hp_grid = self.hyperparameter_grids["SARIMA"]
                
                order = (
                    random.choice(hp_grid["p"]),
                    random.choice(hp_grid["d"]),
                    random.choice(hp_grid["q"])
                )
                
                seasonal_order = (
                    random.choice(hp_grid["P"]),
                    random.choice(hp_grid["D"]),
                    random.choice(hp_grid["Q"]),
                    random.choice(hp_grid["s"])
                )
                
        else:
            # Default orders
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 24)  # Assuming hourly data with daily seasonality
        
        # Log hyperparameters
        mlflow.log_params({
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "P": seasonal_order[0],
            "D": seasonal_order[1],
            "Q": seasonal_order[2],
            "s": seasonal_order[3]
        })
        
        # Fit model with best parameters
        model = sm.tsa.SARIMAX(
            train_data, 
            order=order,
            seasonal_order=seasonal_order
        )
        
        model_fit = model.fit(disp=False)
        
        # Make predictions
        predictions = model_fit.forecast(steps=len(test_data))
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data.values, predictions)
        
        # Save model summary
        with open("sarima_summary.txt", "w") as f:
            f.write(str(model_fit.summary()))
        
        mlflow.log_artifact("sarima_summary.txt")
        
        # Save the model as a pickle file
        import pickle
        with open("sarima_model.pkl", "wb") as f:
            pickle.dump(model_fit, f)
        
        mlflow.log_artifact("sarima_model.pkl")
        
        # Return model and metrics
        return model_fit, metrics
    
    def _create_sequences(self, X, y, timesteps):
        """
        Create input sequences for LSTM model.
        
        Args:
            X: Features array
            y: Target array
            timesteps: Number of timesteps for each input sequence
            
        Returns:
            X and y arrays reshaped for LSTM
        """
        Xs, ys = [], []
        
        for i in range(len(X) - timesteps):
            Xs.append(X[i:(i + timesteps)])
            ys.append(y[i + timesteps])
            
        return np.array(Xs), np.array(ys)
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Handle potential shape issues
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # to percentage
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate custom metrics
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate mean absolute scaled error (MASE) if sufficient data
        if len(y_true) > 1:
            # Calculate naive forecast errors (using one-step ahead forecast)
            naive_errors = np.abs(np.diff(y_true))
            naive_mae = np.mean(naive_errors)
            
            # Calculate MASE
            if naive_mae != 0:
                mase = mae / naive_mae
            else:
                mase = np.nan
        else:
            mase = np.nan
        
        return {
            "mape": mape,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mase": mase
        }
