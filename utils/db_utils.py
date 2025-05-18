import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

class DatabaseHandler:
    """
    Handles database interactions for the forecasting system.
    """
    
    def __init__(self):
        """Initialize the database connection."""
        self.conn = None
        self.connect()
    
    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            # Get database credentials from environment variables
            self.conn = psycopg2.connect(
                host=os.environ.get('PGHOST'),
                database=os.environ.get('PGDATABASE'),
                user=os.environ.get('PGUSER'),
                password=os.environ.get('PGPASSWORD'),
                port=os.environ.get('PGPORT')
            )
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def save_dataset(self, name, description, data):
        """
        Save dataset information to the database.
        
        Args:
            name: Name of the dataset
            description: Description of the dataset
            data: Pandas DataFrame with the dataset
        
        Returns:
            Dataset ID if successful, None otherwise
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Extract metadata
            columns = list(data.columns)
            sample_data = data.head(5).to_dict(orient='records')
            row_count = len(data)
            
            # Insert into database
            cursor.execute(
                """
                INSERT INTO datasets (name, description, columns, sample_data, row_count)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, description, Json(columns), Json(sample_data), row_count)
            )
            
            dataset_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            return dataset_id
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving dataset: {e}")
            return None
    
    def save_model(self, name, model_type, metrics, parameters):
        """
        Save model information to the database.
        
        Args:
            name: Name of the model
            model_type: Type of model (e.g., 'RandomForest', 'LSTM')
            metrics: Dictionary of evaluation metrics
            parameters: Dictionary of model parameters
        
        Returns:
            Model ID if successful, None otherwise
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Insert into database
            cursor.execute(
                """
                INSERT INTO forecast_models (name, model_type, metrics, parameters)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, model_type, Json(metrics), Json(parameters))
            )
            
            model_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            return model_id
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving model: {e}")
            return None
    
    def save_forecast(self, model_id, forecast_data, horizon, actual_values=None):
        """
        Save forecast results to the database.
        
        Args:
            model_id: ID of the model used for forecasting
            forecast_data: DataFrame or dictionary with forecast values
            horizon: Number of periods forecasted
            actual_values: Optional DataFrame or dictionary with actual values
        
        Returns:
            Forecast ID if successful, None otherwise
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Convert to dictionary if DataFrame
            if isinstance(forecast_data, pd.DataFrame):
                forecast_values = forecast_data.to_dict(orient='records')
            else:
                forecast_values = forecast_data
            
            # Convert actual values if provided
            if actual_values is not None:
                if isinstance(actual_values, pd.DataFrame):
                    actual_values = actual_values.to_dict(orient='records')
            else:
                actual_values = {}
            
            # Insert into database
            cursor.execute(
                """
                INSERT INTO forecasts (model_id, forecast_date, horizon, forecast_values, actual_values)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (model_id, datetime.now(), horizon, Json(forecast_values), Json(actual_values))
            )
            
            forecast_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            return forecast_id
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving forecast: {e}")
            return None
    
    def get_dataset_list(self):
        """
        Get a list of all saved datasets.
        
        Returns:
            DataFrame with dataset information
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                SELECT id, name, description, uploaded_at, row_count
                FROM datasets
                ORDER BY uploaded_at DESC
                """
            )
            
            columns = ['id', 'name', 'description', 'uploaded_at', 'row_count']
            datasets = pd.DataFrame(cursor.fetchall(), columns=columns)
            cursor.close()
            return datasets
        
        except Exception as e:
            print(f"Error getting dataset list: {e}")
            return None
    
    def get_model_list(self):
        """
        Get a list of all saved models.
        
        Returns:
            DataFrame with model information
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                SELECT id, name, model_type, created_at
                FROM forecast_models
                ORDER BY created_at DESC
                """
            )
            
            columns = ['id', 'name', 'model_type', 'created_at']
            models = pd.DataFrame(cursor.fetchall(), columns=columns)
            cursor.close()
            return models
        
        except Exception as e:
            print(f"Error getting model list: {e}")
            return None
    
    def get_forecast_list(self):
        """
        Get a list of all saved forecasts.
        
        Returns:
            DataFrame with forecast information
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                SELECT f.id, m.name, m.model_type, f.forecast_date, f.horizon, f.created_at
                FROM forecasts f
                JOIN forecast_models m ON f.model_id = m.id
                ORDER BY f.created_at DESC
                """
            )
            
            columns = ['id', 'model_name', 'model_type', 'forecast_date', 'horizon', 'created_at']
            forecasts = pd.DataFrame(cursor.fetchall(), columns=columns)
            cursor.close()
            return forecasts
        
        except Exception as e:
            print(f"Error getting forecast list: {e}")
            return None
    
    def get_forecast_by_id(self, forecast_id):
        """
        Get forecast details by ID.
        
        Args:
            forecast_id: ID of the forecast to retrieve
        
        Returns:
            Dictionary with forecast information
        """
        if self.conn is None and not self.connect():
            return None
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                SELECT f.id, m.name, m.model_type, f.forecast_date, f.horizon, 
                       f.created_at, f.forecast_values, f.actual_values, m.metrics
                FROM forecasts f
                JOIN forecast_models m ON f.model_id = m.id
                WHERE f.id = %s
                """,
                (forecast_id,)
            )
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            forecast = {
                'id': row[0],
                'model_name': row[1],
                'model_type': row[2],
                'forecast_date': row[3],
                'horizon': row[4],
                'created_at': row[5],
                'forecast_values': row[6],
                'actual_values': row[7],
                'model_metrics': row[8]
            }
            
            cursor.close()
            return forecast
        
        except Exception as e:
            print(f"Error getting forecast details: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()