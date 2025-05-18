import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL if DATABASE_URL else 'postgresql://postgres:postgres@localhost:5432/postgres')
Base = declarative_base()
Session = sessionmaker(bind=engine)

class LoadData(Base):
    """SQLAlchemy model for storing load data"""
    __tablename__ = 'load_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    load = Column(Float, nullable=False)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    is_holiday = Column(Boolean, nullable=True)
    
    def __repr__(self):
        return f"<LoadData(timestamp='{self.timestamp}', load='{self.load}')>"

class ModelMetadata(Base):
    """SQLAlchemy model for storing model metadata"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    trained_date = Column(DateTime, nullable=False)
    mape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    model_path = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<ModelMetadata(model_name='{self.model_name}', trained_date='{self.trained_date}')>"

class ForecastResult(Base):
    """SQLAlchemy model for storing forecast results"""
    __tablename__ = 'forecast_results'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    forecast_date = Column(DateTime, nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<ForecastResult(prediction_timestamp='{self.prediction_timestamp}', predicted_value='{self.predicted_value}')>"

def init_db():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(engine)
    print("Database tables created successfully")

def store_sample_data(df):
    """
    Store sample data in the database
    
    Args:
        df: DataFrame containing load data
    """
    # Rename columns if necessary
    if 'timestamp' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert dataframe to SQL
    df.to_sql('load_data', engine, if_exists='replace', index=False)
    
    print(f"Stored {len(df)} rows of load data in the database")

def load_data_from_db(start_date=None, end_date=None):
    """
    Load data from the database with optional date filtering
    
    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        DataFrame containing the load data
    """
    query = "SELECT * FROM load_data"
    
    if start_date and end_date:
        query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        query += f" WHERE timestamp >= '{start_date}'"
    elif end_date:
        query += f" WHERE timestamp <= '{end_date}'"
    
    query += " ORDER BY timestamp"
    
    df = pd.read_sql(query, engine)
    return df

def store_model_metadata(model_name, metrics, model_path=None):
    """
    Store model metadata in the database
    
    Args:
        model_name: Name of the model
        metrics: Dictionary containing model performance metrics
        model_path: Optional path to the saved model
        
    Returns:
        ID of the inserted model metadata
    """
    session = Session()
    
    model_metadata = ModelMetadata(
        model_name=model_name,
        trained_date=datetime.datetime.now(),
        mape=metrics.get('mape'),
        rmse=metrics.get('rmse'),
        r2=metrics.get('r2'),
        model_path=model_path
    )
    
    session.add(model_metadata)
    session.commit()
    
    model_id = model_metadata.id
    session.close()
    
    return model_id

def store_forecast_results(model_id, forecasts):
    """
    Store forecast results in the database
    
    Args:
        model_id: ID of the model used for forecasting
        forecasts: DataFrame containing forecast results
    """
    # Ensure correct column names
    if 'date' in forecasts.columns:
        forecasts = forecasts.rename(columns={'date': 'prediction_timestamp'})
    elif 'timestamp' in forecasts.columns:
        forecasts = forecasts.rename(columns={'timestamp': 'prediction_timestamp'})
    
    if 'forecast' in forecasts.columns:
        forecasts = forecasts.rename(columns={'forecast': 'predicted_value'})
    
    # Add model_id and forecast_date
    forecasts['model_id'] = model_id
    forecasts['forecast_date'] = datetime.datetime.now()
    
    # Store in database
    forecasts.to_sql('forecast_results', engine, if_exists='append', index=False)
    
    print(f"Stored {len(forecasts)} forecast results in the database")

if __name__ == "__main__":
    # Initialize the database
    init_db()