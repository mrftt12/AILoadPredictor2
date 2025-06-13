# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

### Development Server
```bash
streamlit run app.py                    # Full-featured app with agent architecture
streamlit run app_simplified.py         # Simplified app without external dependencies
```

### Database Setup
```bash
python setup_database.py               # Initialize database and load sample data
```

### Dependencies
```bash
# Dependencies are managed via pyproject.toml
# Main dependencies: streamlit, lightgbm, mlflow, prophet, tensorflow, psycopg2-binary
```

## Architecture Overview

### Agent-Based System Design
The system follows a modular agent-based architecture where specialized agents handle different aspects of the forecasting pipeline:

- **CoordinatingAgent**: Central orchestrator that manages workflow and agent communication
- **DataProcessingAgent**: Handles data ingestion, validation, and preprocessing
- **EDAAgent**: Performs exploratory data analysis and pattern detection
- **ModelingAgent**: Manages model training with multiple algorithms (RandomForest, LinearRegression, Prophet, LSTM, LightGBM)
- **ModelVerificationAgent**: Evaluates model performance and cross-validation
- **ModelDeploymentAgent**: Handles model versioning and deployment via MLflow
- **ForecastingAgent**: Generates forecasts with uncertainty quantification
- **VisualizationAgent**: Creates interactive visualizations using Plotly

### Application Variants
- **app.py**: Full agent-based application with MLflow integration, TensorFlow models, and comprehensive features
- **app_simplified.py**: Streamlined version using only scikit-learn models, suitable for environments without complex dependencies

### Database Integration
- Uses PostgreSQL for persistent storage of datasets, models, and forecasts
- Two database handling approaches:
  - `db_manager.py`: SQLAlchemy-based ORM approach
  - `db_utils.py`: Direct psycopg2 implementation
- Database credentials managed via environment variables (PGHOST, PGDATABASE, PGUSER, PGPASSWORD, PGPORT)

### Data Flow
1. Data ingestion from multiple sources (CSV upload, URLs, sample data, database)
2. Data validation and preprocessing with feature engineering
3. Exploratory data analysis with pattern detection
4. Model training and evaluation across multiple algorithms
5. Model selection based on performance metrics
6. Forecast generation with confidence intervals
7. Results visualization and report generation

## Key Technical Details

### Model Support
- Traditional ML: RandomForest, LinearRegression
- Time Series: Prophet, SARIMA, ARIMA
- Deep Learning: LSTM networks via TensorFlow
- Ensemble: LightGBM

### Feature Engineering
- Automatic time-based feature generation (hour, day_of_week, month, cyclical encoding)
- Support for external features (temperature, humidity, etc.)
- Standardization and scaling handled automatically

### Development Patterns
- Agent methods return structured dictionaries with results and metadata
- Error handling throughout the pipeline with user-friendly messages
- Session state management in Streamlit for workflow persistence
- Database transactions properly managed with rollback capabilities

### Testing and Validation
- Cross-validation integrated into model training
- Multiple evaluation metrics (MAPE, RMSE, R²)
- Model comparison and automatic best model selection
- Forecast uncertainty quantification