# Flight Delay Predictor

A machine learning project for predicting flight departure delay in minutes using historical flight data from the U.S. Bureau of Transportation Statistics (BTS) and weather data from Open-Meteo.

The project implements an end-to-end ML pipeline covering:

* Data collection
* Data preprocessing
* Historical feature engineering
* Weather data integration
* Model training and validation
* Hyperparameter tuning
* MLflow experiment tracking
* MLflow Model Registry
* Champion/Challenger model management
* Online prediction through FastAPI
* Batch prediction
* Streamlit dashboard
* Docker-based deployment

---

# Project Architecture

```text
                         ┌─────────────────────┐
                         │      BTS Data       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Data Collector   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PreProcessor     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Historical Features │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Pipeline    │
                         └──────────┬──────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │            Model Training              │
              │                                        │
              │ Linear Regression                       │
              │ Lasso                                  │
              │ Ridge                                  │
              │ ElasticNet                             │
              │ Random Forest                          │
              │ Gradient Boosting                      │
              │ LightGBM                               │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                         ┌─────────────────────┐
                         │      MLflow         │
                         │ Tracking + Registry │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                  Champion                   Challenger
                       │
                       ▼
                 ┌─────────────┐
                 │   FastAPI   │
                 └──────┬──────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Online Prediction      Batch Prediction
             │                     │
             └──────────┬──────────┘
                        ▼
                  Streamlit UI
```

---

# Project Structure

```text
flight_delay_predictor/
│
├── api/
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── batch.py
│   └── BatchParser.py
│
├── config/
│   └── config.py
│
├── DataPipeline/
│   ├── data_collector.py
│   ├── DataPreprocessor.py
│   └── FeatureEngineering.py
│
├── models/
│   ├── ModelTrainer.py
│   └── register.py
│
├── streamlit_gui/
│   └── app.py
│
├── data/
│   ├── airports.csv
│   └── optional_data.json
│
├── mlartifacts/
├── mlflow.db
│
├── flow.py
│
├── Dockerfile.api
├── Dockerfile.train
├── Dockerfile.streamlit
├── docker-compose.yml
│
├── requirements-api.txt
├── requirements-train.txt
├── requirements-dashboard.txt
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Data Sources

## Flight Data

Flight data is collected from the U.S. Bureau of Transportation Statistics (BTS).

The project uses historical on-time performance data containing information such as:

* Flight date
* Airline
* Tail number
* Origin
* Destination
* Scheduled departure and arrival times
* Scheduled elapsed time
* Distance
* Departure delay
* Arrival delay
* Cancellation
* Diversion

## Weather Data

Weather information is collected from Open-Meteo.

The project uses weather features including:

```text
temperature_2m
relative_humidity_2m
dew_point_2m
precipitation
wind_speed_10m
wind_direction_10m
surface_pressure
cloud_cover
weather_code
```

Flight and weather data are combined during preprocessing.

---

# Features

## Flight Features

```text
Year
Month
DayofMonth
DayOfWeek
DayOfYear
Tail_Number
Reporting_Airline
Origin
Dest
CRSDepHour
CRSDepMinute
CRSArrHour
CRSArrMinute
CRSElapsedTime
Distance
Cancelled
```

## Time Features

Cyclic representations are generated for several time-related features using sine and cosine transformations.

## Historical Features

The project generates historical features based on previous observations, including:

```text
PrevArrDelay
PrevOriginDelay_mean_*
PrevOrigin_AirlineDelay_mean_*
PrevRouteDelay_mean_*
Tail_Mean_arrdelay_*
```

Historical features use previous observations rather than the current flight's future information.

---

# Data Processing Pipeline

The training pipeline follows:

```text
Collect
   ↓
Preprocess
   ↓
Historical Feature Engineering
   ↓
Train / Validation / Test Split
   ↓
Feature Transformation
   ↓
Model Training
   ↓
Model Selection
   ↓
Optional Hyperparameter Tuning
   ↓
Test Evaluation
   ↓
MLflow Registration
   ↓
Champion / Challenger
```

The dataset is split chronologically to preserve the temporal nature of the prediction problem.

---

# Models

The training stage evaluates multiple models:

```text
Linear Regression
Lasso
Ridge
ElasticNet
Random Forest
Gradient Boosting
LightGBM
```

The best model is selected using validation `R2`.

The selected model can optionally undergo hyperparameter tuning.

---

# Evaluation Metrics

The project records metrics including:

```text
R2
RMSE
MAE
Training Time
Overfitting Gap
```

Final test metrics are logged in MLflow.

Model performance is dependent on the training period, available data, feature configuration, and preprocessing configuration. Therefore, metrics should be obtained from the corresponding MLflow run rather than treated as fixed project-wide results.

---

# MLflow

MLflow is used for:

* Experiment tracking
* Parameter logging
* Metric logging
* Model artifacts
* Feature importance artifacts
* Model Registry
* Model aliases

The registered model is:

```text
FlightDelayPredictor
```

The project uses:

```text
champion
challenger
```

aliases.

The model with the better validation `R2` can become the champion when compared with the existing champion.

The preprocessing pipeline is stored separately from the trained model.

The project uses Python `pickle` for model/preprocessor serialization where applicable. `joblib` is not required by the current implementation.

---

# Running Locally

## Requirements

The local environment requires:

* Python 3.13
* `uv`
* Internet access for BTS/Open-Meteo data
* MLflow
* Dependencies specified in `pyproject.toml`

## Install Dependencies

```bash
uv sync
```

## Start MLflow

Run:

```bash
uv run mlflow server --host 0.0.0.0 --port 1090 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Set the tracking URI if necessary.

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:1090"
```

## Train the Model

Run:

```bash
uv run python flow.py
```

The current `flow.py` executes:

```python
flight_delay_pipeline()
```

To enable tuning:

```python
flight_delay_pipeline(tune=True)
```

The training process performs:

1. Data collection
2. Data preprocessing
3. Historical feature engineering
4. Train/validation/test splitting
5. Feature transformation
6. Training multiple models
7. Model selection
8. Optional tuning
9. Test evaluation
10. MLflow registration
11. Champion/Challenger assignment

---

# Run the API Locally

After a model has been trained and registered:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 1091
```

API:

```text
http://localhost:1091
```

Health endpoint:

```text
GET /health
```

Optional data:

```text
GET /optional_data
```

Online prediction:

```text
POST /predict
```

---

# Batch Prediction

Start a monthly analysis:

```text
GET /analyze/{year}/{month}
```

Example:

```text
/analyze/2023/10
```

Retrieve the result:

```text
GET /result/{year}/{month}
```

Retrieve all stored results:

```text
GET /results
```

---

# Streamlit Dashboard

Run:

```bash
uv run streamlit run streamlit_gui/app.py --server.port 1092
```

Dashboard:

```text
http://localhost:1092
```

The dashboard contains:

### Flight Predictor

Performs online flight-delay prediction.

### Batch Prediction

Starts monthly batch prediction and displays the result.

### Monitor

Displays previously processed monthly results and metrics.

---

# Docker Deployment

The project provides separate Docker images for:

```text
Dockerfile.api
Dockerfile.train
Dockerfile.streamlit
```

Docker Compose manages:

```text
MLflow
API
Trainer
Streamlit
```

## Build

From the project root:

```bash
docker compose build
```

## Start the Application

```bash
docker compose up
```

The services are available at:

```text
MLflow:    http://localhost:1090
FastAPI:   http://localhost:1091
Streamlit: http://localhost:1092
```

---

# Docker Training

The training service uses the Compose profile:

```text
training
```

Run training with:

```bash
docker compose --profile training up --build trainer
```

The trainer communicates with MLflow using the Docker service name:

```text
http://mlflow:1090
```

Inside Docker, containers must communicate using service names rather than `localhost`.

For example:

```text
Correct:
http://mlflow:1090
http://api:1091

Incorrect from another container:
http://localhost:1090
http://localhost:1091
```

---

# MLflow SQLite Database with Docker

MLflow uses SQLite as its backend database.

The database file is:

```text
mlflow.db
```

The artifact directory is:

```text
mlartifacts/
```

The Docker Compose configuration mounts these paths into the MLflow container.

A typical configuration is:

```yaml
volumes:
  - ./mlflow.db:/app/mlflow.db
  - ./mlartifacts:/app/mlartifacts
```

The MLflow server uses:

```text
sqlite:////app/mlflow.db
```

inside the container.

## Important SQLite Docker Problem

When using a bind mount such as:

```yaml
- ./mlflow.db:/app/mlflow.db
```

Docker expects the host-side path to correspond to the intended file.

If `mlflow.db` does not exist on the host when Docker creates the container, Docker can create a directory at that path depending on the Docker/Compose environment.

This causes MLflow to fail because MLflow expects:

```text
/app/mlflow.db
```

to be a SQLite database file, but it is actually a directory.

Typical errors can mention that the database path is a directory or that SQLite cannot open the database.

## How to Fix It

First stop the containers:

```bash
docker compose down
```

Check the project directory.

On Windows PowerShell:

```powershell
Get-Item .\mlflow.db
```

If `mlflow.db` is a directory instead of a file, remove the directory:

```powershell
Remove-Item .\mlflow.db -Recurse -Force
```

Then create an empty database file before starting Docker:

```powershell
New-Item .\mlflow.db -ItemType File
```

Also make sure the artifact directory exists:

```powershell
New-Item .\mlartifacts -ItemType Directory -Force
```

Then start MLflow again:

```bash
docker compose up mlflow
```

After MLflow starts successfully, the database will be initialized and populated by MLflow.

## Recommended Project Layout

Before running Docker:

```text
project/
├── mlflow.db
├── mlartifacts/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.train
├── Dockerfile.streamlit
└── ...
```

The important distinction is:

```text
mlflow.db       → FILE
mlartifacts/    → DIRECTORY
```

Do not create `mlflow.db` as a directory.

## If the Database Already Contains Important Runs

Do not delete the database just to solve a mount problem.

First stop the containers:

```bash
docker compose down
```

Then verify that the host path is actually a file:

```powershell
Get-Item .\mlflow.db
```

If the existing database is valid, keep it and make sure Compose mounts that exact file.

Only remove/recreate the database if losing the existing MLflow tracking history is acceptable.

---

# MLflow Artifact Storage

MLflow stores artifacts separately from the SQLite backend database.

The database contains tracking information such as:

```text
Experiments
Runs
Parameters
Metrics
Tags
Model Registry metadata
```

The artifact directory contains files such as:

```text
Model artifacts
Preprocessor
Feature importance files
Other logged artifacts
```

Therefore, both should be preserved:

```text
mlflow.db
mlartifacts/
```

Deleting only `mlflow.db` can remove the tracking metadata even if the artifact files still exist.

Deleting `mlartifacts/` can remove model artifacts even if the database still contains their metadata.

For this reason, both should be backed up together.

---

# Docker Services

## MLflow

```text
Port: 1090
```

MLflow provides:

* Tracking server
* Experiment management
* Model Registry
* Artifact serving

## API

```text
Port: 1091
```

The API loads the registered model associated with the `champion` alias.

## Trainer

The trainer is disabled from the normal Compose startup and is enabled through the `training` profile.

## Streamlit

```text
Port: 1092
```

Inside Docker, Streamlit communicates with:

```text
http://api:1091
```

---

# Online Prediction Flow

```text
Streamlit
   ↓
POST /predict
   ↓
FastAPI
   ↓
Weather Data
   ↓
Historical Feature Engineering
   ↓
Saved Preprocessor
   ↓
Champion Model
   ↓
Predicted Delay
```

The response format is:

```json
{
  "delay_minutes": 12.5
}
```

---

# Batch Prediction Flow

```text
Streamlit
   ↓
GET /analyze/{year}/{month}
   ↓
FastAPI Background Task
   ↓
Flight Data
   ↓
Weather Data
   ↓
Feature Engineering
   ↓
Champion Model
   ↓
Save Result
   ↓
GET /result/{year}/{month}
```

---

# Model Loading

At API startup:

```text
FastAPI
   ↓
Model Registry
   ↓
champion alias
   ↓
Registered Model
   ↓
Saved Preprocessor
   ↓
Ready for Prediction
```

The API does not train models.

Training and model registration are separate operations.

---

# Configuration

Main configuration is located in:

```text
config/config.py
```

Configuration classes include:

```text
DataConfig
SplitConfig
PathConfig
ModelConfig
MlflowConfig
```

The MLflow tracking URI can be configured through:

```text
MLFLOW_TRACKING_URI
```

Local example:

```text
http://localhost:1090
```

Docker example:

```text
http://mlflow:1090
```

The difference is important because `localhost` inside a container refers to that same container, not the MLflow container.

---

# Line Endings on Windows

The project is developed on Windows but Docker containers run Linux.

Git may display a warning such as:

```text
warning: in the working copy of 'requirements-api.txt',
LF will be replaced by CRLF
```

This is a Git line-ending warning, not a Docker or Python error.

The recommended configuration for this project is to keep repository files using LF:

```bash
git config --global core.autocrlf input
```

A `.gitattributes` file can also enforce LF:

```gitattributes
* text=auto
*.py text eol=lf
*.txt text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
Dockerfile text eol=lf
```

After adding `.gitattributes`, normalize the repository:

```bash
git add --renormalize .
```

This is especially useful because the project builds Linux Docker images.

---

# Important Notes

* `DepDelay` is the prediction target.
* `ArrDelay` is used for historical features and is not used as the current prediction target.
* Historical features use previous observations.
* Training is performed chronologically.
* The preprocessing pipeline is stored separately from the trained model.
* The API loads the model through the MLflow `champion` alias.
* MLflow tracking metadata is stored in `mlflow.db`.
* MLflow artifacts are stored in `mlartifacts/`.
* `mlflow.db` must be a file, not a directory.
* The API does not train models.
* The trainer communicates with MLflow using the Docker service name.
* Streamlit communicates with the API using the Docker service name.
* Model metrics can change when the training data or feature configuration changes.

---

# Troubleshooting

## Docker Cannot Build the API Image

Make sure the requirements file is copied before installation:

```dockerfile
COPY requirements-api.txt .

RUN uv pip install --system -r requirements-api.txt
```

The same principle applies to the training and dashboard images.

## MLflow Cannot Open the Database

Check:

```text
mlflow.db
```

It must be a file.

On PowerShell:

```powershell
Get-Item .\mlflow.db
```

If it is a directory:

```powershell
Remove-Item .\mlflow.db -Recurse -Force
New-Item .\mlflow.db -ItemType File
```

Then restart:

```bash
docker compose down
docker compose up
```

## API Cannot Connect to MLflow

Inside Docker use:

```text
http://mlflow:1090
```

not:

```text
http://localhost:1090
```

## Streamlit Cannot Connect to API

Inside Docker use:

```text
http://api:1091
```

not:

```text
http://localhost:1091
```

## Training Service Is Not Starting

The training service uses the `training` profile.

Run:

```bash
docker compose --profile training up --build trainer
```

---

# Technologies

```text
Python
Pandas
NumPy
Scikit-learn
LightGBM
FastAPI
Streamlit
MLflow
Prefect
Docker
Docker Compose
uv
Open-Meteo
BTS
```

---

# Future Improvements

Possible future improvements include:

* More robust batch-job status management
* Better handling of missing weather observations
* Automated model retraining
* Scheduled data collection
* Model drift monitoring
* Data-quality monitoring
* More efficient historical feature computation
* Additional hyperparameter optimization
* Automated Champion/Challenger evaluation
* Production-grade task queue for batch prediction
