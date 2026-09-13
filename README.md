# flight delay predictor project
## over view
departure delay in airports is one of problems that affect in airlines and can decrease the costomers from booking at their flights, and the companies want the more possible flights to gain profits. when I met this problems i think to build this projects to tell the costomers for predicted time of delay, because they can know that rather than unpredicted delays.
this project for predictin the ddeparture delay minutes for flights that use the bts aamerican dataset for g=flights, and the open meteo for weather conditions.
i used some models for training and evaluatin but the lightgbm the best of them and gradientboosting, the measurement used is r2_score that to tell us the model understanding of data, the score is
 approximate 0.34, that is predicted because the problem is regression problem and their are some factors for delays:
 security, weather, aircraft, etc. but the MAE is < 10 that means the the predicted time is near the actual delay time.


## the archetecture:

api:
this module for serving online and batch requests, using fast api in one app to prevent overhead of confegration in education project.
DataPipeline
that used for collecting, processing and make feature engineering in data.
config:
that for config of app recall when the module wants it
models:
for training and registering the champion model and preprocessor
mlartifacts
for storing models and processors

data:
to save the datasets and predictions database
mlflow.db
flow.py
the training script
pyproject.toml
uv.lock
Dockerfile.api
the file for api
Dockerfile.train
docker-compose.yml
readme.md


## dataset:
use dataset from bts and open meteo i merged it together for training and prediction
i faced problem in my device in loading data so i download it in my phone then i transfer it to my device.
the data from 2021 to 2024 but in web site to 2026 may when i download it

## features:
the flights features used:

"Year","Month","DayofMonth","DayOfWeek","FlightDate","Tail_Number","Reporting_Airlin","Origin","OriginCityName","OriginStateName","Dest","DestCityName","DestStateName","CRSDepTime","CRSArrTime","CRSElapsedTime","Distance","DepDelay","ArrDelay","Cancelled","Diverted"
CRSDepTime and CRSArrTime are converted to minutes and hours.
cancelled are use to discard the cancelled flights
time features i converted to sin and cos.
the categorical are encoded to target if it has more values or onehot if it has small values
the weather feature

"temperature_2m","relative_humidity_2m","dew_point_2m","precipitation","wind_speed_10m","wind_direction_10m","surface_pressure","cloud_cover","weather_code"
collect from api


## steps to running locally:

enter the folder of project
open the cmd at that folder

1. running the flow.py
uv run flow.py
to train and registor
after that
run the api
uvicorn api.main:app --reload