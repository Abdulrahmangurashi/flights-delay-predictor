from dataclasses import dataclass,field
from typing import Tuple, List, ClassVar
import os
from pathlib import Path

project_root=Path(__file__).resolve().parent.parent

@dataclass
class DataConfig:
    bts_url: str= "https://transtats.bts.gov/PREZIP/"
    bts_prefix = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    airports_info_url="https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
    start_year:int=2022
    end_year:int=2022
    start_month:int=2
    end_month:int=6
    chunk_size=100000
    pred_cols: ClassVar[list[str]]=["Year","Month","DayofMonth","DayOfWeek","FlightDate","Tail_Number","Reporting_Airline","Origin","OriginCityName","OriginStateName","Dest","DestCityName","DestStateName","CRSDepTime","CRSArrTime","CRSElapsedTime","Distance","DepDelay","ArrDelay","Cancelled","Diverted"]
    weather_cols:ClassVar[list[str]]= ["temperature_2m","relative_humidity_2m","dew_point_2m","precipitation","wind_speed_10m","wind_direction_10m","surface_pressure","cloud_cover","weather_code"]
    DepDelayFirst=-60
    DepDelayLast=60

@dataclass
class SplitConfig:
    train_start_year:int=2022
    train_start_month:int=2
    train_end_year:int=2022
    train_end_month:int=4
    val_start_year:int=2022
    val_start_month:int=5
    val_end_year:int=2022
    val_end_month:int=5
    test_start_year:int=202
    test_start_month:int=6
    test_end_year:int=2022
    test_end_month:int=6

@dataclass
class PathConfig:
    basedir = project_root
    data_dir="data"
    model_dir="model"


class ModelConfig:
    random_state=42
    n_jobs=-1
    rf_estimators=300
    rf_max_depth=20
    rf_min_split=5
    gb_estimators=300
    gb_learning_rate=0.05
    tuning_models=["lightgbm"]

@dataclass
class MlflowConfig:
    uri = os.getenv("MLFLOW_TRACKING_URI","http://localhost:1090")
    experiment_name:str="flight delay prediction"
    model_name="FlightDelayPredictor"
    championAlias = "champion"
    challengerAlias="challenger"