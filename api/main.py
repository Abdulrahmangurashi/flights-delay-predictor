from .model_loader import load_model,get_state
from .schemas import FlightRequest, DelayResponse,optional
import pandas as pd
from DataPipeline import DataPreprocessor, FeatureEngineering
from DataPipeline.data_collector import Collector,WeatherDownloader
from config.config import DataConfig,PathConfig
from fastapi import FastAPI,HTTPException
from .BatchParser import init_db
from .batch import router as batch_router
from contextlib import asynccontextmanager

collector = Collector()
airports_info = collector.get_airports_info()
weather_downloader=WeatherDownloader(airports_info=airports_info)

@asynccontextmanager
async def lifespan(app:FastAPI):
        load_model()
        init_db()
        yield

app=FastAPI(title="flight predictor",lifespan=lifespan)
app.include_router(batch_router)
@app.get("/health")
def health():
    s=get_state()
    if s.processor is None or s.model is None:
            raise HTTPException(status_code=503,detail="processor or model does not loaded")
    return {"state":"OK","version":s.version,"alias":s.alias}

@app.get("/optional_data")
def get_optional():
        return optional

@app.post("/predict",response_model=DelayResponse)
def predict(request:FlightRequest):
        s=get_state()
        if not s.model or not s.processor:
                raise HTTPException(status_code=503)
        df=pd.DataFrame([request.model_dump()])
        df["FlightDate"]=pd.Timestamp(year=request.Year,month=request.Month,day=request.DayofMonth)
        df["DayOfWeek"]=df["FlightDate"].dt.day_of_week
        df["DayOfYear"]=df["FlightDate"].dt.day_of_year
        df["CRSDepDateTime"]=pd.to_datetime(
                df["FlightDate"]
                +pd.to_timedelta(df["CRSDepHour"],unit="h")
                +pd.to_timedelta(df["CRSDepMinute"],unit="m")
        )
        weather_df=weather_downloader.get_online_weather(airport_code=request.Origin,date=df.iloc[0]["FlightDate"].strftime("%Y-%m-%d"))
        weather_row=weather_df[weather_df["time"].dt.hour==request.CRSDepHour]
        if weather_row.empty:
                raise HTTPException(status_code=404,detail="weather data is not found")
        weather_row=weather_row.iloc[0]
        for col in DataConfig.weather_cols:
                df[col]=weather_row[col]
                
        hist=FeatureEngineering.HistoricalFeatureEngineering()
        engineered=hist.transform(df)
        processed=s.processor.transform(engineered)
        prediction=s.model.predict(processed)
        return DelayResponse(delay_minutes=prediction[0])