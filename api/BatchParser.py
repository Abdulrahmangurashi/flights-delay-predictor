from DataPipeline.data_collector import Collector
from DataPipeline.DataPreprocessor import PreProcessor
from DataPipeline.FeatureEngineering import HistoricalFeatureEngineering
import sqlite3
import pandas as pd
from pathlib import Path
from config.config import PathConfig
from sklearn.metrics import mean_absolute_error

data_dir=Path(PathConfig.basedir) / PathConfig.data_dir
prediction_dir=data_dir/"predictions"
db_path=data_dir/"PredictionsResults.db"

prediction_dir.mkdir(parents=True,exist_ok=True)
def init_db():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("create table if not exists results (year integer,month integer, total_rows integer, delay_mean real, distance_mean real, mae real, output_path text, primary key (year,month))")
    conn.commit()
    conn.close()

def score(year,month,processor,model):
    collector = Collector(year,month,year,month)
    df = collector.collect()
    cleaner = PreProcessor()
    df=cleaner.run(df)
    hist=HistoricalFeatureEngineering()
    df = hist.transform(df)
    actual_delay=df["DepDelay"]
    features=df.drop("DepDelay",axis=1)
    features_transformed=processor.transform(features)
    predicted_delay=model.predict(features_transformed)
    mae=mean_absolute_error(actual_delay,predicted_delay)
    df["Actual_delay"]=actual_delay
    df["Predicted_Delay"]=predicted_delay
    path=prediction_dir/f"{year}_{month}_predictions.csv"
    df.to_csv(path,index=False)
    results={
        "year":year,
        "month":month,
        "total_rows":len(features),
        "delay_mean":predicted_delay.mean(),
        "distance_mean":features["Distance"].mean(),
        "mae":mae,
        "output_path":str(path)
    }
    return results

def save_result(result:dict):
    conn = sqlite3.connect(db_path)
    cur=conn.cursor()
    cur.execute("insert or replace into results values (?,?,?,?,?,?,?)",(result["year"],result["month"],result["total_rows"],result["delay_mean"],result["distance_mean"],result["mae"],result["output_path"]))
    conn.commit()
    conn.close()

def get_results():
    conn = sqlite3.connect(db_path)
    cur=conn.cursor()
    cur.row_factory = sqlite3.Row
    cur.execute("select * from results order by year,month")
    result=cur.fetchall()
    conn.close()
    return result

def get_result(year,month):
    conn=sqlite3.connect(db_path)
    cur=conn.cursor()
    cur.row_factory=sqlite3.Row
    cur.execute(f"select * from results where year = ? and month = ?",(year,month))
    result=cur.fetchone()
    conn.close()
    return result