from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler,OneHotEncoder,TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from config.config import ModelConfig
from .data_collector import Collector
from .DataPreprocessor import PreProcessor

class HistoricalFeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self,windows=[3,5,7]):
        self.windows=windows
        self.collector=None

    def fit(self,x,y=None):
        return self

    def transform(self,x:pd.DataFrame)->pd.DataFrame:
        x=x.copy()
        x=x.sort_values(["CRSDepDateTime"]).reset_index(drop=True)
        
        # checking online or batch
        if len(x)==1:
            x=self.transform_online(x).drop("DepDelay",axis=1)
        else:
            x=self.transform_batch(x)

# handling missing values by 0 value
        for col in x.columns:
            if col.startswith("Prev") or col.startswith("Tail_Mean"):
                x[col] = x[col].fillna(0)
        x=x.drop(["CRSDepDateTime","ArrDelay"],axis=1)
        return x

    def create_features(self,df:pd.DataFrame)->pd.DataFrame:
        df=df.copy().drop(["FlightDate","OriginCityName","OriginStateName","DestCityName","DestStateName","airport_code","time"],axis=1,errors="ignore")
        df=df.sort_values(["CRSDepDateTime"]).reset_index(drop=True)
        df["Route"]=df["Origin"]+"_" +df["Dest"]
        df["Origin_Airline"]=df["Reporting_Airline"]+"_"+df["Origin"]
        df["PrevArrDelay"]=df.groupby("Tail_Number",sort=False)["ArrDelay"].shift(1)
        for col in ["Origin","Origin_Airline","Route"]:
            group=df.groupby(col,sort=False)["DepDelay"]
            for window in self.windows:
                df[f"Prev{col}Delay_mean_{window}"] = group.transform(lambda x: x.shift(1).rolling(window,min_periods=1).mean())
                df[f"Tail_Mean_arrdelay_{window}"]=df.groupby("Tail_Number",sort=False)["ArrDelay"].transform(lambda x:x.shift(1).rolling(window,min_periods=1).mean())
        return df

    def previous_month(self,year,month):
        if month ==1:
            return year-1,12
        return year,month -1

    def download_prev_data(self,start_year,start_month,end_year,end_month,until=None):
        self.collector=Collector(start_year,start_month,end_year,end_month)
        df=self.collector.collect()
        if df.shape[0] == 0:
            print(f"the previous month is not loaded successfully")
            return pd.DataFrame()
        processor=PreProcessor()
        df=processor.run(df)
        if until is not None:
            return df[df["CRSDepDateTime"]<until]
        return df
        
    def transform_online(self,x:pd.DataFrame):
        sample=x.copy()
        sample["DepDelay"]=np.nan
        sample["ArrDelay"]=np.nan
        date=pd.to_datetime(sample["CRSDepDateTime"].iloc[0])
        year=date.year
        month=date.month
        prev_year,prev_month=self.previous_month(year,month)
        history_data=self.download_prev_data(prev_year,prev_month,year,month,date)
        data=pd.concat([history_data,sample],ignore_index=True)
        sample=self.create_features(data)
        sample = sample.sort_values("CRSDepDateTime").reset_index(drop=True)
        return sample.iloc[[-1]].reset_index(drop=True)

    def transform_batch(self,x:pd.DataFrame):
        x=x.copy()
        x= x.sort_values("CRSDepDateTime").reset_index(drop=True)
        first=x.iloc[0]["CRSDepDateTime"]
        year=first.year
        month=first.month
        prev_year,prev_month=self.previous_month(year,month)
        data=self.download_prev_data(prev_year,prev_month,prev_year,prev_month,first)
        x=pd.concat([data,x],ignore_index=True)
        x=self.create_features(x)
        return x[x["CRSDepDateTime"] >= first]

    def fit_transform(self,x,y=None):
        self.fit(x)
        return self.transform(x)

        
class FlightDelayFeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self,remove_origin_cols=True):
        self.remove_origin_cols=remove_origin_cols
        
    def fit(self,x,y=None):
        return self

    def _angle_feature(self,col:pd.Series,angles)->pd.Series:
        col_sin= np.sin(2*np.pi * col /angles)
        col_cos=np.cos(2 * np.pi * col/angles)
        return col_sin,col_cos
        
    def transform(self, x:pd.DataFrame)->pd.DataFrame:
        x=x.copy()
        x["Speed"]=x["Distance"] /x["CRSElapsedTime"]
        time_features = {"Month":12,"DayOfWeek":7,"DayOfYear":365.25,"CRSDepHour":24,"CRSDepMinute":60,"CRSArrHour":24,"CRSArrMinute":60}
        for feature,count in time_features.items():
            x[f"{feature}_sin"],x[f"{feature}_cos"]=self._angle_feature(x[feature],count)
        if self.remove_origin_cols:
            x=x.drop(time_features.keys(),axis=1)
        return x
        
    def fit_transform(self, x,y=None):
        self.fit(x)
        return self.transform(x)
    
class OutLayerHandler(BaseEstimator,TransformerMixin):
    def __init__(self,factor):
        self.factor=factor
        self.lower_bounds={}
        self.upper_bounds={}

    def fit(self,x:pd.DataFrame,y=None):
        for col in x.select_dtypes(include=["number"]).columns.to_list():
            q1=x[col].quantile(0.25)
            q3=x[col].quantile(0.75)
            iqr=q3-q1
            self.lower_bounds[col]=q1-self.factor*iqr
            self.upper_bounds[col]=q3+self.factor*iqr
        return self

    def transform(self,x:pd.DataFrame):
        x=x.copy()
        for col in x.select_dtypes(include=["number"]).columns:
            x[col]=x[col].clip(self.lower_bounds[col],self.upper_bounds[col])
        return x
    
    def fit_transform(self,x,y=None):
        self.fit(x)
        return self.transform(x)
    
def pipeline_builder(factor=1.5):
    one_hot_cols=["Reporting_Airline"]
    target_cols=["Origin","Dest","Route","Origin_Airline","Tail_Number"]
    transformer=ColumnTransformer([
        ("onehot",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),one_hot_cols),
        ("target",TargetEncoder(target_type="continuous",smooth="auto",random_state=ModelConfig.random_state),target_cols)
    ],
                                  remainder="passthrough").set_output(transform="pandas")
    pipeline=Pipeline([
        ("feature_engineering",FlightDelayFeatureEngineering()),
    #("outlayer",OutLayerHandler(factor)),
    ("transformer",transformer),
    #("scaler",RobustScaler())
])
    return pipeline