from config.config import DataConfig,SplitConfig
import numpy as np
import pandas as pd

class PreProcessor:
    def __init__(self):
        self.config=DataConfig        
    def clean_data(self,dataframe:pd.DataFrame)->pd.DataFrame:
        clean=dataframe.copy()
    
        # choose not cancelled trip
        clean=clean[clean["Cancelled"] == 0]
        
        # not diverted trip
        clean = clean[clean["Diverted"]==0]
        
        # remove unusual delay time
        clean=clean[(clean["DepDelay"] >=self.config.DepDelayFirst) & (clean["DepDelay"]<=self.config.DepDelayLast)]
        # extracting date features from flightdate
        clean["FlightDate"] = pd.to_datetime(clean["FlightDate"])
        clean["DayOfYear"] = clean["FlightDate"].dt.day_of_year
        clean["time"]=pd.to_datetime(clean["time"])


    # parsing dept and arr time to minutes and hours
        clean["CRSDepHour"],clean["CRSDepMinute"]=self.parsing_time(clean["CRSDepTime"])
        clean["CRSArrHour"],clean["CRSArrMinute"] = self.parsing_time(clean["CRSArrTime"])
        
        # remove missing rows
        clean.dropna(inplace=True)
        clean=clean.drop_duplicates()
        cols= ["Year","Month","DayofMonth","DayOfWeek","DayOfYear","Tail_Number","Reporting_Airline","Origin","Dest","CRSDepDateTime","CRSDepHour","CRSDepMinute","CRSArrHour","CRSArrMinute","CRSElapsedTime","Distance","DepDelay","ArrDelay","Cancelled","temperature_2m","relative_humidity_2m","dew_point_2m","precipitation","wind_speed_10m","wind_direction_10m","surface_pressure","cloud_cover","weather_code"]

        clean=clean[cols]
        clean=clean.reset_index(drop=True)
        return clean

    def parsing_time(self,time:int)->tuple[int,int]:
        hour=time//100
        minute = time%100
        return hour,minute

    def run(self,dataframe:pd.DataFrame)->pd.DataFrame:
        return self.clean_data(dataframe)
    
class DataSplitter:
    def __init__(self,):
        self.dataframe=None

    def get_monthes(self,start_year,start_month,end_year,end_month):
        periods = pd.period_range(start=f"{start_year}-{start_month}",end=f"{end_year}-{end_month}",freq="M")
        return [(period.year,period.month) for period in periods]

    def get_df(self, start_year,start_month,end_year,end_month)->pd.DataFrame:
        return pd.concat([self.dataframe[(self.dataframe["Year"]== year)& (self.dataframe["Month"]==month)] for year, month in self.get_monthes(start_year,start_month,end_year,end_month)],ignore_index=True)

    def get_features_target(self,dataframe:pd.DataFrame,target:str)->tuple[pd.DataFrame,pd.DataFrame]:
        features=dataframe.drop(target,axis=1)
        target=dataframe[target]
        return features,target

    def run(self,dataframe:pd.DataFrame,target="DepDelay",train_start_year=SplitConfig.train_start_year,train_start_month=SplitConfig.train_start_month,train_end_year=SplitConfig.train_end_year,train_end_month=SplitConfig.train_end_month,val_start_year=SplitConfig.val_start_year,val_start_month=SplitConfig.val_start_month,val_end_year=SplitConfig.val_end_year,val_end_month=SplitConfig.val_end_month,test_start_year=SplitConfig.test_start_year,test_start_month=SplitConfig.test_start_month,test_end_year=SplitConfig.test_end_year,test_end_month=SplitConfig.test_end_month):
        self.dataframe=dataframe
        train_df=self.get_df(train_start_year,train_start_month,train_end_year,train_end_month)
        val_df=self.get_df(val_start_year,val_start_month,val_end_year,val_end_month)
        test_df=self.get_df(test_start_year,test_start_month,test_end_year,test_end_month)
        x_train,y_train=self.get_features_target(train_df,target)
        x_val,y_val=self.get_features_target(val_df,target)
        x_test,y_test=self.get_features_target(test_df,target)
        return x_train,x_val,x_test,y_train,y_val,y_test