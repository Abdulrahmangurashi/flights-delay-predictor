"""
data_collector:
this module for collecting flights and weather data for specific period of time

classes:
Collector:
for specifying the period of time to download data blong to it
has attributes from FlightDownoader and weather downloader
merge flight data and weather data the near to it for the origin until 1 hour to flight
return data in pandas dataframe

Downloader:
the base class for downloaders: flight and weather
and check the existing of files to prevent downloading again

FlightDownloader:
the class use bts archive data to download the flights data

WeatherDownloader:
can download the data for monthes from archive or online weather from forcast api
"""
from config.config import DataConfig, PathConfig
import requests
import os
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
from calendar import monthrange
import time
import logging

logger=logging.getLogger(__file__)
class Collector:
    def __init__(self,start_year:int=DataConfig.start_year,start_month:int = DataConfig.start_month, end_year:int=DataConfig.end_year,end_month:int=DataConfig.end_month, chunk_size:int=DataConfig.chunk_size):
        """the entry point of class for downloading and and merging data

        Args:
            start_year (int, optional): the start year of period to download. Defaults to DataConfig.start_year.
            start_month (int, optional): the start month. Defaults to DataConfig.start_month.
            end_year (int, optional): the end year of period. Defaults to DataConfig.end_year.
            end_month (int), optional): the last month. Defaults to DataConfig.end_month.
            chunk_size (_type_, optional): _description_. Defaults to DataConfig.chunk_size.
        """
        self.start_year=start_year
        self.start_month=start_month
        self.end_year=end_year
        self.end_month=end_month
        self.chunk_size=chunk_size
        self.flight_downloader=FlightsDownloader() # the object to download flights data
        self.weather_downloader=WeatherDownloader() # weather data downloader
        self.files=[]
        self.dirname=Path(PathConfig.basedir)/PathConfig.data_dir # the data dirrectory path

    def csv_extracter(self,path):
        dirname=os.path.abspath(os.path.dirname(path))
        csv_file=""
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    print(f"extracting {member}")
                    zf.extract(member,dirname)
                    csv_file=os.path.join(dirname,member)
                    #Path(path).unlink()
        return csv_file

    def get_airports_info(self):
        """this method to loading airports information for use in merge weather and specify the cordinates

        Returns:
the airport code
latitude
longitude
        """
        logger.info("downloading airports data ")
        
        # the path of airports data to flights and weather
        airports_path=self.dirname/"airports.csv"
        logger.info(f"at {airports_path}")

# check if airports data to don't repeat downloading it
        if os.path.exists(airports_path):
            logger.info("airports is already exists")
            df=pd.read_csv(airports_path)
        else:
            try:
                logger.info("start downloading airprots data")
                df=pd.read_csv(DataConfig.airports_info_url)
                logger.info("the loading is finished")
            except:
                logger.error("an error occured in loading data, check the url or network")
                return None
            df.to_csv(airports_path,index=False)
        return df[["iata_code","latitude_deg","longitude_deg"]]

    def create_flight_weather_file(self,flight_df:pd.DataFrame,weather_df:pd.DataFrame,year:int,month)->pd.DataFrame:
        """this method to merging flights data and weather  data to make dataset for using, the nearest time

        Args:
            flight_df (pd.DataFrame): 
            weather_df (pd.DataFrame): 
            year (int): the year of datasets that used to folder of year and month to store
            month (int)

        Returns:
            the merged dataset
        """
        logger.info("merging data will start")
        # dropping null values in dataframes
        flight_df.dropna(subset=["CRSDepDateTime","Origin"],inplace=True)
        weather_df.dropna(subset=["time","airport_code"],inplace=True)
        
        # converting times columns to datatime objects
        flight_df["CRSDepDateTime"]=pd.to_datetime(flight_df["CRSDepDateTime"])
        weather_df["time"]=pd.to_datetime(weather_df["time"])

# sorting data
        flight_df=flight_df.sort_values(["CRSDepDateTime","Origin"])
        weather_df=weather_df.sort_values(["time","airport_code"])
        
        # merging data
        merge_df=pd.merge_asof(flight_df,weather_df,left_on="CRSDepDateTime",right_on="time",left_by="Origin",right_by="airport_code",direction="backward",tolerance=pd.Timedelta("1h"))
        
        # save data
        merge_df.to_csv(os.path.join(self.dirname,str(year),str(month),"flight_weather.csv"),index=False)
        logger.info("the merged data is saved")
        return merge_df

    def get_months(self):
        """
        for specifying the year and month of the period
        """
        periods = pd.period_range(start=f"{self.start_year}-{self.start_month}",end=f"{self.end_year}-{self.end_month}",freq="M")
        return [(period.year, period.month) for period in periods]
    
    def load_csv_file(self,path, use_cols=DataConfig.pred_cols)-> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path,usecols=use_cols,chunksize=self.chunk_size)

    def collect(self):
        chunks=[]
        # looping through monthes to loading their data and adding to the returned dataframe
        for year,month in self.get_months():
            logger.info(f"{year} , {month} processing")
            # loading data file with zip extension
            zip_path=self.flight_downloader.data_downloader(year,month)
            
            if zip_path is None:
                logger.error(f"the {year} {month} is not loaded succesfully")
                chunks.append(pd.DataFrame())
                continue
            # extracting csv files from zip
            csv_file=self.csv_extracter(zip_path)
            logger.info(f"parsing {csv_file}")
            origins=set() # the set for airports to help in weather data
            
            # the flights data
            flight_df=self.flight_downloader.load_csv_file(csv_file,use_cols=DataConfig.pred_cols)
            flight_df["CRSDepDateTime"]=pd.to_datetime(flight_df["FlightDate"],errors="coerce")+pd.to_timedelta(flight_df["CRSDepTime"]//100,unit="h")+pd.to_timedelta(flight_df["CRSDepTime"]%100,unit="m")
            
            logger.info("loaded flights data")
            origins.update(flight_df["Origin"].dropna().unique()) # saving all origin names
            
            # getting airports information for flights
            logger.info("airports informations")
            airports_info=self.get_airports_info()
            self.weather_downloader.airports_info=airports_info[airports_info["iata_code"].isin(origins)]
            
            # loading weather data
            logger.info("will loading weather data")
            weather_file=self.weather_downloader.data_downloader(year,month)
            weather_df=self.weather_downloader.load_csv_file(weather_file,DataConfig.weather_cols+["airport_code","time"])
            
            # merging data
            flight_weather_df=self.create_flight_weather_file(flight_df,weather_df,year,month)
            logger.info("merged")
        
            chunks.append(flight_weather_df)
        return pd.concat(chunks,ignore_index=True)

class Downloader:
    """the base downloader class
    """
    def __init__(self,url:str,filename:str,extension:str):
        self.url=url
        self.filename=filename
        self.extension=extension
        self.data_dir=os.path.join(PathConfig.basedir,PathConfig.data_dir)

# checking file
    def is_file_exists(self,month_dir:str,filename:str)->bool:
        return os.path.exists(os.path.join(month_dir,filename))

    def data_downloader(self,year:int,month:int):
        month_dir=os.path.join(self.data_dir,str(year),str(month))
        filename=f"{self.filename}{self.extension}"
        if self.is_file_exists(month_dir=month_dir,filename=filename):
            print(f"{filename} already exists")
            return os.path.join(month_dir,filename)
        os.makedirs(month_dir,exist_ok=True)
        print("month dir: ",month_dir,"file name",filename)
        return self.download(month_dir,filename,year,month)

    def download(self,month_dir,filename,year,month):
        pass

    def load_csv_file(self,path, use_cols)-> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path,usecols=use_cols)



class FlightsDownloader(Downloader):
    def __init__(self, url=DataConfig.bts_url, prefix=DataConfig.bts_prefix, filename="flights",extension=".zip"):
        super().__init__(url,filename,extension)
        self.url = url
        self.prefix= prefix
        
    def download(self,month_dir,filename,year,month):
        url=f"{self.url}/{self.prefix}_{year}_{month}{self.extension}"
        try:
            response = requests.get(url=url,stream=True,timeout=120)
            response.raise_for_status()
        except:
            logger.error("error in loading flights data")
            return None
        logger.info(f"{self.filename}will start  downloading")
        os.makedirs(month_dir,exist_ok=True)
        file_path=os.path.join(month_dir,filename)
        with open(file_path,"wb") as file:
            for chunk in response.iter_content(chunk_size=DataConfig.chunk_size):
                if chunk:
                    file.write(chunk)
            print(f"{filename} saved")
        return file_path

    def parse_time(self,time):
        return time//100,time%100

class WeatherDownloader(Downloader):
    def __init__(self, url=DataConfig.weather_url, filename="weather", extension=".csv",airports_info=None,weather_cols=DataConfig.weather_cols):
        super().__init__(url, filename, extension)
        self.airports_info = airports_info
        self.weather_cols=weather_cols
        
    def download(self, month_dir, filename, year, month):
        filename=f"{self.filename}{self.extension}"
        file_path=os.path.join(month_dir,filename)
        last_month_day=monthrange(year,month)[1]
        batch_size=100
        airports_codes,lats,longs=self.get_coordinates()
        frames=[]
        for start in range(0,len(self.airports_info),batch_size):
            end=start+batch_size
            codes=airports_codes[start:end]
            params={
                "latitude":",".join(map(str,lats[start:end])),
                "longitude":",".join(map(str,longs[start:end])),
                "start_date":f"{year}-{month:02d}-01",
                "end_date":f"{year}-{month:02d}-{last_month_day}",
                "hourly":",".join(self.weather_cols),
                "timezone":"auto"}
            response = requests.get(url=self.url,params=params)
            response.raise_for_status()
            data=response.json()
            for code, location_data in zip(codes,data):
                df=pd.DataFrame(location_data["hourly"])
                df.insert(0,"airport_code",code)
                df["time"]=pd.to_datetime(df["time"])
                frames.append(df)
            time.sleep(30)
        dataframe=pd.concat(frames,ignore_index=True)
        dataframe.to_csv(file_path,index=False)
        print(f"{filename} saved")
        return file_path

    def get_online_weather(self, airport_code, date):

        airport = self.airports_info[self.airports_info["iata_code"] == airport_code]

        if airport.empty:
            raise ValueError(f"Airport {airport_code} not found")

        latitude = airport["latitude_deg"].iloc[0]
        longitude = airport["longitude_deg"].iloc[0]

        params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(self.weather_cols),
        "timezone": "auto"
    }

        response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=30
    )

        response.raise_for_status()

        data = response.json()

        weather_df = pd.DataFrame(data["hourly"])

        weather_df["time"] = pd.to_datetime(weather_df["time"])

        weather_df["airport_code"] = airport_code

        return weather_df

    def get_coordinates(self):
        airports_codes=self.airports_info["iata_code"].tolist()
        lats=self.airports_info["latitude_deg"].tolist()
        longs=self.airports_info["longitude_deg"].tolist()
        return airports_codes, lats,longs