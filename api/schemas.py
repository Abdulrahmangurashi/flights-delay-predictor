from pydantic import BaseModel,Field
from typing import Literal
import json
from config.config import PathConfig

path = PathConfig.basedir/PathConfig.data_dir/"optional_data.json"
with open(path,"r") as f:
    optional = json.load(f)

reporting_airline = Literal[tuple(optional["Reporting_Airline"])]
tail_number=Literal[tuple(optional["Tail_Number"])]
origin = Literal[tuple(optional["Origin"])]
dest=Literal[tuple(optional["Dest"])]

class FlightRequest(BaseModel):
    Year:int=Field(examples=[2025])
    Month: int=Field(le=12,ge=1)
    DayofMonth: int
    Tail_Number: tail_number
    Reporting_Airline: reporting_airline
    Origin: origin
    Dest: dest
    CRSDepHour: int=Field(ge=0,le=23)
    CRSDepMinute: int=Field(ge=0,le=59)
    CRSArrHour: int=Field(ge=0,le=23)
    CRSArrMinute: int=Field(ge=0,le=59)
    CRSElapsedTime: float
    Distance: float
    Cancelled:int

class DelayResponse(BaseModel):
    delay_minutes: float
    
class BatchResponse(BaseModel):
    state:str
    year:int
    month:int
    message:str