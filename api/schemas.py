from pydantic import BaseModel,Field

class FlightRequest(BaseModel):
    Year:int=Field(examples=[2025])
    Month: int=Field(le=12,ge=1)
    DayofMonth: int
    Tail_Number: str
    Reporting_Airline: str
    Origin: str
    Dest: str
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