from .model_loader import load_model,get_state
from . import BatchParser
from .schemas import BatchResponse
from fastapi import APIRouter, HTTPException,BackgroundTasks
from contextlib import asynccontextmanager

running_jobs:set=set()

    
router=APIRouter()

def analyse_data(year,month):
    try:
        s=get_state()
        result=BatchParser.score(year,month,s.processor,s.model)
        BatchParser.save_result(result)
    except Exception as e:
        print(e)
    finally:
        running_jobs.discard((year,month))


@router.get("/analyze/{year}/{month}",response_model=BatchResponse)
def analyze_month(year:int,month:int,task:BackgroundTasks):
    key=(year,month)
    if key in running_jobs:
        return BatchResponse(
            state="still processing",year=year,month=month,message=f"{year} {month} flights in the process, after few minutes to get results visit: result/{year}/{month}"
        )
    else:
        running_jobs.add(key)
        task.add_task(analyse_data,year,month)
        return BatchResponse(state=f"start {year} {month} processing",year=year,month=month,message=f"the operation is started, wait few minutes and visit result/{year}/{month}")

@router.get("/result/{year}/{month}")
def get_result(year:int,month:int):
    result=BatchParser.get_result(year,month)
    if not result:
        return []
    return result

@router.get("/results")
def get_results():
    results=BatchParser.get_results()
    if not results:
        return []
    return results

@router.get("/onpars")
def running():
    return running_jobs