import streamlit as st
import requests
import pandas as pd
import os
from datetime import date

base_url =os.getenv("api_url","http://localhost:1091")
st.set_page_config(page_title="flights delay predictor",page_icon="✈️",layout = "wide")

@st.cache_data(ttl=300)
def get_data():
    resp=requests.get(f"{base_url}/optional_data")
    resp.raise_for_status()
    return resp.json()
    
def get_health():
    try:
        resp=requests.get(f"{base_url}/health")
        return resp.ok, resp.json()
    except Exception as e:
        return False, {"error":str(e)}


is_healthy, health_data=get_health()
if not is_healthy:
    st.error("an error in api or connection")
    st.stop()

st.caption("the service is running now")
tap1,tap2,tap3=st.tabs(["flight predictor","batch prediction","monitor"])
with tap1:
    st.subheader("the online flight predictor")
    optional=get_data()

    col1,col2,col3=st.columns(3)
    with col1:
        FlightDate=st.date_input("the flight date",date.today())
            
    with col2:
        Airline=st.selectbox("airline",optional["Reporting_Airline"])
        Tail_Number=st.selectbox("tail number",optional["Tail_by_Airline"][Airline])
        Origin = st.selectbox("origin",optional["Origin"])
        Dest = st.selectbox("destination",optional["Dest"])
        Distance = st.number_input("distance")
            
    with col3:
        CRSDepHour=st.slider("departure hour",0,23,13)
        CRSDepMinute=st.slider("departure minute",0,59,1)
        CRSArrHour=st.number_input("arrival hour",0,23,13)
        CRSArrMinute=st.number_input("arrival minute",0,59,13)
        CRSElapsedTime = st.number_input("elapse time",min_value=1,value=50)

    if st.button("predict",use_container_width=True):
        if Origin == Dest:
            st.error("origin and destination doesn;t be same")
        else:
            data={
            "Year":FlightDate.year,
            "Month":FlightDate.month,
            "DayofMonth":FlightDate.day,
            "Reporting_Airline":Airline,
            "Tail_Number":Tail_Number,
            "Origin":Origin,
            "Dest":Dest,
            "Distance":Distance,
            "CRSDepHour":CRSDepHour,
            "CRSDepMinute":CRSDepMinute,
            "CRSArrHour":CRSArrHour,
            "CRSArrMinute":CRSArrMinute,
            "CRSElapsedTime":CRSElapsedTime,
            "Cancelled":0
        }
            resp=requests.post(f"{base_url}/predict",json=data)
        if resp.status_code!=200:
            st.error(f"{resp.status_code}:{resp.text}")
        else:
            st.write(f"the delay minutes is {resp.json()['delay_minutes']}")

with tap2:
    st.subheader("the batch predictor")
    col1,col2=st.columns(2)
    with col1:
        pred_year=st.number_input("year",min_value=2020,max_value=2030,step=1)

    with col2:
        pred_month=st.number_input("month",min_value=1,max_value=12,step=1)
        
    if st.button("batch_predict",use_container_width=True):
        resp=requests.get(f"{base_url}/analyze/{pred_year}/{pred_month}")
        st.write(resp.json())

    if st.button("show result",use_container_width=True):
        resp=requests.get(f"{base_url}/result/{pred_year}/{pred_month}")
        result=resp.json()
        if not result:
            st.subheader("their is not predicted data, predict first and show result")
        else:
            colA,colB,colC=st.columns(3)
            with colA:
                st.write(f"the total smples: {result["total_rows"]}")
            with colB:
                st.write(f"the delay mean: {result["delay_mean"]}")
            with colC:
                st.write(f"the mean errors: {result["mae"]}")
        
with tap3:
    if st.button("update data"):
        get_data().clear()
        st.rerun()
        
    resp=requests.get(f"{base_url}/results")
    try:
        results=resp.json()
        if not results:
            st.error("no predictions yet, please analyze firstly and show the results")
        else:
            result_df=pd.DataFrame(resp.json())
            result_df["year_month"]=result_df["year"].astype(str)+"-" +result_df["month"].astype(str).str.zfill(2)
            filter=st.selectbox("the month",["all"]+result_df["year_month"].tolist())
            display_df=result_df if filter=="all" else result_df[result_df["year_month"]==filter]
            st.dataframe(display_df[["year_month","delay_mean","distance_mean","mae"]])
            st.line_chart(display_df,x="year_month",y="mae")
    except Exception as e:
        st.error("error in connection")