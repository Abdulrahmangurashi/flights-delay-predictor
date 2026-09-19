from DataPipeline.data_collector import Collector
from DataPipeline.DataPreprocessor import PreProcessor, DataSplitter
from config.config import MlflowConfig,PathConfig
from DataPipeline.FeatureEngineering import pipeline_builder, HistoricalFeatureEngineering
from sklearn.metrics import r2_score
from models.ModelTrainer import Trainer,model_stage
from models.register import ModelRegister
from prefect import flow,task
import pandas as pd
import numpy as np
import json

#@task(name="collect",retries=3,retry_delay_seconds=30)
def collect_data():
    print("start collecting data")
    collector = Collector()
    df=collector.collect()
    print(f"collecting: {df.shape[0]} samples")
    return df

#@task(name="preprocesseer")
def processing_data(df):
    print("start data preprocessing")
    preProcessor=PreProcessor()
    df=preProcessor.run(dataframe=df)
    print(f"data preprocessing: {df.shape[0]} samples")
    return df

#@task(name="split")
def split_data(df):
    historicalFeature=HistoricalFeatureEngineering()
    df=historicalFeature.transform(df)
    splitter=DataSplitter()
    x_train,x_val,x_test,y_train,y_val,y_test=splitter.run(dataframe=df)
    print(f"x train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"x val shape: {x_val.shape}")
    print(f"y train shape: {y_val.shape}")
    print(f"x test shape: {x_test.shape}")
    print(f"y test shape: {y_test.shape}")
    return x_train,x_val,x_test,y_train,y_val,y_test

#@task(name="save_important")
def save_optional_cat_data(x_train:pd.DataFrame):
    df=x_train.copy()
    data={}
    for col in ["Reporting_Airline","Tail_Number","Origin","Dest"]:
        data[col] = df[col].dropna().unique().tolist()

    data["Tail_by_Airline"] = df.dropna(subset=["Reporting_Airline","Tail_Number"]).groupby("Reporting_Airline")["Tail_Number"].unique().apply(lambda x: x.tolist()).to_dict()

    path=PathConfig.basedir/PathConfig.data_dir/"optional_data.json"
    with open(path,"+w") as f:
        json.dump(data,f,indent=2)
    print(f"the origin, dest,airline,tail_numbe saved at {str(path)}")

#@task(name="pipeline")
def data_pipeline(x_train,x_val,x_test,y_train):
    pipeline=pipeline_builder()
    x_train_processed=pipeline.fit_transform(x_train,y_train)
    x_val_processed=pipeline.transform(x_val)
    x_test_processed=pipeline.transform(x_test)
    return x_train_processed,x_val_processed,x_test_processed,pipeline

#@task(name="train_model",retries=2,retry_delay_seconds=15)
def train_model(model,x_train,y_train,x_val,y_val,model_name):
    trainer=Trainer()
    result,trained_model,run_id=trainer.train_and_validate(model,x_train,y_train,x_val,y_val,model_name)
    return {
        "model_name":model_name,
        "metrics":result,
        "model":trained_model,
        "run_id":run_id
    }

#@task(name="select_best")
def search_best_model(results):
    model_name=max(results,key=lambda name:results[name]["metrics"]["r2_val"])
    best_model=results[model_name]
    return best_model

#@task(name="tune_model")
def tune_model(best_model,x_train,y_train,x_val,y_val):
    trainer=Trainer()
    result=trainer.model_tuning(best_model["model"],x_train,y_train,best_model["model_name"],best_model["metrics"]["r2_val"])
    if len(result)==2:
        return best_model["model"],best_model["run_id"]
    metrics,tuned_model,run_id=result
    if metrics["improvement"] > 0:
        return tuned_model,run_id
    return best_model["model"],best_model["run_id"]

#@task(name="final_test")
def test_model(model,x_test,y_test,run_id):
    trainer=Trainer()
    test_metrics=trainer.evaluate_and_test(model,x_test,y_test,run_id)
    return test_metrics

#@task(name="register_model")
def register_model(run_id,pipeline):
    
    print("registering model")
    model_register=ModelRegister()
    model_register.save_preprocessor(run_id,pipeline)
    model =model_register.register_model(run_id,"model")
    if not model_register.is_champion_exists():
        model_register.set_champion(model.version)
        champ_version,champ_run=model_register.get_champion()
    else:
        champ_version,champ_run = model_register.get_champion()
        if model_register.get_metric(model.run_id,"r2_val") > model_register.get_metric(champ_run,"r2_val"):
            model_register.set_champion(model.version)
            champ_version,champ_run=model_register.get_champion()
        else:
            model_register.set_challenger(model.version)
    print(f"the champion version:{champ_version} \n the champion run id: {champ_run}")
    return model_register.get_uri(MlflowConfig.championAlias),model_register.get_model(MlflowConfig.championAlias)

#@flow(name="flight_delay_pipeline")
def flight_delay_pipeline(tune=False):
    df=collect_data()
    df=processing_data(df)
    x_train,x_val,x_test,y_train,y_val,y_test=split_data(df)
    save_optional_cat_data(x_train)
    x_train_processed,x_val_processed,x_test_processed,pipeline=data_pipeline(x_train,x_val,x_test,y_train)
    trained_models={}
    for model_name,model in model_stage().items():
        print(model_name)
        result=train_model(model,x_train_processed,y_train,x_val_processed,y_val,model_name)
        trained_models[model_name]=result

    best_model = search_best_model(trained_models)
    final_model=best_model["model"]
    final_run_id=best_model["run_id"]
    if tune:
        tuned_model,tune_run=tune_model(best_model,x_train_processed,y_train,x_val_processed,y_val)
        final_model=tuned_model
        final_run_id=tune_run

    test_metrics=test_model(final_model,x_test_processed,y_test,final_run_id)

    uri,model=register_model(final_run_id,pipeline)
    return best_model

flight_delay_pipeline()
