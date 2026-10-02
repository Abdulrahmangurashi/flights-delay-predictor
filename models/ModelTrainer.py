from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from lightgbm import LGBMRegressor
import time
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from mlflow.sklearn import log_model
from config.config import MlflowConfig, ModelConfig

def model_stage():
    models={
        "linear regressor":LinearRegression(),
        "lasso":Lasso(),
        "ridge":Ridge(),
 "elastic net":ElasticNet(),
        "random forest":RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_split=5,random_state=42,n_jobs=-1),
        "gradient boosting":GradientBoostingRegressor(),
        "lightgbm":LGBMRegressor(n_estimators=500,random_state=42,learning_rate=0.05,n_jobs=-1)
    }
    return models

class Trainer:
    def __init__(self):
        self.models=model_stage()
        mlflow.set_tracking_uri(MlflowConfig.uri)
        mlflow.set_experiment(MlflowConfig.experiment_name)

    def _predict(self,model,x,y,prefix):
        y_pred=model.predict(x)
        metrics={
            f"rmse_{prefix}":np.sqrt(mean_squared_error(y,y_pred)),
            f"r2_{prefix}":r2_score(y,y_pred),
            f"mae_{prefix}":mean_absolute_error(y,y_pred)
        }
        return y_pred,metrics

    def _log_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            column_name = "feature_importance"

        elif hasattr(model, "coef_"):
            values = np.ravel(model.coef_)
            column_name = "coefficient"

        else:
            return

        df = pd.DataFrame({
        "feature": feature_names,
        column_name: values
    })

        # ترتيب من الأكبر للأصغر
        df["absolute_value"] = df[column_name].abs()
        df = df.sort_values("absolute_value",ascending=False).drop(columns="absolute_value")

        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / "feature_importance.csv"

        df.to_csv(file_path, index=False)

        mlflow.log_artifact(
        str(file_path),
        artifact_path="feature_importance"
    )

    def train_and_validate(self,model,x_train,y_train,x_val,y_val,model_name):
        with mlflow.start_run(run_name=model_name) as run:
            start_time=time.time()
            model.fit(x_train,y_train)
            training_time=time.time()-start_time
        
            y_pred,train_metrics=self._predict(model,x_train,y_train,"train")
            y_val_pred,val_metrics=self._predict(model,x_val,y_val,"val")

            metrics={
            **train_metrics,
            **val_metrics,
            "overfiting_gap":train_metrics["r2_train"]-val_metrics["r2_val"],
            "training_time":training_time
        }
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())
            self._log_feature_importance(model,x_train.columns)
            signature=infer_signature(x_train,y_pred)
            log_model(model,artifact_path="model",signature=signature,serialization_format="pickle")
        [print(f"{key}: {val}") for key,val in metrics.items()]
        return metrics,model, run.info.run_id
    
    def model_tuning(self,model,x_train,y_train,x_val,y_val,model_name,measure):
        if model_name not in ModelConfig.tuning_models:
            return model, None
        params_grid ={
            "lightgbm":{
                "n_estimators":[100, 200, 300, 400],
                "max_depth":[5,10,15,20,25,30],
                "learning_rate":[0.01,0.03,0.05,0.08,0.1]
            }
        }
        search=GridSearchCV(model,params_grid[model_name],scoring="r2",cv=TimeSeriesSplit(n_splits=5),n_jobs=-1)
        with mlflow.start_run(run_name=f"{model_name}_tuned")as run:
            start_time=time.time()
            search.fit(x_train,y_train)
            tuning_time=time.time()-start_time
            best_model=search.best_estimator_
            y_pred,train_metrics=self._predict(best_model,x_train,y_train,"train")
            y_val_pred,val_metrics=self._predict(best_model,x_val,y_val,"val")
            metrics={
                **train_metrics,
                **val_metrics,
                "improvement":val_metrics["r2_val"]-measure
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_score",search.best_score_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("tuning_time",tuning_time)
            self._log_feature_importance(best_model,x_train.columns)
            signature=infer_signature(x_train,y_pred)
            log_model(best_model,artifact_path="model",signature=signature,serialization_format="pickle")
        return metrics, best_model, run.info.run_id

    def evaluate_and_test(self,model,x_test,y_test,run_id):
        with mlflow.start_run(run_id=run_id):
            y_test_pred,metrics=self._predict(model,x_test,y_test,"test")
            mlflow.log_metrics(metrics)
        return metrics