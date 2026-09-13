import mlflow
from mlflow import MlflowClient
from mlflow.sklearn import load_model
from config.config import MlflowConfig
import tempfile
import pickle
from pathlib import Path

class ModelRegister:
    def __init__(self):
        mlflow.set_tracking_uri(MlflowConfig.uri)
        mlflow.set_experiment(MlflowConfig.experiment_name)
        self.client=MlflowClient()
        self.model_name=MlflowConfig.model_name

    def register_model(self,run_id,artifact_path):
        print("registering model")
        model_uri=f"runs:/{run_id}/{artifact_path}"
        return mlflow.register_model(model_uri,name=self.model_name)

    def save_preprocessor(self,run_id,preprocessor):
        temp=Path(tempfile.mkdtemp())
        pkl=temp / "preprocessor.pkl"
        with open(pkl,"wb") as f:
            pickle.dump(preprocessor,f)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(pkl,"preprocessor")

    def get_all_versions(self):
        versions=self.client.search_model_versions(f"name='{self.model_name}'")
        return sorted(versions,key=lambda x:int(x.version))

    def get_versions_by_run_id(self,run_id):
        for version in self.get_all_versions():
            if version.run_id==run_id:
                print(version)
                return version
        return None

    def set_model_alias(self,version,alias):
        try:
            self.client.set_registered_model_alias(self.model_name,alias,version)
            print(f"set alias: {alias} for version")
            return True
        except Exception:
            return False

    def get_metric(self,run_id,metric):
        try:
            run=self.client.get_run(run_id)
            return run.data.metrics[metric]
        except:
            return None

    def get_model_alias(self,alias):
        try:
            return self.client.get_model_version_by_alias(self.model_name,alias)
        except:
            return None

    def get_uri(self,alias):
        return f"models:/{self.model_name}@{alias}"

    def get_model(self,alias):
        model=load_model(self.get_uri(alias))
        return model
    
    def get_preprocessor(self,run_id):
        temp=tempfile.mkdtemp()
        path=mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/preprocessor.pkl",dst_path=temp)
        with open(path,"rb") as f:
            preprocessor=pickle.load(f)
        return preprocessor
    
    def get_production(self):
        version,run_id=self.get_champion()
        model=self.get_model(MlflowConfig.championAlias)
        preprocessor=self.get_preprocessor(run_id)
        return model,preprocessor,version

    def is_champion_exists(self):
        return self.get_model_alias(MlflowConfig.championAlias) is not None
    
    def get_champion(self):
        champion=self.get_model_alias(MlflowConfig.championAlias)
        if not champion:
            return None
        return champion.version,champion.run_id
    
    def set_champion(self,version):
        return self.set_model_alias(version,MlflowConfig.championAlias)
    
    def set_challenger(self,version):
        return self.set_model_alias(version,MlflowConfig.challengerAlias)