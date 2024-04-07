from typing import Any

from pydantic import BaseModel

# nullify protected namespaces so that model_* attributes can be set
if hasattr(BaseModel, "model_config"):
    BaseModel.model_config["protected_namespaces"] = ()
headers = {"Content-Type": "application/json"}


class TrainPayload(BaseModel):
    file: str
    model_name: str
    model_path: str
    test_size: int = 25
    ncpu: int = 4
    mlflow_tracking_uri: str = None
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None


class PredictionPayload(BaseModel):
    sample: list[dict[str, Any]]
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = "./"
    d4p_file_name: str = None
    d4p_destination: str = None
