from pydantic import BaseModel
 
class TrainPayload(BaseModel):
    file: str
    _model_name: str        # renamed to avoid conflicting with protected namespace
    _model_path: str        # renamed to avoid conflicting with protected namespace
    test_size: int = 25  
    ncpu: int = 4 
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None

class PredictionPayload(BaseModel):
    _model_name: str        # renamed to avoid conflicting with protected namespace
    stage: str
    sample: list
    _model_run_id: str      # renamed to avoid conflicting with protected namespace
    scaler_file_name: str
    scaler_destination: str = './'