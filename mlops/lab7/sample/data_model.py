from pydantic import BaseModel
 
class TrainPayload(BaseModel):
    file: str
    model_name: str
    model_path: str
    test_size: int = 25  
    ncpu: int = 4 
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None

class PredictionPayload(BaseModel):
    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = './'
    d4p_file_name: str
    d4p_destination: str