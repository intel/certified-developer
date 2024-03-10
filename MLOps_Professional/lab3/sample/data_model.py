from pydantic import BaseModel
 
class TrainPayload(BaseModel):
    file: str
    name: str
    path: str
    test_size: int = 25  
    ncpu: int = 4 
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None
    