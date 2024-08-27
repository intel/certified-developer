from pydantic import BaseModel

class TrainPayload(BaseModel):
    file: str
    identifier: str
    storage_path: str
    test_size: int = 25  
    ncpu: int = 4 
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None

    class Config:
        from_attributes = True  # Updated from orm_mode for Pydantic V2
