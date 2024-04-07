from dataclasses import dataclass


@dataclass
class TrainPayload:
    file: str
    model_name: str
    model_path: str
    test_size: int = 25  
    ncpu: int = 4 
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None


@dataclass
class PredictionPayload:
    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = './'
    d4p_file_name: str
    d4p_destination: str
