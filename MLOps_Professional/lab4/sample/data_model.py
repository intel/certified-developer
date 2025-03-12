from pydantic import BaseModel


class TrainPayload(BaseModel):
    """
    Data model for holding training configuration parameters.

    Attributes:
        file (str): The path to the training data file.
        model_name (str): The name of the model to be trained.
        model_path (str): The path where the trained model will be saved.
        test_size (int): The percentage of the data to be used for testing. Default is 25.
        ncpu (int): The number of CPU cores to be used for training. Default is 4.
        mlflow_tracking_uri (str): The URI for the MLflow tracking server.
        mlflow_new_experiment (str, optional): The name of the new MLflow experiment. Default is None.
        mlflow_experiment (str, optional): The name of the existing MLflow experiment. Default is None.
    """

    file: str
    model_name: str
    model_path: str
    test_size: int = 25
    ncpu: int = 4
    mlflow_tracking_uri: str
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None


class PredictionPayload(BaseModel):
    """
    Data model for prediction payload.
    Attributes:
        model_name (str): The name of the model to be used for prediction.
        stage (str): The stage of the model (e.g., 'development', 'production').
        sample (list): The input data sample for which prediction is to be made.
        model_run_id (str): The unique identifier for the model run.
        scaler_file_name (str): The name of the scaler file used for preprocessing.
        scaler_destination (str): The destination path where the scaler file is stored. Default is './'.
    """

    model_name: str
    stage: str
    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = "./"
