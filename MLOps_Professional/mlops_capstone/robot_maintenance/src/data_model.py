from pydantic import BaseModel


class TrainPayload(BaseModel):
    """
    Data model for training payload.

    Attributes:
        file (str): Path to the training data file.
        model_name (str): Name of the model to be trained.
        model_path (str): Path where the trained model will be saved.
        test_size (int, optional): Percentage of data reserved for testing. Defaults to 25.
        ncpu (int, optional): Number of CPU threads used for training. Defaults to 4.
        mlflow_tracking_uri (str): URI for MLFlow tracking.
        mlflow_new_experiment (str, optional): Name of the new experiment to create if no experiment is specified. Defaults to None.
        mlflow_experiment (str, optional): Name of the existing experiment. Defaults to None.
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
        model_name (str): Name of the model to be used for prediction.
        stage (str): Stage of the model to be used for prediction.
        sample (list): List of samples for prediction.
        model_run_id (str): ID of the model run.
        scaler_file_name (str): Name of the scaler file.
        scaler_destination (str, optional): Destination path for the scaler file. Defaults to './'.
    """

    model_name: str
    stage: str
    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = "./"
