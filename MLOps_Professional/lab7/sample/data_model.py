from pydantic import BaseModel


class TrainPayload(BaseModel):
    """
    Data model for training payload.

    Attributes:
        file (str): Path to the training data file.
        model_name (str): Name of the model to be trained.
        model_path (str): Path where the trained model will be saved.
        test_size (int): Percentage of data reserved for testing. Defaults to 25.
        ncpu (int): Number of CPU threads used for training. Defaults to 4.
        mlflow_tracking_uri (str): URI for MLFlow tracking.
        mlflow_new_experiment (str): Name of the new experiment to create if no experiment is specified. Defaults to None.
        mlflow_experiment (str): Name of the existing experiment. Defaults to None.
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
        sample (list): List of samples for prediction.
        model_run_id (str): ID of the model run.
        scaler_file_name (str): Name of the scaler file.
        scaler_destination (str): Destination path for the scaler file. Defaults to './'.
        d4p_file_name (str): Name of the d4p file.
        d4p_destination (str): Destination path for the d4p file.
    """

    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = "./"
    d4p_file_name: str
    d4p_destination: str
