from pydantic import BaseModel


class TrainPayload(BaseModel):
    """
    Data model for holding the parameters required to train a machine learning model.

    Attributes:
        file (str): The path to the training data file.
        model_name (str): The name of the machine learning model.
        model_path (str): The path where the trained model will be saved.
        test_size (int, optional): The size of the test dataset as a percentage. Default is 25.
        ncpu (int, optional): The number of CPU cores to use for training. Default is 4.
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
