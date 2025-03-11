#!/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error

"""
Module to train and prediction using XGBoost Classifier
"""

import os
import sys
import logging
import warnings
import joblib
import mlflow
from werkzeug.utils import secure_filename
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab3")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class HarvesterMaintenance:

    def __init__(self, model_name: str):
        """
        Initializes the model with the given model name and sets up various attributes.

        Args:
            model_name (str): The name of the model to be initialized.

        Attributes:
            model_name (str): The name of the model.
            file (str): Placeholder for file path or file name.
            y_train (str): Placeholder for training labels.
            y_test (str): Placeholder for test labels.
            X_train_scaled_transformed (str): Placeholder for scaled and transformed training features.
            X_test_scaled_transformed (str): Placeholder for scaled and transformed test features.
            accuracy_scr (str): Placeholder for accuracy score.
            model_path (str): Placeholder for the model path.
            parameters (str): Placeholder for model parameters.
            robust_scaler (str): Placeholder for robust scaler object.
            run_id (str): Placeholder for run ID.
            active_experiment (str): Placeholder for active experiment.
            xgb_model (str): Placeholder for XGBoost model.
        """
        self.model_name = model_name
        self.file = ""
        self.y_train = ""
        self.y_test = ""
        self.X_train_scaled_transformed = ""
        self.X_test_scaled_transformed = ""
        self.accuracy_scr = ""
        self.model_path = ""
        self.parameters = ""
        self.robust_scaler = ""
        self.run_id = ""
        self.active_experiment = ""
        self.xgb_model = ""

    def mlflow_tracking(
        self,
        tracking_uri: str = "./mlruns",
        experiment: str = None,
        new_experiment: str = None,
    ) -> None:
        """
        Sets up MLflow tracking for experiments.

        Args:
            tracking_uri (str): The URI where the MLflow tracking server is hosted. Defaults to "./mlruns".
            experiment (str): The name of the existing experiment to use. If None, a new experiment will be created.
            new_experiment (str): The name of the new experiment to create if no existing experiment is specified.
        """
        # sets tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # creates new experiment if no experiment is specified
        if experiment is None:
            mlflow.create_experiment(new_experiment)
            self.active_experiment = new_experiment
            mlflow.set_experiment(new_experiment)
        else:
            mlflow.set_experiment(experiment)
            self.active_experiment = experiment

    def process_data(self, file: str, test_size: float = 0.25) -> None:
        """Processes raw data for training.

        Args:
            file (str): Path to raw training data.
            test_size (float, optional): Percentage of data reserved for testing. Defaults to 0.25.
        """

        # Validate file name
        if not isinstance(file, str) or not file.endswith(".parquet"):
            raise ValueError(
                "Invalid file name. It should be a string ending with '.parquet'"
            )

        # Validate test size
        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError("Invalid test size. It should be a float between 0 and 1")

        # Generating our data
        logger.info("Reading the dataset from %s...", file)
        if not file.startswith(os.path.abspath(SAFE_BASE_DIR) + os.sep):
            raise ValueError(
                f"Path is not within the allowed directory {SAFE_BASE_DIR}"
            )
        try:
            data = pd.read_parquet(file)
            if not isinstance(data, pd.DataFrame):
                sys.exit("Invalid data format")
        except Exception as e:
            sys.exit(f"Error reading dataset: {e}")

        X = data.drop("Asset_Label", axis=1)
        y = data.Asset_Label

        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size
        )

        df_num_train = X_train.select_dtypes(["float", "int", "int32"])
        df_num_test = X_test.select_dtypes(["float", "int", "int32"])
        self.robust_scaler = RobustScaler()
        X_train_scaled = self.robust_scaler.fit_transform(df_num_train)
        X_test_scaled = self.robust_scaler.transform(df_num_test)

        # Making them pandas dataframes
        X_train_scaled_transformed = pd.DataFrame(
            X_train_scaled, index=df_num_train.index, columns=df_num_train.columns
        )
        X_test_scaled_transformed = pd.DataFrame(
            X_test_scaled, index=df_num_test.index, columns=df_num_test.columns
        )

        del X_train_scaled_transformed["Number_Repairs"]
        del X_test_scaled_transformed["Number_Repairs"]

        # Dropping the unscaled numerical columns
        X_train = X_train.drop(
            ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
        )
        X_test = X_test.drop(
            ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
        )

        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        # Creating train and test data with scaled numerical columns
        X_train_scaled_transformed = pd.concat(
            [X_train_scaled_transformed, X_train], axis=1
        )
        X_test_scaled_transformed = pd.concat(
            [X_test_scaled_transformed, X_test], axis=1
        )

        self.X_train_scaled_transformed = X_train_scaled_transformed.astype(
            {"Motor_Current": "float64"}
        )
        self.X_test_scaled_transformed = X_test_scaled_transformed.astype(
            {"Motor_Current": "float64"}
        )

    def train(self, ncpu: int = 4) -> None:
        """Trains an XGBoost Classifier and tracks models with MLFlow.

        Args:
            ncpu (int, optional): Number of CPU threads used for training. Defaults to 4.
        """
        # Validate ncpu
        if not isinstance(ncpu, int) or ncpu <= 0:
            raise ValueError("Invalid ncpu. It should be a positive integer.")

        # Set xgboost parameters
        self.parameters = {
            "max_bin": 256,
            "scale_pos_weight": 2,
            "lambda_l2": 1,
            "alpha": 0.9,
            "max_depth": 8,
            "num_leaves": 2**8,
            "verbosity": 0,
            "objective": "multi:softmax",
            "learning_rate": 0.3,
            "num_class": 3,
            "nthread": ncpu,
        }

        with mlflow.start_run() as run:
            mlflow.xgboost.autolog()
            xgb_train = xgb.DMatrix(
                self.X_train_scaled_transformed, label=np.array(self.y_train)
            )

        self.xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=100)

    def validate(self) -> float:
        """Performs model validation with testing data.

        Returns:
            float: Accuracy metric.
        """
        dtest = xgb.DMatrix(self.X_test_scaled_transformed, self.y_test)
        xgb_prediction = self.xgb_model.predict(dtest)
        xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(self.y_test))
        self.accuracy_scr = 1 - xgb_errors_count / xgb_prediction.shape[0]

        xp = mlflow.get_experiment_by_name(self.active_experiment)._experiment_id
        self.run_id = mlflow.search_runs(xp, output_format="list")[0].info.run_id

        with mlflow.start_run(self.run_id):
            mlflow.log_metric("accuracy", self.accuracy_scr)

        return self.accuracy_scr

    def save(self, model_path: str) -> None:
        """Saves trained model and scaler to the specified path.

        Args:
            model_path (str): Path where trained model should be saved.
        """
        # Validate model path
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("Invalid model path. It should be a non-empty string.")

        sanitized_model_path = secure_filename(model_path)
        self.model_path = os.path.normpath(
            os.path.join(
                SAFE_BASE_DIR, sanitized_model_path, self.model_name + ".joblib"
            )
        )
        self.model_path = os.path.abspath(self.model_path)
        if not self.model_path.startswith(os.path.abspath(SAFE_BASE_DIR) + os.sep):
            raise ValueError("Path is not within the allowed model directory.")

        self.scaler_path = os.path.normpath(
            os.path.join(
                SAFE_BASE_DIR, sanitized_model_path, self.model_name + "_scaler.joblib"
            )
        )
        self.scaler_path = os.path.abspath(self.scaler_path)
        if not self.scaler_path.startswith(os.path.abspath(SAFE_BASE_DIR) + os.sep):
            raise ValueError("Path is not within the allowed model directory.")

        logger.info("Saving model")
        try:
            with open(self.model_path, "wb") as fh:
                joblib.dump(self.xgb_model, fh.name)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        logger.info("Saving Scaler")
        try:
            with open(self.scaler_path, "wb") as fh:
                joblib.dump(self.robust_scaler, fh.name)
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            raise
