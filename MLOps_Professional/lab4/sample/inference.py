import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from string import Template

SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab4")


def validate_inputs(
    model_name: str,
    stage: str,
    model_run_id: int,
    scaler_file_name: str,
    scaler_destination: str,
    data: str,
) -> None:
    """Validates the inputs for inference.

    Args:
        model_name (str): The name of the model to be used for inference.
        stage (str): The stage of the model.
        model_run_id (int): The run ID of the model in MLflow.
        scaler_file_name (str): The name of the scaler file to be used for data scaling.
        scaler_destination (str): The destination path for the scaler.
        data (str): The input data for inference.

    Raises:
        ValueError: If any of the inputs are invalid.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Invalid model name. It should be a non-empty string.")
    if not isinstance(stage, str) or not stage:
        raise ValueError("Invalid stage. It should be a non-empty string.")
    if not isinstance(model_run_id, int) or model_run_id <= 0:
        raise ValueError("Invalid model run ID. It should be a positive integer.")
    if not isinstance(scaler_file_name, str) or not scaler_file_name:
        raise ValueError("Invalid scaler file name. It should be a non-empty string.")
    if not isinstance(scaler_destination, str) or not scaler_destination:
        raise ValueError("Invalid scaler destination. It should be a non-empty string.")
    if not isinstance(data, str) or not data:
        raise ValueError("Invalid data. It should be a non-empty string.")


def inference(
    model_name: str,
    stage: str,
    model_run_id: int,
    scaler_file_name: str,
    scaler_destination: str,
    data: str,
) -> str:
    """
    Perform inference using a pre-trained model and a robust scaler on the provided data.

    Parameters:
    model_name (str): The name of the model to be used for inference.
    stage (str): The stage of the model.
    model_run_id (int): The run ID of the model in MLflow.
    scaler_file_name (str): The name of the scaler file to be used for data scaling.
    scaler_destination (str): The destination path for the scaler.
    data (str): The input data for inference.

    Returns:
    str: The maintenance status of the equipment based on the model's prediction.
    """
    try:
        # Validate inputs
        validate_inputs(
            model_name, stage, model_run_id, scaler_file_name, scaler_destination, data
        )

        scaler_destination = os.path.normpath(
            os.path.join(SAFE_BASE_DIR, scaler_destination)
        )
        scaler_file_path = os.path.normpath(
            os.path.join(scaler_destination, scaler_file_name)
        )
        if not scaler_destination.startswith(
            SAFE_BASE_DIR
        ) or not scaler_file_path.startswith(SAFE_BASE_DIR):
            raise ValueError(
                "Scalar file path is not within the allowed model directory."
            )

        # retrieve scaler
        try:
            mlflow.artifacts.download_artifacts(
                run_id=model_run_id,
                artifact_path=scaler_file_name,
                dst_path=scaler_destination,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve scaler: {e}")

        # load robust scaler
        try:
            with open(scaler_file_path, "rb") as fh:
                robust_scaler = joblib.load(fh.name)
        except Exception as e:
            raise RuntimeError(f"Failed to load robust scaler: {e}")

        # load model
        try:
            model_uri_template = Template("models:/$model_name/$stage")
            model_uri = model_uri_template.substitute(
                model_name=model_name, stage=stage
            )
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # process data sample
        Categorical_Variables = pd.get_dummies(
            data[["Manufacturer", "Generation", "Lubrication", "Product_Assignment"]],
            drop_first=False,
        )
        data = pd.concat([data, Categorical_Variables], axis=1)
        data.drop(
            ["Manufacturer", "Generation", "Lubrication", "Product_Assignment"],
            axis=1,
            inplace=True,
        )

        data = data.astype({"Motor_Current": "float64", "Number_Repairs": "float64"})

        number_samples = data.select_dtypes(["float", "int", "int32"])
        scaled_samples = robust_scaler.transform(number_samples)
        scaled_samples_transformed = pd.DataFrame(
            scaled_samples, index=number_samples.index, columns=number_samples.columns
        )
        del scaled_samples_transformed["Number_Repairs"]
        data = data.drop(
            ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
        )
        data = data.astype(int)
        processed_sample = pd.concat([scaled_samples_transformed, data], axis=1)
        processed_sample = processed_sample.astype({"Motor_Current": "float64"})

        column_names = [
            "Age",
            "Temperature",
            "Last_Maintenance",
            "Motor_Current",
            "Number_Repairs",
            "Manufacturer_A",
            "Manufacturer_B",
            "Manufacturer_C",
            "Manufacturer_D",
            "Manufacturer_E",
            "Manufacturer_F",
            "Manufacturer_G",
            "Manufacturer_H",
            "Manufacturer_I",
            "Manufacturer_J",
            "Generation_Gen1",
            "Generation_Gen2",
            "Generation_Gen3",
            "Generation_Gen4",
            "Lubrication_LTA",
            "Lubrication_LTB",
            "Lubrication_LTC",
            "Product_Assignment_Gala",
            "Product_Assignment_Golden_Delicious",
            "Product_Assignment_Granny_Smith",
        ]

        zeroes_dataframe = pd.DataFrame(0, index=np.arange(1), columns=column_names)
        merged_df = pd.merge(
            zeroes_dataframe,
            processed_sample,
            on=processed_sample.columns.tolist(),
            how="right",
        ).fillna(0)

        columns_to_convert = [
            "Manufacturer_A",
            "Manufacturer_B",
            "Manufacturer_C",
            "Manufacturer_D",
            "Manufacturer_E",
            "Manufacturer_F",
            "Manufacturer_G",
            "Manufacturer_H",
            "Manufacturer_I",
            "Manufacturer_J",
            "Generation_Gen1",
            "Generation_Gen2",
            "Generation_Gen3",
            "Generation_Gen4",
            "Lubrication_LTA",
            "Lubrication_LTB",
            "Lubrication_LTC",
            "Product_Assignment_Gala",
            "Product_Assignment_Golden_Delicious",
            "Product_Assignment_Granny_Smith",
        ]

        merged_df[columns_to_convert] = merged_df[columns_to_convert].astype(int)

        xgb_prediction = model.predict(merged_df)

        for prediction in xgb_prediction:
            if prediction == 0:
                status = "Equipment Does Not Require Scheduled Maintenance"
                return status
            elif prediction == 1:
                status = "Equipment Requires Scheduled Maintenance - Plan Accordingly"
                return status

        return status
    except ValueError as e:
        raise RuntimeError(f"Validation error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
