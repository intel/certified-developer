import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import daal4py as d4p

SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab7")


def validate_inputs(
    model_run_id: int,
    scaler_file_name: str,
    scaler_destination: str,
    d4p_file_name: str,
    d4p_destination: str,
    data: str,
) -> None:
    """Validates the inputs for inference.

    Args:
        model_run_id (int): ID of the model run.
        scaler_file_name (str): Name of the scaler file.
        scaler_destination (str): Destination path for the scaler file.
        d4p_file_name (str): Name of the d4p file.
        d4p_destination (str): Destination path for the d4p file.
        data (str): Path to the data file.

    Raises:
        ValueError: If any of the inputs are invalid.
    """
    if not isinstance(model_run_id, int) or model_run_id <= 0:
        raise ValueError("Invalid model run ID. It should be a positive integer.")
    if not isinstance(scaler_file_name, str) or not scaler_file_name:
        raise ValueError("Invalid scaler file name. It should be a non-empty string.")
    if not isinstance(scaler_destination, str) or not scaler_destination:
        raise ValueError("Invalid scaler destination. It should be a non-empty string.")
    if not isinstance(d4p_file_name, str) or not d4p_file_name:
        raise ValueError("Invalid d4p file name. It should be a non-empty string.")
    if not isinstance(d4p_destination, str) or not d4p_destination:
        raise ValueError("Invalid d4p destination. It should be a non-empty string.")
    if not isinstance(data, str) or not data:
        raise ValueError("Invalid data. It should be a non-empty string.")


def inference(
    model_run_id: int,
    scaler_file_name: str,
    scaler_destination: str,
    d4p_file_name: str,
    d4p_destination: str,
    data: str,
) -> str:
    """
    Perform inference using a pre-trained model and scaler.

    Args:
        model_run_id (int): ID of the model run.
        scaler_file_name (str): Name of the scaler file.
        scaler_destination (str): Destination path for the scaler file.
        d4p_file_name (str): Name of the d4p file.
        d4p_destination (str): Destination path for the d4p file.
        data (str): Path to the data file.

    Returns:
        str: Inference result indicating maintenance status.
    """
    try:
        # Validate inputs
        validate_inputs(
            model_run_id,
            scaler_file_name,
            scaler_destination,
            d4p_file_name,
            d4p_destination,
            data,
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

        d4p_destination = os.path.normpath(os.path.join(SAFE_BASE_DIR, d4p_destination))
        d4p_file_path = os.path.normpath(os.path.join(d4p_destination, d4p_file_name))
        if not d4p_destination.startswith(
            SAFE_BASE_DIR
        ) or not d4p_file_path.startswith(SAFE_BASE_DIR):
            raise ValueError("d4p file path is not within the allowed model directory.")

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

        # retrieve d4p model
        try:
            mlflow.artifacts.download_artifacts(
                run_id=model_run_id,
                artifact_path=d4p_file_name,
                dst_path=d4p_destination,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve d4p model: {e}")

        # load d4p model
        try:
            with open(d4p_file_path, "rb") as fh:
                daal_model = joblib.load(fh.name)
        except Exception as e:
            raise RuntimeError(f"Failed to load d4p model: {e}")

        # process data sample
        try:
            data = pd.read_parquet(data)
        except Exception as e:
            raise RuntimeError(f"Failed to read data file: {e}")

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

        # perform inference
        daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=3, resultsToEvaluate="computeClassLabels", fptype="float"
        )

        daal_prediction = daal_predict_algo.compute(merged_df, daal_model)

        for prediction in daal_prediction.prediction[:, 0]:
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
