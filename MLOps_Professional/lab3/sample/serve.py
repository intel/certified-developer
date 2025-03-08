import uvicorn
import logging
import warnings

from fastapi import FastAPI, HTTPException
from data_model import TrainPayload
from train import HarvesterMaintenance

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@app.get("/ping")
async def ping() -> dict:
    """Ping server to determine status

    Returns:
        dict: API response
        response from server on health status
    """
    return {"message": "Server is Running"}


@app.post("/train")
async def train(payload: TrainPayload) -> dict:
    """Training Endpoint
    This endpoint process raw data and trains an XGBoost Classifier

    Args:
        payload (TrainPayload): Training endpoint payload model

    Returns:
        dict: Accuracy metrics and other logger feedback on training progress.
    """
    try:
        # Validate inputs
        if not isinstance(payload.model_name, str) or not payload.model_name:
            raise ValueError("Invalid model name. It should be a non-empty string.")
        if not isinstance(payload.file, str) or not payload.file.endswith(".parquet"):
            raise ValueError(
                "Invalid file name. It should be a string ending with '.parquet'"
            )
        if not isinstance(payload.test_size, float) or not (0 < payload.test_size < 1):
            raise ValueError("Invalid test size. It should be a float between 0 and 1")
        if not isinstance(payload.ncpu, int) or payload.ncpu <= 0:
            raise ValueError("Invalid ncpu. It should be a positive integer.")
        if not isinstance(payload.model_path, str) or not payload.model_path:
            raise ValueError("Invalid model path. It should be a non-empty string.")

        model = HarvesterMaintenance(payload.model_name)
        model.mlflow_tracking(
            tracking_uri=payload.mlflow_tracking_uri,
            new_experiment=payload.mlflow_new_experiment,
            experiment=payload.mlflow_experiment,
        )
        logger.info("Configured Experiment and Tracking URI for MLFlow")
        model.process_data(payload.file, payload.test_size)
        logger.info("Data has been successfully processed")
        model.train(payload.ncpu)
        logger.info("Maintenance Apple Harvester Model Successfully Trained")
        model.save(payload.model_path)
        logger.info("Saved Harvester Maintenance Model")
        accuracy_score = model.validate()
        return {
            "msg": "Model trained successfully",
            "validation scores": accuracy_score,
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    """Main entry point for the server.

    This block runs the FastAPI application using Uvicorn.
    """
    try:
        uvicorn.run("serve:app", host="127.0.0.1", port=5000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
