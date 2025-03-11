import uvicorn
import logging
import warnings

from fastapi import FastAPI, HTTPException
from data_model import MaintenancePayload
from maintenance import test_maintenance


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


@app.post("/maintenance")
async def predict(payload: MaintenancePayload) -> dict:
    """
    Predicts the maintenance status based on the given payload.

    Args:
        payload (MaintenancePayload): The payload containing the temperature data.

    Returns:
        dict: A dictionary containing the message and maintenance status.
    """
    try:
        # Validate temperature
        if not isinstance(payload.temperature, (int, float)):
            raise ValueError("Invalid temperature. It should be a number.")

        maintenance_result = test_maintenance(payload.temperature)
        return {"msg": "Completed Analysis", "Maintenance Status": maintenance_result}
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
