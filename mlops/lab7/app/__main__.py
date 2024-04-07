from argparse import ArgumentParser

from requests import post

from app.data_model import TrainPayload, PredictionPayload, headers

train_payload = TrainPayload(
    file="sensor_data.pkl",
    model_name="model",
    model_path="./",
    test_size=25,
    ncpu=4,
    mlflow_tracking_uri="./mlruns",
    mlflow_new_experiment="apples07",
)

prediction_payload = PredictionPayload(
    sample=[
        {
            "Age": 30,
            "Temperature": 55,
            "Last_Maintenance": 3,
            "Motor_Current": 2,
            "Number_Repairs": 4,
            "Manufacturer": "B",
            "Generation": "Gen1",
            "Lubrication": "LTA",
            "Product_Assignment": "Gala",
        }
    ],
    model_run_id="model_run_id",
    scaler_file_name="model_scaler.joblib",
    scaler_destination="./",
    d4p_destination="./d4p",
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Predict using the model")
    args = parser.parse_args()
    if args.train:
        post(train_payload.url, headers=headers, data=train_payload.json_str)
    if args.predict:
        post(prediction_payload.url, headers=headers, data=prediction_payload.json_str)
