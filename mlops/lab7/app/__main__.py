from argparse import ArgumentParser
from json import dumps
from os import getenv

from requests import get, post

from app.__init__ import URL_BASE
from app.data_model import TrainPayload, PredictionPayload, headers
from app.generate_data import generate


train_payload = dict(
    file="/data/sensor_data.pkl",
    model_name=getenv("MLFLOW_MODEL_NAME"),
    model_path="./",
    test_size=25,
    ncpu=4,
    mlflow_tracking_uri=getenv("MLFLOW_TRACKING_URI"),
    mlflow_experiment=getenv("MLFLOW_EXPERIMENT_NAME"),
)

prediction_payload = dict(
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
    model_run_id=getenv("MLFLOW_RUN_ID"),
    scaler_file_name=f"{getenv("MLFLOW_MODEL_NAME")}_scaler.joblib",
    scaler_destination="./",
    d4p_destination="./d4p",
    d4p_file_name= f"{getenv("MLFLOW_MODEL_NAME")}.joblib",
)

if __name__ == "__main__":
    print("URL_BASE: ", URL_BASE)
    parser = ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="Ping the server")
    parser.add_argument(
        "--generate", action="store_true", help="Generate data for training"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--predict", action="store_true", help="Predict using the model"
    )
    args = parser.parse_args()
    if args.ping:
        request_type = "ping"
        response = get(f"{URL_BASE}/ping")
    elif args.generate:
        request_type = "data generation"
        generate()
    elif args.train:
        print(f"headers: {headers}")
        request_type = "training"
        print(f"sending {train_payload}")
        response = post(f"{URL_BASE}/train", headers=headers, data=dumps(train_payload))
    elif args.predict:
        print(f"headers: {headers}")
        request_type = "prediction"
        print(f"sending {prediction_payload}")
        response = post(f"{URL_BASE}/predict", headers=headers, data=dumps(prediction_payload))
    else:
        raise ValueError(f"No valid arguments passed | {parser.print_help()}")

    if args.generate or response.status_code == 200:
        print(f"{request_type} request was successful\n{response.json()}")
    else:
        print(f"{request_type} request failed")
        print(response.status_code, response.text)
        exit(1)
