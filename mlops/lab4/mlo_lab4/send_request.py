from dataclasses import dataclass
from json import loads
from pathlib import Path
from requests import Response, get, post

from __init__ import Settings, here
Settings(__file__)

header = {"Content-Type": "application/json"}

@dataclass
class RequestManager:
    host: str = "localhost"
    port: int = 5000
    train_payload_path: Path = here / "train_payload.json"
    predict_payload_path: Path = here / "predict_payload.json"

    def get_payload(self, path: Path) -> dict:
        pl = loads(path.read_text())
        if not isinstance(pl, dict):
            raise TypeError(f"pl is not dict, pl is {pl.__class__.__name__}")
        return pl

    @property
    def predict_payload(self) -> dict:
        return self.get_payload(self.predict_payload_path)

    @property
    def train_payload(self) -> dict:
        return self.get_payload(self.train_payload_path)

    @property
    def url_base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def get_url(self, endpoint: str) -> str:
        return f"{self.url_base}/{endpoint}"

    def ping(self) -> Response:
        return get(self.get_url("ping"))

    def post(self, url: str, payload: dict) -> Response:
        return post(url, headers=header, json=payload)

    def train(self) -> Response:
        return self.post(
            self.get_url("train"),
            self.train_payload,
        )

    def predict(self) -> Response:
        return self.post(
            self.get_url("predict"),
            self.predict_payload,
        )

if __name__ == "__main__":
    request = RequestManager()
    #print(request.train().text)
    print(request.predict().text)
