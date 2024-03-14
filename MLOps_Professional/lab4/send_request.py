from dataclasses import dataclass
from json import loads
from pathlib import Path
from requests import post

from sample.__init__ import here

header = {"Content-Type": "application/json"}

@dataclass
class Request:
    endpoint: str
    host: str = "localhost"
    port: int = 5000
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/{self.endpoint}"


@dataclass
class TrainRequest(Request):
    endpoint: str = "train"
    payload_path: Path = here.parent / "train_payload.json"

    @property
    def payload(self) -> dict:
        return loads(self.payload_path.read_text())

    def post(self) -> None:
        return post(self.url, headers=header, json=self.payload)


if __name__ == "__main__":
    request = TrainRequest()
    print(request.payload)
    request.post()
