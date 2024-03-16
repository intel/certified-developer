from dataclasses import dataclass
from json import loads
from pathlib import Path
from requests import Response, get, post

from __init__ import Settings, here
Settings(__file__)

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
class Ping(Request):
    endpoint: str = "ping"

    def ping(self) -> Response:
        return get(self.url)

@dataclass
class TrainRequest(Request):
    endpoint: str = "train"
    payload_path: Path = here / "train_payload.json"

    @property
    def payload_text(self) -> str:
        return self.payload_path.read_text()

    @property
    def payload(self) -> dict:
        pl = loads(self.payload_text)
        if not isinstance(pl, dict):
            raise TypeError(f"pl is not dict, pl is {pl.__class__.__name__}")
        return pl

    def post(self) -> Response:
        print("URL: ", self.url)
        print("Payload: ", self.payload_text)
        return post(self.url, headers=header, json=self.payload)

    def get_curl_text(
            self,
            func: str = "post",
            header: str = header,
    ) -> str:
        head = "-H " + str(header) if header else ""
        data = "-d " + str(self.payload)
        text = " ".join([
            "curl",
            "-X",
            func.upper(),
            head,
            data,
        ])
        return text

if __name__ == "__main__":
    p = Ping()
    print(p.ping().text)
    request = TrainRequest()
    print(request.post().text)
