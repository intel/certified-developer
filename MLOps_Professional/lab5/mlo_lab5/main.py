""" Experiment Logger """

from dataclasses import dataclass, field
from json import JSONDecodeError, loads
from logging import getLogger, warning
from pathlib import Path

from __init__ import here

logger = getLogger(__name__)

@dataclass
class ExperimentObject:
    name: str = field()
    description: str = field(init=False)
    data: dict = field(init=False)
    path: Path = field(init=False)
    
    @staticmethod
    def get_path(clsname: str) -> Path:
        path = here.parent / f"data/{clsname}.json"
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        return path

    @staticmethod
    def get_data(path: Path) -> dict:
        try:
            data = loads(path.read_text())
        except JSONDecodeError:
            warning(f"Could not load data from {path}")
        return data
        
    def __post_init__(self):
        self.path = ExperimentObject.get_path(self.__class__.__name__)
        self.data = ExperimentObject.get_data(self.path)


@dataclass
class ExperimentRecord(ExperimentObject):
    """Experiment record"""
    description: str = "Record of the experiment"

    def __post_init__(self):
        super().__post_init__()
        self.data["name"] = self.data["name"].replace("{}", self.name)


@dataclass
class ExperimentParameters(ExperimentObject):
    """Environment variables for the experiment"""
    name: str = field(default="parameters")
    description: str = "Parameters for the experiment"


if __name__ == '__main__':
    record = ExperimentRecord(name="Example")
    print(record.data)
    parameters = ExperimentParameters()
    print(parameters.data)
