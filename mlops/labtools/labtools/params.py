""" Experiment Logger """

from dataclasses import dataclass, field, make_dataclass
from functools import cached_property
from itertools import product
from json import JSONDecodeError, loads
from logging import getLogger, warning
from os import environ
from pathlib import Path

logger = getLogger(__name__)



@dataclass
class ExperimentRecord:
    """Record of an experiment"""

    @property
    def attrs(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.__annotations__}

    def set_envs(self):
        for attr in self.__annotations__:
            environ[attr] = str(getattr(self, attr))


@dataclass
class Parameters:
    """Environment variables for the experiment"""
    path: Path = field(default=None)
    data: dict = field(default=None, repr=False)
    record_object: type = field(init=False)
    
    @staticmethod
    def get_path(clsname: str) -> Path:
        (data_path := Path(__file__).parent / "data").mkdir(exist_ok=True)
        return data_path / f"{clsname}.json"

    @staticmethod
    def get_data(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        try:
            data = loads(path.read_text())
        except JSONDecodeError:
            warning(f"Could not load data from {path}")
        return data

    @staticmethod
    def make_record_object(clsname: str, data: dict) -> type:
        return make_dataclass(clsname, data.keys(), bases=(ExperimentRecord,))

    def __post_init__(self):
        if self.path is None and self.data is None:
            raise ValueError("Must provide either a path or data")
        if self.path is None:
            self.path = Parameters.get_path(self.__class__.__name__)
        if self.data is None:
            self.data = Parameters.get_data(self.path)
        self.record_object = Parameters.make_record_object("LabRecord", self.data)

    @property
    def record_params(self) -> product:
        return product(*self.data.values())

    @cached_property
    def experiments(self) -> list[object]:
        return [self.record_object(*record) for record in self.record_params]

