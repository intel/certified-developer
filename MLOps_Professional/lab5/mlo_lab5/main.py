""" Experiment Logger """

from dataclasses import dataclass, field, make_dataclass
from functools import cached_property
from itertools import product
from json import JSONDecodeError, loads
from logging import getLogger, warning
from os import environ
from pathlib import Path

from pandas import DataFrame

from __init__ import here
from IntelPyTorch_Optimizations import main as mainfunc

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
class ExperimentParameters:
    """Environment variables for the experiment"""
    data: dict = field(init=False)
    path: Path = field(init=False)
    record_object: type = field(init=False)
    
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

    @staticmethod
    def make_record_object(clsname: str, data: dict) -> type:
        return make_dataclass(clsname, data.keys(), bases=(ExperimentRecord,))

    def __post_init__(self):
        self.path = ExperimentParameters.get_path(self.__class__.__name__)
        self.data = ExperimentParameters.get_data(self.path)
        self.record_object = ExperimentParameters.make_record_object("ExperimentRecord", self.data)

    @property
    def record_params(self) -> product:
        return product(*self.data.values())

    @cached_property
    def experiments(self) -> list[object]:
        return [self.record_object(*record) for record in self.record_params]


def run_experiment(experiment: ExperimentRecord) -> dict:
    experiment.set_envs()
    record = experiment.attrs
    record["training_time"] = mainfunc(experiment)
    return record


def run_experiments(parameters: ExperimentParameters) -> DataFrame:
    records = [run_experiment(experiment) for experiment in parameters.experiments]
    return DataFrame.from_records(records)


if __name__ == '__main__':
    parameters = ExperimentParameters()
    results = run_experiments(parameters)
    print(results)
