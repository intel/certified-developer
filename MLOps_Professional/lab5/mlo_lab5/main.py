from dataclasses import dataclass, field
from json import loads

from __init__ import 

@dataclass(frozen=True)
class ExperimentData:


    def __post_init__(self):
        self.__dict__ = loads()
