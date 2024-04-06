from dataclasses import dataclass, field
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path

from warnings import filterwarnings

this = Path(__file__)
here = this.parent


@dataclass()
class Settings:
    init_path: str
    settings_path: Path = here
    eventlvl: object = INFO
    loglvl: object = INFO
    filterwarn: str = "ignore"
    logger: Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.init_path, Path):
            self.init_path = Path(self.init_path)
        basicConfig(level=self.loglvl)
        self.logger = getLogger(self.init_path.name)

        filterwarnings(self.filterwarn)


# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error