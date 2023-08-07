from dataclasses import dataclass


@dataclass
class Hyperparameters:
    num_epochs: int
    batch_size: int
    lr: float


@dataclass
class Model:
    layers: list[int]


@dataclass
class Logger:
    run_name: str


@dataclass
class Config:
    hyperparameters: Hyperparameters
    model: Model
    logger: Logger
