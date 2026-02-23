
from dataclasses import dataclass
from enum import Enum


class LinearMode(str, Enum):
    INFERENCE = "inference"
    TRAIN = "train"
