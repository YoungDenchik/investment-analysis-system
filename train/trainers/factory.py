from __future__ import annotations

from typing import Dict, Type

from train.trainers.base import BaseTrainer  # type: ignore
from train.trainers.sklearn_trainer import SklearnTrainer
# from train.trainers.lightning_trainer import LightningLSTMTrainer
from train.trainers.multi_trainer import MultiModelTrainer

TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    "sklearn": SklearnTrainer,
    # "lstm": LightningLSTMTrainer,
    "multi": MultiModelTrainer,
}


def make_trainer(cfg):
    return TRAINER_REGISTRY[cfg["trainer"]](cfg)