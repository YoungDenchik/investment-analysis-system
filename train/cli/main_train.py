from __future__ import annotations

import sys
import os

# Додаємо кореневу директорію до sys.path, щоб Python міг знаходити модулі.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import hydra
import mlflow
from omegaconf import DictConfig

from train.registries.mlflow_registry import setup_tracking
from train.trainers.factory import make_trainer
from train.utils.utils import load_prices
from di.container import Container


from train.manager.train_manager import TrainModelManager

@hydra.main(version_base=None, config_path="../../config/train/", config_name="train_config.yaml")
def main(cfg):
    # 1) Ініціалізуємо контейнер один раз
    container = Container()
    container.init_resources()
    # якщо load_prices використовує @inject — потрібно його wired:
    container.wire(modules=["train.utils.utils"])

    # 2) Явно беремо сервіс із контейнера
    price_svc = container.price_data_service()

    setup_tracking(cfg)
    df = load_prices(cfg.data, price_svc)
    trainer = make_trainer(cfg.trainer)
    manager = TrainModelManager(cfg, trainer)
    manager.run(df)

if __name__ == "__main__":
    main()