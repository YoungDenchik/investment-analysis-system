from train.manager.train_manager import TrainModelManager
from train.trainers.sklearn_trainer import SklearnTrainer
import types

import mlflow
import pandas as pd
import numpy as np
import pytest

# --- Dummy mlflow logger so tests don't require running server
class _DummyRun:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    class info:
        run_id = "dummy"

def _noop(*args, **kwargs):
    return None

@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "start_run", lambda *a, **kw: _DummyRun())
    monkeypatch.setattr(mlflow, "log_param", _noop)
    monkeypatch.setattr(mlflow, "log_params", _noop)
    monkeypatch.setattr(mlflow, "log_metric", _noop)
    monkeypatch.setattr(mlflow, "log_artifact", _noop)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _noop)
    monkeypatch.setattr(mlflow, "set_registry_uri", _noop)
    monkeypatch.setattr(mlflow, "set_experiment", _noop)

# --- Sample dataframe fixture
@pytest.fixture
def sample_df():
    idx = pd.date_range("2022-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "feature1": np.arange(10),
        "feature2": np.arange(10, 20),
        "close": np.linspace(20, 30, 10)
    }, index=idx)
    return df

class ConfigDict(dict):
    """A dict that also lets you do `cfg.foo` instead of `cfg['foo']`."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name!r} not in config")

def test_train_manager_runs(sample_df):
    # build config as ConfigDict so .get() and .attr both work
    cfg = ConfigDict({
        "data": types.SimpleNamespace(ticker="TST"),              # namespace for data
        "model_tag": "rf_test",                                   # simple string
        "validation": {"val_size_days": 2},                       # dict for .get(...)
        "strategy": types.SimpleNamespace(kind="none"),           # namespace for strategy
        "hyperparameters": {                                      # dict for trainer
            "model_cls": "sklearn.ensemble.RandomForestRegressor",
            "params": {}
        },
    })

    # trainer expects a plain dict for cfg, so hand it cfg.__dict__-like mapping:
    trainer = SklearnTrainer({
        **cfg,  # supplies model_tag, validation, hyperparameters
    })

    manager = TrainModelManager(cfg, trainer)
    model = manager.run(sample_df)
    assert model is not None