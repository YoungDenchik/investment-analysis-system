
from train.trainers.sklearn_trainer import SklearnTrainer

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




def test_sklearn_trainer_fit_predict(sample_df):
    cfg = {
        "model_tag": "test_rf",
        "hyperparameters": {"model_cls": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 5}},
        "feature_pipeline_cfg": None,
    }
    trainer = SklearnTrainer(cfg)
    X, y = trainer.preprocess(sample_df)
    model = trainer.fit(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)