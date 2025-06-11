from train.trainers.multi_trainer import MultiModelTrainer

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




def test_multi_trainer_predict_shape(sample_df):
    cfg = {
        "model_tag": "mix",
        "validation": {"val_size_days": 2},
        "models": {
            "lr": {"class": "sklearn.linear_model.LinearRegression", "params": {}},
            "rf": {"class": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 3}},
        },
    }
    trainer = MultiModelTrainer(cfg)
    X, y = trainer.preprocess(sample_df)
    trainer.fit(X, y)
    preds_df = trainer.predict(X.drop(columns=[]))  # give DataFrame
    assert preds_df.shape[1] == 2  # two models
    assert preds_df.shape[0] == len(X)
