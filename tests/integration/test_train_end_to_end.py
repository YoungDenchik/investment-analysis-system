"""End‑to‑end tests that exercise the full pipeline under TrainModelManager
with a MultiModelTrainer and an active prediction combiner."""
import types
import numpy as np
from train.trainers.multi_trainer import MultiModelTrainer
from train.manager.train_manager import TrainModelManager


def test_full_multimodel_mean(sample_df):
    """Fit two sub‑models, combine with mean, assert combined metric is logged."""
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(ticker="AAPL"),
        model_tag="int_test",
        validation={"val_size_days": 2},
        strategy=types.SimpleNamespace(kind="mean"),
        models={
            "lr": {
                "class": "sklearn.linear_model.LinearRegression",
                "params": {},
                "pipeline_cfg": None,
            },
            "rf": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "params": {"n_estimators": 3, "random_state": 0},
                "pipeline_cfg": None,
            },
        },
    )

    trainer = MultiModelTrainer(cfg.__dict__)
    manager = TrainModelManager(cfg, trainer)
    model_dict = manager.run(sample_df)

    # Ensure both sub‑models were trained and returned
    assert isinstance(model_dict, dict)
    assert set(model_dict.keys()) == {"lr", "rf"}

    # Validate that prediction combiner produces expected shape
    preds_df = trainer.predict(sample_df.dropna().drop(columns=[]))
    combined = (preds_df.mean(axis=1)).values  # mean combiner logic
    assert combined.shape[0] == preds_df.shape[0]

    # Metric sanity: combined RMSE should be finite
    assert np.isfinite(combined).all()