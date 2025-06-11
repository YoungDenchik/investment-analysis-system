from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape

# from train.registries.mlflow_registry import register_model
from train.registries.mlflow_registry import register_full_model
from train.trainers.base import TabularTrainerMixin
from train.utils.utils import dynamic_import
from pipelines.preprocessing.manager import FeaturePipelineManager
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class SklearnTrainer(TabularTrainerMixin):
    """Generic trainer for sklearn‑compatible regressors.

    Логує:
    • діапазон дат train‑сету (`date_start`, `date_end`)
    • приклад входу/виходу у CSV‑артефакт `sample_io/sample.csv`
    • `mape_train` на тренувальній вибірці
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._model = None
        self._ds_range: tuple[str, str] | None = None
        self._sample_io_path: str | None = None
        self._pipe = None

    # ──────────────────────────────────────────────────────────
    # Preprocess: shift target & лог sample IO
    # ──────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame):
        df_proc = df.copy()
        # df_proc["target"] = df_proc["close"].shift(-1)
        # df_proc = df_proc.dropna(subset=["target"])
        # Save date range
        self._ds_range = (
            str(df_proc.index.min().date()),
            str(df_proc.index.max().date()),
        )

        X_base =  df_proc
                  # .drop(["target"], axis=1))

        if self.cfg.get("feature_pipeline_cfg") is not None:
            self._pipe = FeaturePipelineManager.get(self.cfg["model_tag"], self.cfg.get("feature_pipeline_cfg"))
            X = self._pipe.transform(X_base)
            # X = self.apply_pipeline(
            #     X_base, self.cfg["model_tag"], self.cfg.get("feature_pipeline_cfg")
            # )
        else:
            X = X_base

        y = X["close"].shift(-1)
        # X = X.dropna(subset=["target"])

        # Save sample before pipeline
        sample_path = "sample_io.csv"
        X.head(1).to_csv(sample_path, index=False)
        self._sample_io_path = sample_path

        # y = df_proc["target"].values
        return X, y

    # ──────────────────────────────────────────────────────────
    # Fit (train‑only)
    # ──────────────────────────────────────────────────────────
    def fit(self, X, y):
        from sklearn.metrics import mean_absolute_percentage_error as mape

        cls = dynamic_import(self.cfg["hyperparameters"]["model_cls"])
        params = self.cfg["hyperparameters"].get("params", {})

        mlflow.log_params(params)
        if self._ds_range:
            mlflow.log_param("date_start", self._ds_range[0])
            mlflow.log_param("date_end",   self._ds_range[1])
        if self._sample_io_path:
            mlflow.log_artifact(self._sample_io_path, artifact_path="sample_io")

        self._model = cls(**params).fit(X, y)
        mlflow.log_metric("mape_train", float(mape(y, self._model.predict(X))))
        register_full_model(pipeline=self._pipe, model=self._model)
        return self._model

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model is not fitted yet")
        return self._model.predict(X)