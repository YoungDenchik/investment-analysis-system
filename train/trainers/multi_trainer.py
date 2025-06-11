from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd

from train.registries.mlflow_registry import register_full_model
from pipelines.preprocessing.manager import FeaturePipelineManager
from train.utils.utils import dynamic_import
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape



class MultiModelTrainer:
    """Orchestrate training of *heterogeneous* sub‑models."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.models_cfg = cfg["models"]
        self.models = {
            name: dynamic_import(mc["class"])(**mc.get("params", {}))
            for name, mc in self.models_cfg.items()
        }


    # ──────────────────────────────────────────────────────────
    # Preprocess: shift target, зберігаємо метадані
    # ──────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame):
        df_proc = df.copy()
        df_proc["target"] = df_proc["close"].shift(-1)
        df_proc = df_proc.dropna(subset=["target"])
        # дата‑range
        self._ds_range = (
            str(df_proc.index.min().date()),
            str(df_proc.index.max().date()),
        )
        # sample IO
        sample_path = "sample_io_multimodel.csv"
        df_proc.head(1).to_csv(sample_path, index=False)
        self._sample_io_path = sample_path

        X = df_proc.drop(["target"], axis=1)
        y = df_proc["target"].values
        return X, y

        # ──────────────────────────────────────────────────────────

    # Fit + validation per model (з індивідуальним sample‑IO)
    # ──────────────────────────────────────────────────────────

    def fit(self, X_full: pd.DataFrame, y_full: np.ndarray):
        print(X_full.to_string())
        val_days = self.cfg.get("validation", {}).get("val_size_days", 80)
        split_idx = -val_days if val_days > 0 else None
        X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        # y_train, y_val = y_full[:split_idx], y_full[split_idx:]
        print(X_train)

        for name, model in self.models.items():
            m_cfg = self.models_cfg[name]
            tag = name
            pipe = FeaturePipelineManager.get(tag, m_cfg.get("pipeline_cfg"))
            # print(1)
            # print(X_train.to_string())
            Xt = pipe.transform(X_train) if pipe else X_train
            # print("kawawaawa")
            # print(Xt)
            Xt['target'] = Xt["close"].shift(-1)
            Xt = Xt.dropna(subset=["target"])
            y_train = Xt['target'].values
            Xt = Xt.drop(['target'], axis=1)

            Xv = pipe.transform(X_val)   if pipe else X_val
            Xv['target'] = Xv["close"].shift(-1)
            Xv = Xv.dropna(subset=["target"])
            y_val = Xv['target'].values
            Xv = Xv.drop(['target'], axis=1)
            print("12345678900")
            print(Xv)

            # per-model sample IO
            sample_io = pd.DataFrame(Xt[:1])
            sample_io["target"] = y_train[:1]
            sample_path = f"sample_io_{name}.csv"
            sample_io.to_csv(sample_path, index=False)

            with mlflow.start_run(nested=True, run_name=name):
                mlflow.log_params(m_cfg.get("params", {}))
                if self._ds_range:
                    mlflow.log_param("date_start", self._ds_range[0])
                    mlflow.log_param("date_end", self._ds_range[1])
                mlflow.log_artifact(sample_path, artifact_path="sample_io")

                model.fit(Xt, y_train)
                y_hat_t = model.predict(Xt)
                y_hat_v = model.predict(Xv)

                # train/val metrics
                mlflow.log_metric("mape_train", float(mape(y_train, y_hat_t)))
                mlflow.log_metric("mape_val",   float(mape(y_val,   y_hat_v)))

                # compute RMSE explicitly
                mse   = mean_squared_error(y_val, y_hat_v)
                rmse  = float(np.sqrt(mse))
                mlflow.log_metric("rmse_val", rmse)

                register_full_model(pipeline=pipe, model=model, tag=name)

        return self.models

        # ──────────────────────────────────────────────────────────
        # Prediction – повертаємо матрицю [n_samples, n_models]
        # ──────────────────────────────────────────────────────────

    #add strategies
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Побудова словника {model_name: predictions}
        preds: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            m_cfg = self.models_cfg[name]
            tag = name
            pipe = FeaturePipelineManager.get(tag, m_cfg.get("pipeline_cfg"))
            Xf = pipe.transform(X) if pipe else X.values
            preds[name] = model.predict(Xf)
        # Створюємо DataFrame: індекс беремо з X, стовпці – імена моделей
        return pd.DataFrame(preds, index=X.index)