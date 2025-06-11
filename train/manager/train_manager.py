from __future__ import annotations

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error

from strategies import make_combiner

class TrainModelManager:
    """Orchestrates training, validation, optional combination & MLflow logging.

    * Якщо trainer повертає лише один стовпець прогнозів – комбінатор **ігнорується**.
    * Якщо `cfg.strategy.kind == "none"` – пропускаємо комбінування навіть для мульти‑моделей.
    """

    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.trainer = trainer
        # створюємо комбінатор лише якщо явно вказано й не "none"
        self.use_combiner = getattr(cfg, "strategy", None) and cfg.strategy.strategy.kind != "none"
        self.combiner = make_combiner(cfg.strategy) if self.use_combiner else None

    def _split(self, X, y):
        """Простий hold‑out tail split за `validation.val_size_days`."""
        val_days = self.cfg.get("validation", {}).get("val_size_days", 60)
        split_idx = -val_days if val_days > 0 else None
        return (X.iloc[:split_idx], y[:split_idx]), (X.iloc[split_idx:], y[split_idx:])

    def run(self, df: pd.DataFrame):
        print(df)


        (X_train, y_train), (X_val, y_val) = self._prepare_data(df)

        with mlflow.start_run(run_name=f"{self.cfg.data.ticker}_{self.cfg.model_tag}"):
            model = self.trainer.fit(X_train, y_train)
            preds_val = self.trainer.predict(X_val)

            # Якщо прогноз – Series/ndarray, конвертуємо в DataFrame з єдиним стовпцем
            if not isinstance(preds_val, pd.DataFrame):
                preds_val_df = pd.DataFrame({"pred": preds_val}, index=X_val.index)
            else:
                preds_val_df = preds_val

            # Комбінування лише якщо >1 стовпців та не вимкнено в конфігу
            if self.use_combiner and preds_val_df.shape[1] > 1:
                mlflow.log_param("strategy", self.cfg.strategy.strategy.kind)
                if hasattr(self.combiner, "fit"):
                    self.combiner.fit(preds_val_df, y_val)
                final_pred = self.combiner.combine(preds_val_df)
                mlflow.log_metric("mape_combined", float(mape(y_val, final_pred)))
                mse = mean_squared_error(y_val, final_pred)
                rmse = float(np.sqrt(mse))
                mlflow.log_metric(
                    "rmse_combined", rmse
                )
            else:
                # Лог тільки однієї моделі
                final_pred = preds_val_df.iloc[:, 0].values

            # Завжди лог артефакт з target + predictions
            out = preds_val_df.copy()
            out["target"] = y_val
            out["final"] = final_pred
            out.to_csv("preds_val.csv", index=False)
            mlflow.log_artifact("preds_val.csv", artifact_path="preds_val")
            return model

    # ──────────────────────────────────────────────────────────
    def _prepare_data(self, df: pd.DataFrame):
        print(df)
        X, y = self.trainer.preprocess(df)
        print(X)
        return self._split(X, y)