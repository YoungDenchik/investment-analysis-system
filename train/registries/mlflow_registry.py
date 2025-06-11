from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import shutil

import mlflow


def setup_tracking(cfg: dict[str, Any]):
    mlflow.set_tracking_uri(cfg.get("tracking_uri"))
    mlflow.set_registry_uri(cfg.get("registry_uri"))
    if "experiment" in cfg:
        mlflow.set_experiment(cfg["experiment"])

import os
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

from pathlib import Path
from typing import Dict





class FullModel(mlflow.pyfunc.PythonModel):
    """
    PythonModel, який у load_context підтягує pipeline.joblib та model.joblib
    із артефактів, а в predict() застосовує transform() (якщо pipeline != None)
    і далі model.predict().
    У __init__ жодних аргументів не потребуємо.
    """

    def __init__(self):
        super().__init__()
        # actual pipeline і модель підвантажаться у load_context()
        self.pipeline = None
        self.model = None

    def load_context(self, context) -> None:
        """
        context.artifacts — словник виду {'pipeline': 'runs:/.../pipeline.joblib', 'model': 'runs:/.../model.joblib'}
        Завантажуємо обидва файли у пам'ять.
        """
        # 1) Якщо в артефактах є ключ 'pipeline', підтягуємо його
        pip_uri = context.artifacts.get("pipeline")
        if pip_uri is not None:
            # MLflow сам скопіює файл у локальний каталог <run_root>/artifacts/full_model/
            local_path = context.artifacts["pipeline"]
            self.pipeline = joblib.load(local_path)

        # 2) Модель обов’язково є
        model_uri = context.artifacts["model"]
        local_model_path = model_uri
        self.model = joblib.load(local_model_path)

    def predict(self, context, model_input):
        """
        model_input за замовчуванням — pandas.DataFrame.
        Якщо self.pipeline != None, спочатку трансформуємо, інакше беремо model_input as is.
        """
        if self.pipeline is not None:
            X_transformed = self.pipeline.transform(model_input)
        else:
            X_transformed = model_input

        return self.model.predict(X_transformed)


# train/registries/mlflow_registry.py

def register_full_model(
    pipeline: Optional[Any],   # pipeline або None
    model: Any,
    tag: str
):
    """
    Логування pyfunc-моделі, де модель та pipeline (якщо pipeline != None)
    зберігаються окремими артефактами.
    Потім MLflow у load_context підхопить їх автоматично.
    """
    # 1) Тимчасова папка для збереження pipeline+model
    tmp_dir = Path("temp_pyfunc_artifacts")
    tmp_dir.mkdir(exist_ok=True)

    # 2) Серіалізуємо модель
    model_path = tmp_dir / "model.joblib"
    joblib.dump(model, model_path)

    # 3) Підготуємо словник artifacts
    artifacts: Dict[str, str] = {"model": str(model_path)}

    # 4) Якщо є pipeline — теж зберігаємо
    if pipeline is not None:
        pipe_path = tmp_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipe_path)
        artifacts["pipeline"] = str(pipe_path)

    # 5) Логування pyfunc без конда_env
    #    Передаємо python_model=FullModel(), бо його __init__ не потребує аргументів.
    mlflow.pyfunc.log_model(
        python_model=FullModel(),
        artifact_path="full_model",          # у run/artifacts/full_model/
        registered_model_name=tag,           # ім’я в Registry
        artifacts=artifacts                  # {'model': '.../model.joblib', 'pipeline': '.../pipeline.joblib'}
        # conda_env не передаємо → MLflow збере дефолтне середовище
    )

    # 6) чистимо папку з тимчасовими файлами
    shutil.rmtree(tmp_dir)
