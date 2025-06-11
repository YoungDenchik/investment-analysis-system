# pipelines/preprocessing/build.py
"""
build.py – фабрика sklearn-Pipeline
-----------------------------------
• читає декларативний YAML (config.yaml);
• підтримує `params:`  і  `when:`  для кроків;
• auto-підставляє  <model_tag>_branch,  коли name == "branch";
• додає атрибут  `.code_hash`  (sha-1 коду + kwargs).

Використання
------------
    pipe_rf   = build("rf")
    pipe_lstm = build("lstm")
"""

import hashlib
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------ #
# 1. ALIAS-РЕЄСТР                                                    #
# ------------------------------------------------------------------ #
_REGISTRY: dict[str, str] = {
    # --- базові трансформери ---
    "core":             "pipelines.preprocessing.core:CorePreprocessor",
    "tech_indicators":  "pipelines.preprocessing.tech_indicators:TechnicalIndicators",
    "rolling_stats":    "pipelines.preprocessing.stats:RollingStats",
    "lag_features":     "pipelines.preprocessing.lag_features:LagFeatures",
    "regime":           "pipelines.preprocessing.regime:RegimeDetector",
    "scaler":           "sklearn.preprocessing:MinMaxScaler",
    "pca":              "sklearn.decomposition:PCA",
    # --- branch-specific (auto) -------------------------------------------
    "arima_branch":     "pipelines.preprocessing.branches:ArimaBranch",
    "tree_branch":      "pipelines.preprocessing.branches:TreeBranch",
    "rf_branch": "pipelines.preprocessing.branches:TreeBranch",
    "lstm_branch":      "pipelines.preprocessing.branches:LstmBranch",
    "svr_branch":       "pipelines.preprocessing.branches:SvrBranch",
}


_REGISTRY = MappingProxyType(_REGISTRY)          #  read-only

_SUPPORTED_TAGS = frozenset(
    ["arima", "rf", "tree", "xgb", "lstm", "svr"]
)

_DEFAULT_CFG = Path(__file__).with_name("config.yaml")

# ------------------------------------------------------------------ #
# 2. HELPERS                                                         #
# ------------------------------------------------------------------ #
@lru_cache                                   # кешуємо імпорт
def _import_class(target: str):
    """
    'pkg.mod:Class'  →  Python class  (одноразове завантаження)
    """
    module_path, cls_name = target.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _hash_pipeline(steps) -> str:
    sha = hashlib.sha1()
    for alias, obj in steps:
        sha.update(alias.encode())
        sha.update(inspect.getsource(obj.__class__).encode())
        sha.update(str(obj.get_params(deep=False)).encode())
    return sha.hexdigest()


# ------------------------------------------------------------------ #
# 3. PUBLIC API                                                      #
# ------------------------------------------------------------------ #
def build(model_tag: str, config_path: str | Path | None = None) -> Pipeline:
    """
    Parameters
    ----------
    model_tag : {'arima','rf','tree','xgb','lstm','svr'}
    config_path :  шлях до YAML;  якщо None – поряд із цим модулем.

    Returns
    -------
    sklearn.pipeline.Pipeline  з атрибутом  `code_hash`
    """
    if model_tag not in _SUPPORTED_TAGS:
        raise ValueError(f"Unknown model_tag '{model_tag}'")

    cfg_file = Path(config_path) if config_path else _DEFAULT_CFG

    try:
        cfg_text = cfg_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(
            f"Cannot decode YAML '{cfg_file}'. "
            "Ensure it is saved as UTF-8.") from e

    cfg = yaml.safe_load(cfg_text)
    if "steps" not in cfg or not isinstance(cfg["steps"], list):
        raise ValueError("config.yaml must contain list 'steps:'")

    steps: list[tuple[str, Any]] = []
    seen_aliases: set[str] = set()

    for block in cfg["steps"]:
        name: str = block["name"]

        if "when" in block and model_tag not in block["when"]:
            continue

        alias = f"{model_tag}_branch" if name == "branch" else name
        if alias not in _REGISTRY:
            raise ValueError(f"Step alias '{alias}' is not registered")

        if alias in seen_aliases:
            raise ValueError(f"Duplicate step alias '{alias}' in YAML")
        seen_aliases.add(alias)

        cls = _import_class(_REGISTRY[alias])
        params = block.get("params", {})
        step_instance = cls(**params)
        steps.append((alias, step_instance))

    pipe = Pipeline(steps)
    pipe.code_hash = _hash_pipeline(steps)
    return pipe
