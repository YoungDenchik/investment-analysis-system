from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import logging
import time
import threading
from typing import Any, Callable, Dict, List, Protocol, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from config.config import MLFLOW_TRACKING_URI

# ---------------- soft dependencies ----------------
try:
    import mlflow.pyfunc  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("mlflow must be installed: pip install mlflow") from exc

try:  # optional metrics
    from prometheus_client import Histogram, Counter
except ModuleNotFoundError:  # pragma: no cover
    Histogram = Counter = None  # type: ignore

try:  # fast in‑memory TTL cache
    from cachetools import TTLCache
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("cachetools must be installed: pip install cachetools") from exc

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
import os

LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format=LOG_FORMAT)
logger = logging.getLogger("predictor")

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
class Model(Protocol):
    version: str | int

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...


class Loader(Protocol):
    def load_model(self, uri: str) -> Model: ...


class Strategy(Protocol):
    name: str

    def combine(self, preds: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------
class AssetCfg(BaseModel):
    symbol: str = Field(..., description="Ticker symbol – upper‑case")
    base_models: Dict[str, str]  # tag → MLflow URI / alias
    strategy: str = Field("mean", description="Aggregation strategy name")
    meta_model: str | None = Field(None, description="Meta‑model URI (stacking)")
    # threshold_mape: float = Field(0.1, gt=0.0, description="Alert threshold")

    @field_validator("symbol", mode="before")
    @classmethod
    def _uppercase_symbol(cls, v: str) -> str:
        return v.upper()

    @field_validator("base_models")
    @classmethod
    def _check_base_models(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            raise ValueError("'base_models' cannot be empty")
        return v

    @field_validator("meta_model")
    @classmethod
    def _meta_requires_strategy(
        cls, v: str | None, values: Dict[str, Any]
    ) -> str | None:
        if values.get("strategy") == "meta" and v is None:
            raise ValueError("'meta' strategy requires 'meta_model' URI")
        return v

# ---------------------------------------------------------------------------
# MLflow model wrapper
# ---------------------------------------------------------------------------
class MlflowModel(Model):
    _lock: threading.RLock

    def __init__(self, uri: str, client: MlflowClient):
        self._uri = uri
        self._client = client
        self._model: mlflow.pyfunc.PyFuncModel | None = None
        self._lock = threading.RLock()
        self.version = self._resolve_version(uri)

    def _resolve_version(self, uri: str) -> int | str:
        name, _, alias = uri.partition("@")
        name = name.replace("models:/", "").rstrip("/")
        if alias.isdigit():
            return int(alias)
        mv = self._client.get_model_version_by_alias(name, alias or "Production")
        return int(mv.version)

    def _ensure_loaded(self) -> mlflow.pyfunc.PyFuncModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading MLflow model … %s", self._uri)
                    self._model = mlflow.pyfunc.load_model(self._uri)
        return self._model

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # noqa: D401
        model = self._ensure_loaded()
        return np.asarray(model.predict(X)).ravel()

# ---------------------------------------------------------------------------
# Default Loader (models only)
# ---------------------------------------------------------------------------
class DefaultLoader(Loader):
    def __init__(self, ttl_sec: int = 300):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._client = MlflowClient()
        self._model_cache: TTLCache[str, Model] = TTLCache(maxsize=128, ttl=ttl_sec)
        self._lock = threading.RLock()

    def load_model(self, uri: str) -> Model:  # type: ignore[override]
        try:
            return self._model_cache[uri]
        except KeyError:
            with self._lock:
                if uri not in self._model_cache:
                    self._model_cache[uri] = MlflowModel(uri, self._client)
                    logger.info("Model cached %s", uri)
                return self._model_cache[uri]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class _BaseStrategy(Strategy):
    name: str = "base"
    def combine(self, preds: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

class SingleStrategy(_BaseStrategy):
    name = "single"
    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        if preds.empty:
            raise ValueError("SingleStrategy: empty predictions")
        return preds.iloc[:, 0].to_numpy()

class MeanStrategy(_BaseStrategy):
    name = "mean"
    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        return preds.mean(axis=1).to_numpy()

class MedianStrategy(_BaseStrategy):
    name = "median"
    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        return preds.median(axis=1).to_numpy()

class StackingStrategy(_BaseStrategy):
    name = "meta"
    def __init__(self, meta_model: Model):
        self._meta = meta_model
    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        return self._meta.predict(preds)

_STRATS: Dict[str, Callable[..., Strategy]] = {
    SingleStrategy.name: lambda **_: SingleStrategy(),
    MeanStrategy.name:   lambda **_: MeanStrategy(),
    MedianStrategy.name: lambda **_: MedianStrategy(),
    StackingStrategy.name: lambda meta_model, **_: StackingStrategy(meta_model=meta_model),
}


def build_strategy(name: str, **kwargs: Any) -> Strategy:
    try:
        return _STRATS[name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown strategy '{name}' – choose one of {list(_STRATS)}") from exc

# ---------------------------------------------------------------------------
# Aggregator (per asset)
# ---------------------------------------------------------------------------
class Aggregator:
    def __init__(self, loader: Loader, cfg: AssetCfg):
        self._models: Dict[str, Model] = {tag: loader.load_model(uri) for tag, uri in cfg.base_models.items()}
        meta = loader.load_model(cfg.meta_model) if cfg.meta_model else None
        self._strategy = build_strategy(cfg.strategy, meta_model=meta)

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        if raw.empty:
            raise ValueError("Aggregator.predict received empty dataframe")

        pred_df = pd.DataFrame(index=raw.index)
        for tag, model in self._models.items():
            pred = model.predict(raw)
            if len(pred) < len(raw):  # вирівнюємо, якщо модель повернула коротший ряд
                pad = np.full(len(raw) - len(pred), np.nan)
                pred = np.concatenate([pad, pred])
            pred_df[tag] = pred

        return self._strategy.combine(pred_df)

    @property
    def versions(self) -> Sequence[str | int]:
        vers: List[str | int] = [m.version for m in self._models.values()]
        if hasattr(self._strategy, "_meta"):
            vers.append(getattr(self._strategy, "_meta").version)  # type: ignore[attr-defined]
        return vers

# ---------------------------------------------------------------------------
# ModelManager facade
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(
        self,
        cfg_path: str | Path,
        loader: Loader | None = None,
        reload_interval: int = 300,
        enable_metrics: bool | None = False,
    ):
        self._assets = self._parse_cfg(cfg_path)
        self._loader = loader or DefaultLoader(ttl_sec=reload_interval)
        self._aggs: Dict[str, Aggregator] = {}
        self._lock = threading.RLock()

        self._enable_metrics = enable_metrics if enable_metrics is not None else Histogram is not None
        if self._enable_metrics:
            self._latency_hist = Histogram("predict_latency_seconds", "Time spent in predict()", ["symbol"])
            self._error_ctr = Counter("predict_errors_total", "Prediction errors", ["symbol"])

    def predict(self, symbol: str, data: pd.DataFrame) -> pd.Series:
        symbol = symbol.upper()
        agg = self._get_or_create(symbol)
        start = time.perf_counter()
        try:
            preds = agg.predict(data)
        except Exception:
            if self._enable_metrics:
                self._error_ctr.labels(symbol=symbol).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            if self._enable_metrics:
                self._latency_hist.labels(symbol=symbol).observe(elapsed)
            logger.info(
                "predict symbol=%s versions=%s latency_ms=%.1f n=%d",
                symbol,
                ",".join(map(str, agg.versions)),
                elapsed * 1_000,
                len(data),
            )
        return pd.Series(preds, index=data.index, name="pred")

    def list_assets(self) -> List[str]:
        return list(self._assets.keys())

    # ---------------- internals ----------------
    def _parse_cfg(self, path: str | Path) -> Dict[str, AssetCfg]:
        import yaml  # local import to keep global namespace clean
        raw = Path(path).expanduser().read_text(encoding="utf-8")
        try:
            cfg_doc = yaml.safe_load(raw)
        except Exception as exc:
            raise RuntimeError(f"YAML parsing failed: {exc}")

        if "assets" not in cfg_doc or not isinstance(cfg_doc["assets"], list):
            raise ValueError("Config YAML must contain a list under 'assets:'")

        assets: Dict[str, AssetCfg] = {}
        for asset_raw in cfg_doc["assets"]:
            try:
                asset = AssetCfg.model_validate(asset_raw)
            except ValidationError as exc:
                raise ValueError(f"Invalid config for asset {asset_raw.get('symbol')}: {exc}") from exc
            assets[asset.symbol] = asset
        return assets

    def _get_or_create(self, symbol: str) -> Aggregator:
        if symbol not in self._assets:
            raise KeyError(f"Asset '{symbol}' is not configured – see YAML")
        try:
            return self._aggs[symbol]
        except KeyError:
            with self._lock:
                if symbol not in self._aggs:
                    self._aggs[symbol] = Aggregator(self._loader, self._assets[symbol])
                return self._aggs[symbol]

# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------
_default_mm: ModelManager | None = None

def get_default_manager(cfg_path: str | Path) -> ModelManager:  # pragma: no cover
    global _default_mm
    if _default_mm is None:
        _default_mm = ModelManager(cfg_path)
    return _default_mm

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys
    import yaml

    if len(sys.argv) != 4:
        print("Usage: python predictor_layer.py <config.yml> <SYMBOL> <prices.csv>")
        sys.exit(1)

    cfg_f, symbol, csv_f = sys.argv[1:]
    manager = get_default_manager(cfg_f)
    df = pd.read_csv(csv_f, parse_dates=[0], index_col=0)
    print(manager.predict(symbol, df).head())
