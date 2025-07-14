import mlflow
import pandas as pd
import numpy as np

# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.persistence.models.base import Base  # упаковка всіх моделей
from data.persistence.repository.price_data_repository import PriceDataRepository
from data.persistence.repository.instrument_repository import InstrumentRepository
from services.price_data.price_data_service import PriceDataService
from data.providers.yfinance_market_provider import YFinanceMarketProvider
from datetime import timezone


# Фікстура для движка SQLite in-memory
@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(eng)
    return eng

# Фікстура для сесійної фабрики
@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)

# Фікстура для репозиторіїв
@pytest.fixture()
def price_repo(session_factory):
    return PriceDataRepository(session_factory=session_factory)

@pytest.fixture()
def instr_repo(session_factory):
    return InstrumentRepository(session_factory=session_factory)

# Примітивна мок-реалізація провайдера
class DummyProvider(YFinanceMarketProvider):
    def fetch_price_data(self, **kwargs):
        import pandas as pd
        idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
        }, index=idx)
        # призначаємо правильні назви колонок для PriceDataService
        df.index.name = "date_time"
        return df

@pytest.fixture()
def price_service(price_repo, instr_repo):
    provider = DummyProvider()
    return PriceDataService(
        provider=provider,
        price_repo=price_repo,
        instr_repo=instr_repo,
        default_tz="UTC",
        auto_create_instrument=True
    )

@pytest.fixture
def sample_prices():
    # індекс дат у UTC
    idx = pd.date_range("2024-01-02", periods=5, freq="D", tz=timezone.utc)

    # генератор випадкових чисел із фіксованим seed
    rng = np.random.default_rng(42)

    # генеруємо «open» в діапазоні 100.0–110.0
    opens = rng.uniform(100.0, 110.0, size=5)
    # «high» трохи вище за «open»
    highs = opens + rng.uniform(0.5, 5.0, size=5)
    # «low» трохи нижче за «open»
    lows = opens - rng.uniform(0.5, 5.0, size=5)
    # «close» — випадковим чином між low і high
    closes = rng.uniform(lows, highs)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10] * 5,  # volume теж float, якщо потрібно
        },
        index=idx,
    )



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

