# # tests/conftest.py
# import pytest
# from datetime import datetime, timezone, timedelta
# import pandas as pd
#
# from services.price_data.price_data_service import PriceDataService, UTC
#
# # --- Fakes --------------------------------------------------------------- #
# class FakeProvider:
#     def __init__(self, frame: pd.DataFrame):
#         self._frame = frame
#         self.calls = 0
#
#     def fetch_price_data(self, *_, **__) -> pd.DataFrame:
#         self.calls += 1
#         return self._frame.copy()
#
# class FakePriceRepo:
#     def __init__(self):
#         self.store = {}  # {(inst_id, interval): DataFrame}
#
#     def bulk_upsert_df(self, df, inst_id, interval):
#         key = (inst_id, interval)
#         self.store[key] = df.copy()
#         return len(df)
#
#     def get_price_data(self, *, instrument_id, interval, start_date, end_date, **_):
#         key = (instrument_id, interval)
#         df = self.store.get(key, pd.DataFrame()).copy()
#         if start_date:
#             df = df[df["date_time"] >= start_date]
#         if end_date:
#             df = df[df["date_time"] <= end_date]
#         return df
#
#     def delete_price_data(self, *, instrument_id, interval, **_):
#         return self.store.pop((instrument_id, interval), pd.DataFrame()).shape[0]
#
# class FakeInstrRepo:
#     def __init__(self):
#         self._map = {"AAPL": 1}
#
#     def get_by_ticker(self, tick):
#         if tick in self._map:
#             return type("Obj", (), {"id": self._map[tick]})
#         return None
#
#     def create(self, ticker):
#         new_id = max(self._map.values(), default=0) + 1
#         self._map[ticker] = new_id
#         return type("Obj", (), {"id": new_id})
#
# # --- Fixtures ------------------------------------------------------------ #
# @pytest.fixture
# def price_frame():
#     idx = pd.date_range("2024-01-02", periods=5, freq="D", tz=UTC)
#     return pd.DataFrame(
#         {"open": range(5), "high": range(5), "low": range(5),
#          "close": range(5), "volume": [10]*5},
#         index=idx,
#     )
#
# @pytest.fixture
# def service(price_frame):
#     provider = FakeProvider(price_frame)
#     repo = FakePriceRepo()
#     instr = FakeInstrRepo()
#     return PriceDataService(provider, repo, instr)


# import pytest
# from datetime import datetime, timezone, timedelta
# from zoneinfo import ZoneInfo
# import pandas as pd
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, scoped_session
#
# from data.persistence.models.base import Base
# # from data.persistence.models.instrument import Instrument
# # from data.persistence.models.price_data import PriceData
# import data.persistence.models
# from data.persistence.repository.price_data_repository import PriceDataRepository
# from data.persistence.repository.instrument_repository import InstrumentRepository
# from services.price_data.price_data_service import PriceDataService
# from data.providers.yfinance_market_provider import YFinanceMarketProvider   # ваш шлях
# import numpy as np
#
# # ---------- БД in-memory --------------------------------------------- #
# _ENGINE = create_engine("sqlite:///:memory:", echo=False, future=True)
# SessionLocal = scoped_session(sessionmaker(_ENGINE, autoflush=False, autocommit=False))
# print("Registered tables:", Base.metadata.tables.keys())
# Base.metadata.create_all(_ENGINE)
# print("Registered tables:", Base.metadata.tables.keys())
#
#
# @pytest.fixture(scope="function")
# def session():
#     """Нова транзакція на тест; rollback після завершення."""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.rollback()
#         db.close()
#
# # ---------- Репозиторії ------------------------------------------------ #
# @pytest.fixture
# def instr_repo(session):
#     return InstrumentRepository(session)
#
# @pytest.fixture
# def price_repo(session):
#     return PriceDataRepository(session)
#
# # ---------- Дані та провайдер ----------------------------------------- #
# @pytest.fixture
# def sample_prices():
#     # індекс дат у UTC
#     idx = pd.date_range("2024-01-02", periods=5, freq="D", tz=timezone.utc)
#
#     # генератор випадкових чисел із фіксованим seed
#     rng = np.random.default_rng(42)
#
#     # генеруємо «open» в діапазоні 100.0–110.0
#     opens = rng.uniform(100.0, 110.0, size=5)
#     # «high» трохи вище за «open»
#     highs = opens + rng.uniform(0.5, 5.0, size=5)
#     # «low» трохи нижче за «open»
#     lows = opens - rng.uniform(0.5, 5.0, size=5)
#     # «close» — випадковим чином між low і high
#     closes = rng.uniform(lows, highs)
#
#     return pd.DataFrame(
#         {
#             "open": opens,
#             "high": highs,
#             "low": lows,
#             "close": closes,
#             "volume": [10] * 5,  # volume теж float, якщо потрібно
#         },
#         index=idx,
#     )
#
# @pytest.fixture
# def fake_provider(sample_prices):
#     """Фіктивний провайдер із лічильником викликів."""
#     class _Fake:
#         def __init__(self, frame):
#             self._frame = frame
#             self.calls = 0
#         def fetch_price_data(self, *_, **__):
#             self.calls += 1
#             return self._frame.copy()
#     return _Fake(sample_prices)
#
# @pytest.fixture
# def service(fake_provider, price_repo, instr_repo):
#     return PriceDataService(provider=fake_provider, price_repo=price_repo, instr_repo=instr_repo)

# @pytest.fixture
# def sample_df():
#     # create small dataframe with 'close' and two features
#     dates = pd.date_range("2022-01-01", periods=10, freq="D")
#     data = {
#         'feature1': np.arange(10),
#         'feature2': np.arange(10, 20),
#         'close': np.arange(20, 30)
#     }
#     return pd.DataFrame(data, index=dates)
#
# @pytest.fixture
# def cfg(tmp_path):
#     return {
#         'model_tag': 'test_model',
#         'hyperparameters': {
#             'model_cls': 'sklearn.ensemble.RandomForestRegressor',
#             'params': {'n_estimators': 10}
#         },
#         'feature_pipeline_cfg': None
#     }

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

