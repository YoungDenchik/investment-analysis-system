# container.py
from dependency_injector import containers, providers
from data.persistence.conf.postgres import SessionLocal
from data.persistence.repository.price_data_repository import PriceDataRepository
from data.persistence.repository.instrument_repository import InstrumentRepository
from data.providers.yfinance_market_provider import YFinanceMarketProvider
from services.price_data.price_data_service import PriceDataService
from services.forecasts.forecast_service import ForecastService
from services.forecasts.model_manager_final import ModelManager
from config.config import CFG_PROD_FILE

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        packages=["api"]
        # packages = ["train"]

    )


    # База даних
    session_factory = providers.Factory(SessionLocal)

    # Репозиторії
    price_repo = providers.Factory(PriceDataRepository, session_factory=session_factory.provider)
    instrument_repo = providers.Factory(InstrumentRepository, session_factory=session_factory.provider)

    # Провайдер ринкових даних
    market_provider = providers.Singleton(YFinanceMarketProvider)

    # Сервіси
    price_data_service = providers.Factory(
        PriceDataService,
        provider=market_provider,
        price_repo=price_repo,
        instr_repo=instrument_repo,
    )

    model_manager = providers.Singleton(ModelManager, CFG_PROD_FILE)
    forecast_service = providers.Singleton(ForecastService, price_data_service=price_data_service,
                                           model_manager=model_manager)
