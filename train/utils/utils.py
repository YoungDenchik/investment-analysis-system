from datetime import datetime
import pandas as pd
from services.price_data.price_data_service import PriceDataService
from di.container import Container
from dependency_injector.wiring import Provide, inject



# @inject
def load_prices(cfg,
                # svc = Provide[Container.price_data_service]
                svc: PriceDataService,  # тепер очікуємо саме сервіс
                )-> pd.DataFrame:
    df = svc.get_prices(
        ticker   = cfg.ticker,
        interval = cfg.interval,
        start    = datetime.fromisoformat(cfg.start) if cfg.start else None,
        end      = datetime.fromisoformat(cfg.end)   if cfg.end   else None,
        period   = cfg.period,
        market_calendar = cfg.market_calendar,
        tz           = cfg.tz,
        auto_download = cfg.auto_download,
        auto_adjust   = cfg.auto_adjust,
        date_time_index= cfg.date_time_index
    )
    return df.sort_index()

def dynamic_import(path: str):
    mod, _, cls = path.rpartition(".")
    module = __import__(mod, fromlist=[cls])
    return getattr(module, cls)
