# # tests/unit/test_price_data_service.py
# import pandas as pd
# import pytest
# from pandas.testing import assert_frame_equal
# from datetime import datetime, timezone as tz
#
# from services.price_data.price_data_service import UTC
#
# # ------------------------------------------------------------------ #
# def test_sync_prices_inserts(service, price_frame):
#     inserted = service.sync_prices(ticker="AAPL")
#     assert inserted == len(price_frame)
#
# def test_get_prices_basic(service, price_frame):
#     service.sync_prices(ticker="AAPL")
#     df = service.get_prices(
#         ticker="AAPL", interval="1d",
#         start=price_frame.index[0],
#         end=price_frame.index[-1],
#         period=None,
#         auto_download=False,
#     )
#     # Local TZ conversion
#     assert df.index.tz == tz.ZoneInfo("Europe/Kyiv")
#     # Content equality (ignore tz in comparison)
#     expect = price_frame.tz_convert("Europe/Kyiv")
#     assert_frame_equal(df, expect, check_freq=False)
#
# @pytest.mark.parametrize("interval", ["1d", "1h"])
# def test_range_covered_true(service, price_frame, interval, monkeypatch):
#     service.sync_prices(ticker="AAPL")
#     from services.price_data.price_data_service import PriceDataService
#     df = service._read_from_store(1, interval="1d", start=None, end=None)
#     covered = PriceDataService._range_covered(
#         df, df.index[0], df.index[-1], "NYSE", "1d"
#     )
#     assert covered is True
#
# def test_auto_download_fills_gap(service, price_frame):
#     # repo empty → should trigger provider.fetch_price_data once
#     df = service.get_prices(
#         ticker="AAPL", interval="1d",
#         start=price_frame.index[0], end=price_frame.index[-1],
#         period=None, auto_download=True
#     )
#     assert not df.empty
#     # provider should have been called exactly once
#     assert service.provider.calls == 1
#
# def test_retry_on_transient_error(monkeypatch, service, price_frame):
#     class FlakyProvider(service.provider.__class__):
#         def fetch_price_data(self, *a, **kw):
#             if self.calls < 2:
#                 self.calls += 1
#                 raise ConnectionError("boom")
#             return price_frame
#     service.provider.__class__ = FlakyProvider
#     with pytest.raises(ConnectionError):
#         service.sync_prices(ticker="AAPL", attempts=2)  # reduce max-retries
#
# def test_purge_prices(service):
#     service.sync_prices(ticker="AAPL")
#     removed = service.purge_prices(instrument_id=1,
#                                    interval="1d",
#                                    start=datetime(2024,1,2,tzinfo=UTC),
#                                    end=datetime(2024,1,5,tzinfo=UTC))
#     assert removed > 0

from pandas.testing import assert_frame_equal

# def test_auto_download_and_cache(service, sample_prices):
#     # repo порожній → service має звернутись до провайдера
#     df = service.get_prices(
#         ticker="AAPL", interval="1d",
#         start=sample_prices.index[0], end=sample_prices.index[-1],
#         period=None, auto_download=True,
#     )
#     # провайдер викликався рівно 1 раз
#     assert service.provider.calls == 1
#
#     # повторний виклик → вже з кешу (0 додаткових викликів)
#     service.get_prices(
#         ticker="AAPL", interval="1d",
#         start=sample_prices.index[0], end=sample_prices.index[-1],
#         period=None, auto_download=True,
#     )
#     assert service.provider.calls == 1  # не змінилось

def test_sync_prices(price_service, sample_prices):
    inserted = price_service.sync_prices(ticker="AAPL")
    assert inserted == len(sample_prices)

def test_range_covered_private(price_service, sample_prices):
    inst_id = price_service._resolve_instrument_id("AAPL")
    price_service.price_repo.bulk_upsert_df(sample_prices, inst_id, "1d")
    df = price_service._read_from_store(inst_id, "1d",
                                  sample_prices.index[0], sample_prices.index[-1])
    covered = price_service._range_covered(
        df, sample_prices.index[0], sample_prices.index[-1], "NYSE", "1d")
    assert covered is True
