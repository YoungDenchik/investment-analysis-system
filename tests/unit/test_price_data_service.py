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
