import pandas as pd
from pandas.testing import assert_frame_equal
from datetime import datetime, timezone

def test_upsert_and_query(price_repo, sample_prices, instr_repo):
    inst_id = instr_repo.create(ticker="AAPL").id
    rows = price_repo.bulk_upsert_df(sample_prices, inst_id, "1d")
    assert rows == 5

    df = price_repo.get_price_data(
        instrument_id=inst_id,
        interval="1d",
        start_date=sample_prices.index[0],
        end_date=sample_prices.index[-1],
        as_dataframe=True,
    )
    print(df)
    sample_prices.index.name = "date_time"
    print(sample_prices)
    assert_frame_equal(df
                       .set_index("date_time")
                       , sample_prices, check_freq=False)

def test_delete(price_repo, sample_prices, instr_repo):
    inst_id = instr_repo.create(ticker="MSFT").id
    price_repo.bulk_upsert_df(sample_prices, inst_id, "1d")
    deleted = price_repo.delete_price_data(
        instrument_id=inst_id,
        interval="1d",
        start_date=datetime(2024,1,1, tzinfo=timezone.utc),
        end_date=datetime(2024,1,10, tzinfo=timezone.utc),
    )
    assert deleted == 5
