from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(cfg_data: dict) -> pd.DataFrame:
    csv_path = cfg_data["csv"]
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0).sort_index()
    return df
