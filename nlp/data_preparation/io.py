"""Read the processed splits without letting pandas turn words like "NA" or "null" into missing values."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False, na_filter=False, dtype={"text": str, "raw_text": str})
