"""Read side of the data lake, including point-in-time ("as-of") access.

The core anti-cheat primitive lives here: :meth:`Lake.as_of` returns a view of
the market that ends at a given date, exactly what a trader standing on that
day could have known. Everything downstream (paper desk, referee re-checks)
uses it instead of slicing full-history frames ad hoc.
"""

from __future__ import annotations

import json
from functools import cached_property

import pandas as pd

from src.config import Config


class Lake:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.lake_dir

    @cached_property
    def manifest(self) -> dict:
        path = self.dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No data lake at {self.dir}. Run: python -m src.cli download"
            )
        return json.loads(path.read_text())

    @property
    def tickers(self) -> list[str]:
        return list(self.manifest["tickers"])

    @property
    def benchmark(self) -> str:
        return self.manifest["benchmark"]

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        path = self.dir / "prices" / f"{ticker}.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def _wide(self, field: str) -> pd.DataFrame:
        cols = {t: self.load_ticker(t)[field] for t in self.tickers}
        wide = pd.DataFrame(cols).sort_index()
        # Only keep dates where the benchmark traded (the market calendar).
        wide = wide[wide[self.benchmark].notna()]
        return wide

    @cached_property
    def close(self) -> pd.DataFrame:
        """Wide matrix: rows = dates, columns = tickers, values = adj close."""
        return self._wide("close")

    @cached_property
    def open(self) -> pd.DataFrame:
        """Wide matrix of daily opens (needed for day-trade mode)."""
        return self._wide("open")

    @cached_property
    def volume(self) -> pd.DataFrame:
        return self._wide("volume")

    def as_of(self, when: str | pd.Timestamp) -> "LakeView":
        """Market data as known at the end of day `when` (inclusive)."""
        ts = pd.Timestamp(when)
        return LakeView(
            close=self.close.loc[:ts],
            open=self.open.loc[:ts],
            volume=self.volume.loc[:ts],
        )


class LakeView:
    """An information snapshot: nothing after its last date exists."""

    def __init__(self, close: pd.DataFrame, open: pd.DataFrame, volume: pd.DataFrame):
        self.close = close
        self.open = open
        self.volume = volume

    @property
    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]
