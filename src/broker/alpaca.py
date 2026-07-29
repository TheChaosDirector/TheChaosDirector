"""Alpaca paper-trading client (practice money only).

Keys come from environment variables — never from git:

  ALPACA_API_KEY      (or APCA_API_KEY_ID)
  ALPACA_SECRET_KEY   (or APCA_API_SECRET_KEY)

This module refuses live/money endpoints. Paper base URL only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

log = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


@dataclass
class AlpacaAccount:
    id: str
    equity: float
    cash: float
    buying_power: float
    status: str
    paper: bool


class AlpacaPaperClient:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        self.secret_key = (
            secret_key
            or os.environ.get("ALPACA_SECRET_KEY")
            or os.environ.get("APCA_API_SECRET_KEY")
        )
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Missing Alpaca paper keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(or APCA_API_KEY_ID / APCA_API_SECRET_KEY), e.g. in a local .env file."
            )
        self.base_url = PAPER_BASE_URL
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            }
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if LIVE_BASE_URL in url:
            raise RuntimeError("Refusing to call live Alpaca endpoints from paper client")
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base_url}{path}"
        if LIVE_BASE_URL in url:
            raise RuntimeError("Refusing to call live Alpaca endpoints from paper client")
        r = self.session.post(url, json=body, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text}")
        return r.json()

    def _delete(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        r = self.session.delete(url, timeout=30)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text}")
        return r.json() if r.content else {}

    def account(self) -> AlpacaAccount:
        raw = self._get("/v2/account")
        return AlpacaAccount(
            id=str(raw.get("id", "")),
            equity=float(raw["equity"]),
            cash=float(raw["cash"]),
            buying_power=float(raw["buying_power"]),
            status=str(raw.get("status", "")),
            paper=True,
        )

    def clock(self) -> dict:
        return self._get("/v2/clock")

    def positions(self) -> dict[str, dict]:
        """Return {symbol: {qty, market_value, current_price, ...}}."""
        rows = self._get("/v2/positions")
        out = {}
        for row in rows:
            sym = row["symbol"]
            out[sym] = {
                "qty": float(row["qty"]),
                "market_value": float(row["market_value"]),
                "current_price": float(row["current_price"]),
                "avg_entry_price": float(row.get("avg_entry_price") or 0),
                "side": row.get("side"),
            }
        return out

    def cancel_open_orders(self) -> None:
        self._delete("/v2/orders")

    def submit_notional_market_order(self, symbol: str, side: str, notional: float) -> dict:
        """Buy/sell approximately `notional` dollars of `symbol` at market."""
        body = {
            "symbol": symbol,
            "notional": f"{abs(notional):.2f}",
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        log.info("Alpaca paper order: %s %s $%.2f", side, symbol, abs(notional))
        return self._post("/v2/orders", body)
