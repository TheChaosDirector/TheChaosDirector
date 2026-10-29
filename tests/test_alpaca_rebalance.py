"""Alpaca paper rebalance planning helpers."""

from src.live.rebalance import plan_rebalance, to_alpaca_symbol


def test_symbol_mapping():
    assert to_alpaca_symbol("BRK-B") == "BRK.B"
    assert to_alpaca_symbol("AAPL") == "AAPL"


def test_plan_rebalance_sells_before_buys_and_skips_dust():
    targets = {"AAPL": 0.10, "MSFT": 0.10}
    current = {"AAPL": 0.20, "MSFT": 0.0, "TSLA": 0.05}
    orders = plan_rebalance(targets, current, equity=10_000, min_notional=25)
    # AAPL sell $1000, TSLA sell $500, MSFT buy $1000
    assert orders[0]["side"] == "sell"
    assert {o["yahoo_symbol"] for o in orders} == {"AAPL", "MSFT", "TSLA"}
    assert orders[0]["symbol"] in {"AAPL", "TSLA"}
    # Dust: 0.1% of 10k = $10 < 25 → ignored
    tiny = plan_rebalance({"AAPL": 0.101}, {"AAPL": 0.100}, equity=10_000, min_notional=25)
    assert tiny == []
