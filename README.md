# Autonomous Trading Agent Lab

A hobby research lab that trains AI trading brains on years of US stock market
history, grades them honestly, and paper-trades with fake money.

**No real brokerage. No real money. No promises of profit.** The agent gets a
lot of data and freedom, and a built-in referee makes sure it never cheats by
peeking at the future.

## Two modes

| Mode | What it does each day |
|---|---|
| **portfolio** | Splits the account across many stocks/ETFs (and cash). Free to spray thin or concentrate within risk caps. |
| **daytrade** | **Must pick exactly one** name. Buys at the **open**, sells at the **close**. Cannot sit in cash. May log **"hand was forced"** ("I wanted to hold, but the rules made me go in") — that signal never cancels the trade. |

Artifacts for each mode live under `artifacts/<mode>/` (or `artifacts/smoke/<mode>/` for smoke configs) so they never overwrite each other.

## How it works (plain English)

1. **Data lake** — downloads ~10+ years of daily OHLCV for liquid US stocks/ETFs.
2. **Feature factory** — lagged "clues" (momentum, volatility, volume shocks, peer ranks, regime). Day T only sees data through T−1.
3. **Trading gym** — portfolio gym or day-trade gym, with fees/slippage.
4. **Self-training (parallel folds)** — walk-forward: learn on ~N years, exam on the next M months never seen, slide, repeat. Folds run in parallel when `training.n_workers > 1`.
5. **Backtester / paper desk / referee** — historical replay, simulated live desk (future deleted each day), anti-cheat report.

## Setup

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart — portfolio smoke (~minutes with parallel folds)

```bash
python -m src.cli download --config configs/smoke.yaml
python -m src.cli train    --config configs/smoke.yaml --mode portfolio
python -m src.cli backtest --config configs/smoke.yaml --mode portfolio
python -m src.cli paper    --config configs/smoke.yaml --mode portfolio --days 10
python -m src.cli referee  --config configs/smoke.yaml --mode portfolio
```

## Quickstart — day-trade smoke

```bash
# reuses the same lake from the portfolio smoke download
python -m src.cli train    --config configs/daytrade-smoke.yaml
python -m src.cli backtest --config configs/daytrade-smoke.yaml
python -m src.cli paper    --config configs/daytrade-smoke.yaml --days 10
python -m src.cli referee  --config configs/daytrade-smoke.yaml
```

Or override mode on any config: `--mode daytrade`.

```bash
streamlit run app.py   # pick config + mode in the sidebar
```

Full-size run: `configs/default.yaml` (hundreds of tickers; expect longer training even with parallel folds).

Offline: add `--synthetic` to `download` for a fake but realistic market.

## Reading the results (the honest way)

- **Equity curve** — account value over time.
- **Max drawdown** — worst peak-to-trough loss.
- **Sharpe** — return per unit of wobble. On never-seen data, >1 is decent; 3+ often means a bug.
- **In-sample vs out-of-sample** — only exam (out-of-sample) grades count.
- **Forced rate (daytrade)** — how often the agent said its hand was forced.
- **Referee** — if any check fails, ignore the profit numbers until fixed.

## Project layout

```
configs/           knobs + daytrade-smoke.yaml
src/data           download + parquet lake + as-of loaders (incl. opens)
src/features       leakage-safe feature factory
src/env            portfolio gym + daytrade gym
src/train          parallel walk-forward PPO self-training
src/backtest       frozen-policy historical replay
src/paper          rolling simulated live desk (both modes)
src/referee        anti-cheat checks (+ must-pick-one for daytrade)
src/registry       champion model storage under artifacts/<mode>/
app.py             Streamlit mission control
tests/             unit tests, including planted-cheat detection
```

## Running the tests

```bash
python -m pytest tests/ -q
```

The most important test plants a deliberately cheating feature (tomorrow's
return, disguised) and asserts the referee flags it.
