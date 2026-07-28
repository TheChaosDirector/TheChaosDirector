# Autonomous Trading Agent Lab

A hobby research lab that trains an AI "portfolio brain" on years of US stock
market history, grades it honestly, and paper-trades it with fake money.

**No real brokerage. No real money. No promises of profit.** The point is a
rigorous playground: the agent gets a lot of data and a lot of freedom, and a
built-in referee makes sure it never cheats by peeking at the future.

## How it works (plain English)

1. **Data lake** (`src/data/`) — downloads ~10 years of daily prices for a few
   hundred liquid US stocks and ETFs into local files.
2. **Feature factory** (`src/features/`) — turns raw prices into "clues"
   (momentum, volatility, volume shocks, how a stock ranks vs its peers).
   Every clue is lagged one day, so a decision made "today" only ever uses
   information that existed *yesterday evening*.
3. **Trading gym** (`src/env/`) — a market simulator. Each day the agent picks
   how much of the portfolio goes into each stock (and cash). The gym applies
   trading costs and slippage and hands back a score.
4. **Self-training loop** (`src/train/`) — walk-forward training: learn on
   ~3 years, freeze the brain, get graded on the *next* 6 months it has never
   seen, slide forward, repeat. The best-graded brain becomes the "champion".
5. **Backtester** (`src/backtest/`) — replays the frozen champion over history
   and compares it to simply buying and holding SPY.
6. **Paper desk** (`src/paper/`) — a simulated live account. The calendar
   advances one day at a time and the agent trades fake cash using only the
   data available on each simulated morning.
7. **Referee** (`src/referee/`) — the anti-cheat unit. It re-derives features
   from truncated data to prove there is no lookahead, checks the train/test
   embargo gap, re-runs the backtest with 3x costs, and fails any result that
   can't beat buy-and-hold out-of-sample.
8. **Mission control** (`app.py`) — a Streamlit dashboard showing training
   results, equity curves, the paper book, and the referee's pass/fail report.

## Setup

```bash
pip install -r requirements.txt
# PyTorch CPU build (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart (small "smoke" run, ~10-15 minutes)

```bash
python -m src.cli download --config configs/smoke.yaml   # fill the data lake
python -m src.cli train    --config configs/smoke.yaml   # walk-forward self-training
python -m src.cli backtest --config configs/smoke.yaml   # grade the champion on history
python -m src.cli paper    --config configs/smoke.yaml --days 10   # advance the paper desk
python -m src.cli referee  --config configs/smoke.yaml   # anti-cheat report
streamlit run app.py                                     # mission control dashboard
```

For the full-size run, swap `configs/smoke.yaml` for `configs/default.yaml`
(hundreds of tickers, longer training — expect hours, not minutes).

If Yahoo Finance is unreachable, add `--synthetic` to `download` to build a
statistically realistic fake market so every other stage still works.

## Reading the results (the honest way)

- **Equity curve** — the value of the portfolio over time. Up and smooth is
  good; up in one lucky spike is not.
- **Max drawdown** — the worst peak-to-trough loss. This is the "how much pain
  before it recovered" number.
- **Sharpe ratio** — return per unit of wobble. Above ~1 on out-of-sample data
  is genuinely decent; 3+ usually means a bug or a cheat.
- **In-sample vs out-of-sample** — grades on data the agent trained on are
  practice-test scores; only *out-of-sample* grades count.
- **Referee report** — if any check fails, the profit numbers don't matter
  yet. Fix the cheat first.

One profitable curve is not "consistent profit". Consistency means positive
out-of-sample results across multiple walk-forward folds, surviving the 3x
cost stress test, and beating buy-and-hold — and even then, past performance
never guarantees the future.

## Project layout

```
configs/     knobs: universe size, costs, risk limits, training schedule
src/data     download + parquet lake + as-of loaders
src/features leakage-safe feature factory
src/env      the portfolio trading gym (Gymnasium API)
src/train    walk-forward PPO self-training
src/backtest frozen-policy historical replay
src/paper    rolling simulated live desk
src/referee  anti-cheat checks and report
src/registry champion model storage
src/metrics  shared scorecard math
app.py       Streamlit mission control
tests/       unit tests, including "does the referee catch a planted cheat?"
```

## Running the tests

```bash
python -m pytest tests/ -q
```

The most important test plants a deliberately cheating feature (tomorrow's
return, disguised) and asserts the referee flags it.
