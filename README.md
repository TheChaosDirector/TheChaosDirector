# Autonomous Trading Agent Lab

A hobby research lab that trains AI trading brains on years of US stock market
history, grades them honestly, and can rebalance an **Alpaca PAPER** (practice)
account every trading day.

**Practice money by default. No promises of profit.** Live/real-money trading is
intentionally not wired up here.

## Three modes

| Mode | What it does each day |
|---|---|
| **portfolio** | Splits the account across many stocks/ETFs (and cash). Free to spray thin or concentrate within risk caps. |
| **daytrade** | **Must pick exactly one** name. Buys at the **open**, sells at the **close**. Cannot sit in cash. May log **"hand was forced"** ("I wanted to hold, but the rules made me go in") — that signal never cancels the trade. |
| **concentrated** | Daily allocation like portfolio, but **few fat bets** (rank-then-size + momentum baseline, **≥80% invested**) aiming to **beat SPY**. Trains under harsher costs; can ensemble green folds. Alpaca not wired until it passes the referee. |

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

## Hook up Alpaca PAPER (trade every day with practice money)

1. Create paper API keys in the Alpaca dashboard (Paper Trading → API Keys).
2. Copy env template and fill keys locally (do **not** commit `.env`):

```bash
cp .env.example .env
# edit .env with your ALPACA_API_KEY and ALPACA_SECRET_KEY
```

3. Check the connection:

```bash
python -m src.cli alpaca-status --config configs/default.yaml --mode portfolio
```

4. Dry-run today’s rebalance plan (no orders sent):

```bash
python -m src.cli alpaca-rebalance --config configs/default.yaml --mode portfolio --no-refresh
```

5. Send the orders to Alpaca PAPER:

```bash
python -m src.cli alpaca-rebalance --config configs/default.yaml --mode portfolio --execute
# if the market clock says closed but you still want paper orders:
python -m src.cli alpaca-rebalance --config configs/default.yaml --mode portfolio --execute --force
```

6. Automate it (pick one):

### A) GitHub Actions (recommended “set and forget”)

Runs weekdays from GitHub’s cloud — your laptop can be off.

1. Repo → **Settings → Secrets and variables → Actions** → add:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`  
   (paper keys only — never live keys)
2. Publish the frozen champion model as a tiny release asset (one-time, or again after retrain):

```bash
chmod +x scripts/pack-champion.sh
./scripts/pack-champion.sh --publish
```

3. Merge the workflow on the default branch. File: `.github/workflows/alpaca-daily-paper.yml`  
   Schedule: weekdays ~10:30 ET. Manual run: **Actions → Alpaca PAPER daily rebalance → Run workflow**.

### B) Local cron (machine must be on, with `.env` + champion)

```cron
30 10 * * 1-5 cd /path/to/repo && ./scripts/alpaca-daily-rebalance.sh >> logs/alpaca.log 2>&1
```

### C) Cursor Automation

Automation: [Daily paper trader](https://cursor.com/automations/608d1cba-8af4-11f1-b532-320a589b8025)

1. In the automation settings, attach repo `TheChaosDirector/TheChaosDirector` and set the branch to **`cursor/autonomous-trading-agent-d8e6`** (not empty `main`).
2. Paste / keep the prompt from [`prompts/cursor-alpaca-daily.md`](prompts/cursor-alpaca-daily.md) (it checks out that trading branch every run).
3. Schedule cron suggestion: `30 14 * * 1-5`.
4. Paper API keys as Cloud Agent **Runtime Secrets**.

That prompt: uses the trading branch, downloads the champion, paper-rebalances, grades the choice diary, then **commits/pushes** `experience/portfolio/` so the history survives.

## Choice diary → future training data

Every Alpaca rebalance (dry-run or execute) appends that day’s choice to:

```text
experience/portfolio/decisions.jsonl   # what it saw + what it picked
experience/portfolio/index.csv         # spreadsheet-friendly summary
```

Later, grade those choices with real next-day returns, then export a training table:

```bash
python -m src.cli decisions-status --config configs/default.yaml --mode portfolio
python -m src.cli decisions-label  --config configs/default.yaml --mode portfolio
python -m src.cli decisions-export --config configs/default.yaml --mode portfolio
```

That doesn’t retrain by itself yet — it builds the homework pile. When enough graded days exist, a future training run can study them.

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

## Quickstart — concentrated smoke

```bash
# reuses the same lake from the portfolio smoke download
python -m src.cli train    --config configs/concentrated-smoke.yaml --mode concentrated
python -m src.cli backtest --config configs/concentrated-smoke.yaml --mode concentrated
python -m src.cli paper    --config configs/concentrated-smoke.yaml --mode concentrated --days 10
python -m src.cli referee  --config configs/concentrated-smoke.yaml --mode concentrated
```

Full concentrated research run: `configs/concentrated.yaml`.

Or override mode on any config: `--mode daytrade` / `--mode concentrated`.

```bash
streamlit run app.py   # pick config + mode in the sidebar
```

Full-size portfolio run: `configs/default.yaml` (hundreds of tickers; expect longer training even with parallel folds).

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
