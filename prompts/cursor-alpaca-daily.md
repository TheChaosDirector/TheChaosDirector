Daily Alpaca PAPER portfolio rebalance + save choices for future training.

Repo: TheChaosDirector/TheChaosDirector
Branch (required): cursor/autonomous-trading-agent-d8e6
Money: PAPER / practice only. Never use live trading. Never print API secrets.

## Setup (every run)
1. Confirm you are in the attached repo checkout.
2. Check out and update the trading branch (do NOT work from empty `main`):
   git fetch origin
   git checkout cursor/autonomous-trading-agent-d8e6
   git pull --ff-only origin cursor/autonomous-trading-agent-d8e6
   Verify these exist: src/data/lake.py, src/cli.py, prompts/cursor-alpaca-daily.md
3. Install deps if needed:
   pip install -r requirements.txt
4. The trained brain is NOT in git. Download it:
   mkdir -p artifacts/bundles
   gh release download portfolio-champion --pattern 'portfolio-champion.zip' --dir artifacts/bundles
   unzip -o artifacts/bundles/portfolio-champion.zip
   Verify: artifacts/portfolio/registry/champion.json exists
5. Confirm env vars ALPACA_API_KEY and ALPACA_SECRET_KEY exist. Do not print their values.

## Trade
Run exactly:
  python -m src.cli alpaca-rebalance --config configs/default.yaml --mode portfolio --execute --skip-if-closed

## Grade + keep the diary
1. Label any past choices that now have next-day prices:
   python -m src.cli decisions-label --config configs/default.yaml --mode portfolio
2. Show diary status:
   python -m src.cli decisions-status --config configs/default.yaml --mode portfolio
3. If files under experience/portfolio/ changed, commit and push them on the trading branch so the cloud VM does not throw the diary away:
   git add experience/portfolio/decisions.jsonl experience/portfolio/index.csv experience/README.md
   git status
   If there is something to commit:
     git commit -m "chore: save paper trading decisions $(date -u +%Y-%m-%d)"
     git push origin cursor/autonomous-trading-agent-d8e6
   Do not commit .env, secrets, artifacts/, or /data/.

## Rules
- If the market is closed and the command skips: treat as success. Summarize “skipped — market closed” and stop (still ok to run decisions-label / status).
- If champion download fails: stop. Tell me to run ./scripts/pack-champion.sh --publish from a machine with the trained champion.
- If Alpaca auth fails: stop. Tell me to fix Cloud Agent Runtime Secrets ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys only).
- If `ModuleNotFoundError: src.data` appears: you are on the wrong branch — go back to Setup step 2.
- Do NOT retrain models.
- Do NOT open a pull request unless something failed and needs a code fix.
- Never switch to live trading.

## Final note (keep short)
End with:
- Status: executed | skipped (market closed) | failed
- Branch used
- Orders / top holdings (brief)
- Decision diary: path + how many saved / labeled
- Whether experience/ was committed and pushed
- Any follow-up needed
