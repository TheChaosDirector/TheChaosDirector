"""Mission Control: a Streamlit dashboard over the lab's artifacts.

Run with:  streamlit run app.py

It only READS artifacts produced by the CLI (train/backtest/paper/referee),
so it can't accidentally change an experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config

st.set_page_config(page_title="Trading Agent Lab — Mission Control", layout="wide")


# ------------------------------------------------------------------ helpers
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def equity_chart(curves: dict[str, pd.Series], title: str) -> go.Figure:
    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines"))
    fig.update_layout(
        title=title,
        yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=1.1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Mission Control")
config_files = sorted(str(p) for p in Path("configs").glob("*.yaml"))
config_path = st.sidebar.selectbox("Config", config_files, index=0)
mode = st.sidebar.selectbox("Mode", ["portfolio", "daytrade"], index=0)
cfg = load_config(config_path, mode=mode)
art = cfg.artifacts_dir

manifest = load_json(cfg.lake_dir / "manifest.json")
if manifest:
    st.sidebar.success(
        f"Data lake: {manifest['n_tickers']} tickers "
        f"({manifest['source']}), {manifest['start']} to {manifest['end']}"
    )
else:
    st.sidebar.warning("No data lake yet. Run: python -m src.cli download")

st.sidebar.caption(f"Artifacts: `{art}`")
st.sidebar.markdown(
    "This lab trades **simulated money only**. If the Referee tab shows a "
    "failure, no profit number on any other tab should be trusted.\n\n"
    "**portfolio** = multi-name brain. **daytrade** = must pick one name, "
    "buy open / sell close (may log 'hand was forced')."
)

st.title(f"Autonomous Trading Agent Lab — {mode}")

tab_train, tab_backtest, tab_paper, tab_referee = st.tabs(
    ["Self-Training", "Backtest", "Paper Desk", "Referee"]
)

# ------------------------------------------------------------------ training
with tab_train:
    st.subheader("Walk-forward self-training")
    st.caption(
        "The agent trains on a window of history, is frozen, and is graded on "
        "the following months it has never seen ('exam'). Then the window "
        "slides forward and it retrains. Only exam grades count."
    )
    experiments = load_json(art / "registry" / "experiments.json")
    champion = load_json(art / "registry" / "champion.json")

    if not experiments:
        st.info("No training runs yet. Run: python -m src.cli train")
    else:
        rows = []
        for r in experiments:
            rows.append(
                {
                    "fold": r["fold"],
                    "train": f"{r['train_start']} → {r['train_end']}",
                    "exam": f"{r['test_start']} → {r['test_end']}",
                    "practice Sharpe (in-sample)": round(r["in_sample"]["sharpe"], 2),
                    "exam Sharpe (out-of-sample)": round(r["out_of_sample"]["sharpe"], 2),
                    "exam return": pct(r["out_of_sample"]["total_return"]),
                    "benchmark Sharpe": round(r["benchmark_oos"]["sharpe"], 2),
                    "champion": "★" if champion and champion["fold"] == r["fold"] else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        fig = go.Figure()
        folds = [r["fold"] for r in experiments]
        fig.add_trace(go.Bar(x=folds, y=[r["out_of_sample"]["sharpe"] for r in experiments], name="Agent (exam)"))
        fig.add_trace(go.Bar(x=folds, y=[r["benchmark_oos"]["sharpe"] for r in experiments], name="Buy & hold benchmark"))
        fig.update_layout(
            title="Exam-period Sharpe by fold (higher = better risk-adjusted return)",
            xaxis_title="Fold",
            barmode="group",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")

        if champion:
            st.success(
                f"Champion: fold {champion['fold']} — selected by best exam Sharpe "
                f"({champion['metrics']['out_of_sample']['sharpe']:.2f})."
            )

# ------------------------------------------------------------------ backtest
with tab_backtest:
    st.subheader("Full-history backtest of the champion")
    summary = load_json(art / "backtest" / "base" / "summary.json")
    journal = load_csv(art / "backtest" / "base" / "journal.csv", index_col=0, parse_dates=True)
    bench_eq = load_csv(art / "backtest" / "base" / "benchmark_equity.csv", index_col=0, parse_dates=True)

    if summary is None or journal is None:
        st.info("No backtest yet. Run: python -m src.cli backtest")
    else:
        curves = {"Agent": journal["equity"]}
        if bench_eq is not None:
            curves["Buy & hold benchmark"] = bench_eq.iloc[:, 0]
        st.plotly_chart(
            equity_chart(curves, f"Growth of $1, {summary['period'][0]} to {summary['period'][1]}"),
            width="stretch",
        )

        a, b = summary["agent"], summary["benchmark"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Agent total return", pct(a["total_return"]), delta=pct(a["total_return"] - b["total_return"]) + " vs index")
        c2.metric("Agent Sharpe", f"{a['sharpe']:.2f}")
        c3.metric("Worst drawdown", pct(a["max_drawdown"]))
        c4.metric("Avg daily turnover", pct(a.get("avg_daily_turnover", 0)))

        st.markdown(f"**In plain English:** {summary['agent_says']}")
        st.markdown(f"**The boring alternative:** {summary['benchmark_says']}")
        if summary.get("forced_says"):
            st.markdown(f"**Forced trades:** {summary['forced_says']}")
        if mode == "daytrade" and "ticker" in journal.columns:
            st.subheader("Day-trade tape (last 30 days of backtest)")
            cols = [c for c in ["ticker", "open", "close", "net_return", "reluctance", "forced", "equity"] if c in journal.columns]
            st.dataframe(journal[cols].tail(30), width="stretch")
        st.caption(
            "Careful: the backtest period overlaps data some folds trained on, so it "
            "flatters the agent. The exam grades (Self-Training tab) and the paper "
            "desk are the honest numbers."
        )

        stress = load_json(art / "backtest" / "stress" / "summary.json")
        if stress:
            st.markdown(
                f"**Cost stress ({stress['cost_multiplier']:g}x fees):** total return "
                f"{pct(stress['agent']['total_return'])} vs {pct(a['total_return'])} at normal costs."
            )

# ------------------------------------------------------------------ paper
with tab_paper:
    st.subheader("Paper-trading desk (simulated live account)")
    st.caption(
        "The desk starts AFTER the last training exam, advances one day at a "
        "time, and rebuilds its view of the market each day with the future "
        "physically deleted. Fake cash, honest calendar."
    )
    state = load_json(art / "paper" / "state.json")
    pjournal = load_csv(art / "paper" / "journal.csv", parse_dates=["date"])

    if state is None:
        st.info("Paper desk not started. Run: python -m src.cli paper --days 5")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Account value", f"${state['equity']:,.2f}",
                  delta=f"${state['equity'] - state['cash_start']:,.2f} since start")
        c2.metric("Started", state["start_date"])
        c3.metric("Last processed day", state["last_date"] or "—")

        if pjournal is not None and len(pjournal):
            eq = pjournal.set_index("date")["equity"]
            st.plotly_chart(equity_chart({"Paper account ($)": eq}, "Paper account value"), width="stretch")
            if mode == "daytrade" and "forced" in pjournal.columns:
                st.metric("Forced-trade rate", f"{pjournal['forced'].astype(float).mean() * 100:.0f}% of days")
            st.dataframe(pjournal.tail(15), width="stretch", hide_index=True)

        if mode == "portfolio" and state.get("weights"):
            pos = pd.Series(state["weights"]).sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=pos.index, y=pos.values * 100))
            fig.update_layout(title="Current positions (% of account)", yaxis_title="%", height=320)
            st.plotly_chart(fig, width="stretch")
        elif mode == "daytrade" and state.get("last_ticker"):
            st.info(f"Last day-trade pick: **{state['last_ticker']}** (always 100% of the account for that day).")

# ------------------------------------------------------------------ referee
with tab_referee:
    st.subheader("Anti-cheat referee")
    report = load_json(art / "referee" / "report.json")
    if report is None:
        st.info("No referee report yet. Run: python -m src.cli referee")
    else:
        if report["overall_pass"]:
            st.success(report["verdict"])
        else:
            st.error(report["verdict"])

        for check in report["checks"]:
            label = "PASS" if check["passed"] else "FAIL"
            with st.expander(f"[{label}] {check['name']}", expanded=not check["passed"]):
                st.markdown(check["plain_english"])
                st.json(check["details"])
