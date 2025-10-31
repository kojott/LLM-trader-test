#!/usr/bin/env python3
"""Streamlit dashboard for monitoring the DeepSeek trading bot."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_CSV = DATA_DIR / "portfolio_state.csv"
TRADES_CSV = DATA_DIR / "trade_history.csv"
DECISIONS_CSV = DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = DATA_DIR / "ai_messages.csv"
ENV_PATH = BASE_DIR / ".env"
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_SNAPSHOT_SECONDS = 180.0

COIN_TO_SYMBOL: Dict[str, str] = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "BTCUSDT",
    "DOGE": "DOGEUSDT",
    "BNB": "BNBUSDT",
}

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")


def resolve_risk_free_rate() -> float:
    """Return annualized risk-free rate configured for Sortino ratio."""
    env_value = os.getenv("SORTINO_RISK_FREE_RATE") or os.getenv("RISK_FREE_RATE")
    if env_value is None:
        return DEFAULT_RISK_FREE_RATE
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid SORTINO_RISK_FREE_RATE/RISK_FREE_RATE value '%s'; using default %.4f",
            env_value,
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE


RISK_FREE_RATE = resolve_risk_free_rate()

BINANCE_CLIENT: Client | None = None
if BN_API_KEY and BN_SECRET:
    try:
        BINANCE_CLIENT = Client(BN_API_KEY, BN_SECRET, testnet=False)
    except Exception as exc:
        logging.warning("Unable to initialize Binance client: %s", exc)
else:
    logging.info("Binance credentials not provided; live prices disabled.")


def load_csv(path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning empty frame when missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data(ttl=15)
def get_portfolio_state() -> pd.DataFrame:
    df = load_csv(STATE_CSV, parse_dates=["timestamp"])
    if df.empty:
        return df

    numeric_cols = [
        "total_balance",
        "total_equity",
        "total_return_pct",
        "num_positions",
        "total_margin",
        "net_unrealized_pnl",
        "btc_price",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


@st.cache_data(ttl=15)
def get_trades() -> pd.DataFrame:
    df = load_csv(TRADES_CSV, parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    numeric_cols = [
        "quantity",
        "price",
        "profit_target",
        "stop_loss",
        "leverage",
        "confidence",
        "pnl",
        "balance_after",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=15)
def get_ai_decisions() -> pd.DataFrame:
    df = load_csv(DECISIONS_CSV, parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


@st.cache_data(ttl=15)
def get_ai_messages() -> pd.DataFrame:
    df = load_csv(MESSAGES_CSV, parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


def parse_positions(position_text: str | float) -> pd.DataFrame:
    """Split compact position text into structured rows."""
    if pd.isna(position_text) or not isinstance(position_text, str):
        return pd.DataFrame()
    if position_text.strip().lower() == "no positions":
        return pd.DataFrame()

    rows: List[Dict[str, str | float]] = []
    for chunk in position_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            symbol, side, rest = chunk.split(":")
            quantity, entry_price = rest.split("@")
            rows.append(
                {
                    "coin": symbol,
                    "side": side,
                    "quantity": float(quantity),
                    "entry_price": float(entry_price),
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def fetch_current_prices(coins: List[str]) -> Dict[str, float | None]:
    """Fetch latest market prices for the provided coin tickers."""
    prices: Dict[str, float | None] = {coin: None for coin in coins}
    if not BINANCE_CLIENT:
        return prices

    for coin in coins:
        symbol = COIN_TO_SYMBOL.get(coin.upper(), f"{coin.upper()}USDT")
        try:
            ticker = BINANCE_CLIENT.get_symbol_ticker(symbol=symbol)
            prices[coin] = float(ticker["price"])
        except Exception as exc:
            logging.warning("Failed to fetch price for %s: %s", symbol, exc)
            prices[coin] = None
    return prices


def estimate_period_seconds(index: pd.Index, default: float = DEFAULT_SNAPSHOT_SECONDS) -> float:
    """Infer measurement cadence from a datetime-like index."""
    if index.size < 2:
        return default
    try:
        diffs = index.to_series().diff().dropna()
    except Exception:
        return default
    if diffs.empty:
        return default
    try:
        period_seconds = diffs.dt.total_seconds().median()
    except AttributeError:
        period_seconds = default
    if not period_seconds or not np.isfinite(period_seconds) or period_seconds <= 0:
        return default
    return float(period_seconds)


def compute_sharpe_ratio(trades_df: pd.DataFrame) -> float | None:
    """Compute annualized Sharpe ratio from realized (closed) trades."""
    if trades_df.empty or "action" not in trades_df.columns:
        return None

    actions = trades_df["action"].astype(str).str.upper()
    closes = trades_df.loc[actions == "CLOSE"].copy()
    if closes.empty or "balance_after" not in closes.columns:
        return None

    closes.sort_values("timestamp", inplace=True)
    closes = closes.set_index("timestamp")

    balances = pd.to_numeric(closes["balance_after"], errors="coerce").dropna()
    if balances.size < 2:
        return None

    returns = balances.pct_change().dropna()
    if returns.empty:
        return None

    std = returns.std()
    if std is None or np.isclose(std, 0.0):
        return None

    period_seconds = estimate_period_seconds(closes.index)

    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    sharpe = returns.mean() / std * np.sqrt(periods_per_year)
    return float(sharpe) if np.isfinite(sharpe) else None


def compute_sortino_ratio(state_df: pd.DataFrame, risk_free_rate: float) -> float | None:
    """Compute annualized Sortino ratio from total equity snapshots."""
    if state_df.empty or "total_equity" not in state_df.columns:
        return None

    equity = pd.to_numeric(state_df["total_equity"], errors="coerce").dropna()
    if equity.size < 2:
        return None

    returns = equity.pct_change().dropna()
    if returns.empty:
        return None

    period_seconds = estimate_period_seconds(equity.index)
    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_rf
    if not np.isfinite(excess_return):
        return None

    downside = np.minimum(returns - per_period_rf, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return None

    sortino = (excess_return / downside_deviation) * np.sqrt(periods_per_year)
    return float(sortino) if np.isfinite(sortino) else None


def render_portfolio_tab(state_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    if state_df.empty:
        st.info("No portfolio data logged yet.")
        return

    latest = state_df.iloc[-1]
    margin_allocated = latest.get("total_margin", 0.0)
    if pd.isna(margin_allocated):
        margin_allocated = 0.0
    margin_allocated = float(margin_allocated)
    unrealized_pnl = latest.get("net_unrealized_pnl", np.nan)
    if pd.isna(unrealized_pnl):
        unrealized_pnl = latest["total_equity"] - latest["total_balance"] - margin_allocated

    prev_unrealized = 0.0
    if len(state_df) > 1:
        prior = state_df.iloc[-2]
        prev_margin = prior.get("total_margin", 0.0)
        if pd.isna(prev_margin):
            prev_margin = 0.0
        prev_margin = float(prev_margin)
        prev_unrealized = prior.get("net_unrealized_pnl", np.nan)
        if pd.isna(prev_unrealized):
            prev_unrealized = prior["total_equity"] - prior["total_balance"] - prev_margin

    realized_pnl: float | None = None
    initial_equity_series = state_df["total_equity"].dropna()
    if not initial_equity_series.empty:
        initial_equity = float(initial_equity_series.iloc[0])
        realized_pnl = float(latest["total_equity"] - initial_equity)
        if np.isfinite(unrealized_pnl):
            realized_pnl -= float(unrealized_pnl)

    if realized_pnl is None or not np.isfinite(realized_pnl):
        realized_pnl = 0.0
        if not trades_df.empty and "action" in trades_df.columns and "pnl" in trades_df.columns:
            actions = trades_df["action"].fillna("").str.upper()
            realized_pnl = trades_df.loc[actions == "CLOSE", "pnl"].sum(skipna=True)
            if pd.isna(realized_pnl) or not np.isfinite(realized_pnl):
                realized_pnl = 0.0

    sharpe_ratio = compute_sharpe_ratio(trades_df)
    sortino_ratio = compute_sortino_ratio(state_df, RISK_FREE_RATE)

    col_a, col_b, col_c, col_d, col_e, col_f, col_g, col_h = st.columns(8)
    col_a.metric("Available Balance", f"${latest['total_balance']:.2f}")
    col_b.metric("Total Equity", f"${latest['total_equity']:.2f}")
    col_c.metric("Total Return %", f"{latest['total_return_pct']:.2f}%")
    col_d.metric("Margin Allocated", f"${margin_allocated:.2f}")
    col_e.metric(
        "Unrealized PnL",
        f"${unrealized_pnl:.2f}",
        delta=f"${unrealized_pnl - prev_unrealized:.2f}",
    )
    col_f.metric("Realized PnL", f"${realized_pnl:.2f}")
    col_g.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A",
    )
    col_h.metric(
        "Sortino Ratio",
        f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A",
    )

    st.subheader("Equity Over Time (with BTC benchmark)")
    base_investment = 10_000.0

    chart_frames = [
        pd.DataFrame(
            {
                "timestamp": state_df.index,
                "Series": "Portfolio equity",
                "Value": pd.to_numeric(state_df["total_equity"], errors="coerce").values,
            }
        )
    ]

    btc_caption = None
    if "btc_price" in state_df.columns and len(state_df.index) > 0:
        timeline = (
            state_df.reset_index()[["timestamp"]]
            .assign(
                timestamp=lambda df_: pd.to_datetime(
                    df_["timestamp"], errors="coerce", utc=True
                ).dt.tz_convert(None)
            )
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        btc_series = (
            state_df.reset_index()[["timestamp", "btc_price"]]
            .assign(
                timestamp=lambda df_: pd.to_datetime(
                    df_["timestamp"], errors="coerce", utc=True
                ).dt.tz_convert(None),
                btc_price=lambda df_: pd.to_numeric(df_["btc_price"], errors="coerce"),
            )
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        btc_timeline = (
            btc_series.dropna(subset=["btc_price"])
        )
        if not timeline.empty and not btc_timeline.empty:
            benchmark = pd.merge_asof(
                timeline,
                btc_timeline,
                on="timestamp",
                direction="backward",
            )
            benchmark["btc_price"] = benchmark["btc_price"].ffill().bfill()
            valid_prices = benchmark["btc_price"].dropna()
            if not valid_prices.empty:
                base_price = float(valid_prices.iloc[0])
                if base_price > 0:
                    btc_values = base_investment * (benchmark["btc_price"] / base_price)
                    chart_frames.append(
                        pd.DataFrame(
                            {
                                "timestamp": benchmark["timestamp"],
                                "Series": "BTC buy & hold",
                                "Value": btc_values,
                            }
                        )
                    )
                    btc_caption = "BTC benchmark derived from portfolio_state.csv."

    equity_chart_df = pd.concat(chart_frames, ignore_index=True)
    equity_chart_df["timestamp"] = pd.to_datetime(
        equity_chart_df["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    equity_chart_df["Value"] = pd.to_numeric(equity_chart_df["Value"], errors="coerce")
    equity_chart_df.dropna(subset=["timestamp", "Value"], inplace=True)
    equity_chart_df.sort_values("timestamp", inplace=True)

    lower = float(equity_chart_df["Value"].min())
    upper = float(equity_chart_df["Value"].max())
    span = upper - lower
    if span <= 0:
        span = max(upper * 0.02, 1.0)
    padding = span * 0.1
    lower_bound = max(0.0, lower - padding)
    upper_bound = upper + padding

    equity_chart = (
        alt.Chart(equity_chart_df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y(
                "Value:Q",
                title="Value ($)",
                scale=alt.Scale(domain=[lower_bound, upper_bound]),
            ),
            color=alt.Color(
                "Series:N",
                title="Series",
                scale=alt.Scale(
                    domain=["Portfolio equity", "BTC buy & hold"],
                    range=["#f58518", "#4c78a8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="Value", format="$.2f"),
            ],
        )
        .properties(height=280)
        .interactive()
    )
    baseline = (
        alt.Chart(pd.DataFrame({"Value": [base_investment]}))
        .mark_rule(color="#888888", strokeDash=[6, 3])
        .encode(y="Value:Q")
    )
    combined_chart = (equity_chart + baseline).resolve_scale(color='independent')
    st.altair_chart(combined_chart, use_container_width=True)  # type: ignore[arg-type]
    if btc_caption:
        st.caption(btc_caption)

    st.subheader("Open Positions")
    positions_df = parse_positions(latest.get("position_details", ""))
    if positions_df.empty:
        st.write("No open positions.")
    else:
        price_map = fetch_current_prices(positions_df["coin"].unique().tolist())
        positions_df["current_price"] = positions_df["coin"].map(price_map)

        def _row_unrealized(row: pd.Series) -> float | None:
            price = row.get("current_price")
            if price is None or pd.isna(price):
                return None
            diff = price - row["entry_price"]
            if str(row["side"]).lower() == "short":
                diff = row["entry_price"] - price
            return diff * row["quantity"]

        positions_df["unrealized_pnl"] = positions_df.apply(_row_unrealized, axis=1)  # type: ignore

        if positions_df["current_price"].isna().all():
            st.caption("Live price lookup unavailable; showing entry data only.")

        st.dataframe(
            positions_df,
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.4f"),
                "entry_price": st.column_config.NumberColumn(format="$%.4f"),
                "current_price": st.column_config.NumberColumn(format="$%.4f"),
                "unrealized_pnl": st.column_config.NumberColumn(format="$%.2f"),
            },
            use_container_width=True,
        )


def render_trades_tab(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        st.info("No trades recorded yet.")
        return

    st.dataframe(
        trades_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "price": st.column_config.NumberColumn(format="$%.4f"),
            "profit_target": st.column_config.NumberColumn(format="$%.4f"),
            "stop_loss": st.column_config.NumberColumn(format="$%.4f"),
            "pnl": st.column_config.NumberColumn(format="$%.2f"),
            "balance_after": st.column_config.NumberColumn(format="$%.2f"),
        },
        use_container_width=True,
        height=420,
    )


def render_ai_tab(decisions_df: pd.DataFrame, messages_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent AI Decisions")
        if decisions_df.empty:
            st.write("No decisions yet.")
        else:
            st.dataframe(
                decisions_df.head(50),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "confidence": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
            )

    with col2:
        st.subheader("Recent AI Messages")
        if messages_df.empty:
            st.write("No messages logged yet.")
        else:
            st.dataframe(
                messages_df.head(50),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                },
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="DeepSeek Bot Monitor", layout="wide")
    st.title("DeepSeek Trading Bot Monitor")
    st.caption(
        "Source code available at "
        "[github.com/kojott/LLM-trader-test](https://github.com/kojott/LLM-trader-test)"
    )

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    state_df = get_portfolio_state()
    trades_df = get_trades()
    decisions_df = get_ai_decisions()
    messages_df = get_ai_messages()

    portfolio_tab, trades_tab, ai_tab = st.tabs(["Portfolio", "Trades", "AI Activity"])

    with portfolio_tab:
        render_portfolio_tab(state_df, trades_df)

    with trades_tab:
        render_trades_tab(trades_df)

    with ai_tab:
        render_ai_tab(decisions_df, messages_df)


if __name__ == "__main__":
    main()
