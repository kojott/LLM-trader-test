#!/usr/bin/env python3
"""Generate an animated trade replay web page from archived backtest data.

The script reads the CSV/JSON artifacts under ``replay/data`` and emits a
standalone ``index.html`` that visualizes the portfolio curve and trade events.
It purposefully avoids heavy dependencies (such as pandas) so it can run inside
the lightweight Codex environment or any vanilla Python installation.
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List, Sequence

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "index.html"


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("Encountered empty timestamp.")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_records(base_path: Path) -> List[dict]:
    """Load rows from <base>.(csv|json)."""
    csv_path = base_path.with_suffix(".csv")
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    json_path = base_path.with_suffix(".json")
    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            # Accept { "rows": [...] } style payloads.
            if "rows" in payload and isinstance(payload["rows"], list):
                return payload["rows"]
            raise ValueError(
                f"JSON file {json_path} must contain a list of rows, got a dict."
            )
        raise ValueError(f"Unsupported JSON payload type in {json_path}")
    raise FileNotFoundError(
        f"No csv/json file found for {base_path.name} inside {base_path.parent}"
    )


@dataclass
class PortfolioPoint:
    timestamp: str
    dt: datetime
    balance: float | None
    equity: float | None
    return_pct: float | None
    positions: float | None
    btc_price: float | None
    hodl_equity: float | None

    def to_payload(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "balance": self.balance,
            "equity": self.equity,
            "return_pct": self.return_pct,
            "positions": self.positions,
            "btc_price": self.btc_price,
            "hodl_equity": self.hodl_equity,
        }


@dataclass
class TradeEvent:
    timestamp: str
    dt: datetime
    action: str
    side: str
    coin: str
    price: float | None
    quantity: float | None
    pnl: float | None
    balance_after: float | None
    profit_target: float | None
    stop_loss: float | None
    leverage: float | None
    confidence: float | None
    reason: str
    plot_value: float | None

    def to_payload(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "side": self.side,
            "coin": self.coin,
            "price": self.price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "balance_after": self.balance_after,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "leverage": self.leverage,
            "confidence": self.confidence,
            "reason": self.reason,
            "plot_value": self.plot_value,
        }


@dataclass
class CompletedTrade:
    entry_timestamp: str
    exit_timestamp: str | None
    coin: str
    side: str
    entry_price: float | None
    exit_price: float | None
    quantity: float | None
    pnl: float | None
    duration_seconds: float | None
    leverage: float | None
    confidence: float | None
    entry_reason: str
    exit_reason: str

    def to_payload(self) -> dict:
        return {
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "coin": self.coin,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "duration_seconds": self.duration_seconds,
            "leverage": self.leverage,
            "confidence": self.confidence,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }


def build_portfolio_points(records: Sequence[dict]) -> List[PortfolioPoint]:
    points: List[PortfolioPoint] = []
    for row in records:
        raw_ts = row.get("timestamp")
        if not raw_ts:
            continue
        dt = parse_timestamp(str(raw_ts))
        points.append(
            PortfolioPoint(
                timestamp=dt.isoformat(),
                dt=dt,
                balance=to_float(row.get("total_balance")),
                equity=to_float(row.get("total_equity")),
                return_pct=to_float(row.get("total_return_pct")),
                positions=to_float(row.get("num_positions")),
                btc_price=to_float(row.get("btc_price")),
                hodl_equity=None,
            )
        )
    points.sort(key=lambda point: point.dt)
    if not points:
        raise ValueError("portfolio_state dataset is empty – nothing to replay.")

    btc_units = None
    for point in points:
        if btc_units is None and point.equity is not None and point.btc_price not in (None, 0):
            btc_units = point.equity / point.btc_price
        if btc_units is not None and point.btc_price not in (None, 0):
            point.hodl_equity = btc_units * point.btc_price
        else:
            point.hodl_equity = None

    return points


def infer_plot_value(trade_dt: datetime, timeline: List[datetime], values: List[float | None]) -> float | None:
    if not timeline:
        return None
    idx = bisect_right(timeline, trade_dt) - 1
    if idx < 0:
        return None
    candidate = values[idx]
    return candidate if candidate is not None else None


def build_trade_events(records: Sequence[dict], portfolio_points: List[PortfolioPoint]) -> List[TradeEvent]:
    timeline = [point.dt for point in portfolio_points]
    equity_values = [
        point.equity if point.equity is not None else point.balance for point in portfolio_points
    ]

    events: List[TradeEvent] = []
    for row in records:
        raw_ts = row.get("timestamp")
        if not raw_ts:
            continue
        dt = parse_timestamp(str(raw_ts))
        balance_after = to_float(row.get("balance_after"))
        plot_value = balance_after
        if plot_value is None:
            plot_value = infer_plot_value(dt, timeline, equity_values)
        events.append(
            TradeEvent(
                timestamp=dt.isoformat(),
                dt=dt,
                action=str(row.get("action", "")).upper(),
                side=str(row.get("side", "")).upper(),
                coin=str(row.get("coin", "")),
                price=to_float(row.get("price")),
                quantity=to_float(row.get("quantity")),
                pnl=to_float(row.get("pnl")),
                balance_after=balance_after,
                profit_target=to_float(row.get("profit_target")),
                stop_loss=to_float(row.get("stop_loss")),
                leverage=to_float(row.get("leverage")),
                confidence=to_float(row.get("confidence")),
                reason=str(row.get("reason", "")),
                plot_value=plot_value,
            )
        )
    events.sort(key=lambda event: event.dt)
    if not events:
        raise ValueError("trade_history dataset is empty – nothing to replay.")
    return events


def pair_trade_events(events: List[TradeEvent]) -> List[CompletedTrade]:
    open_positions: dict[tuple[str, str], deque[TradeEvent]] = {}
    completed: List[CompletedTrade] = []

    for event in events:
        key = (event.coin, event.side)
        if event.action == "ENTRY":
            open_positions.setdefault(key, deque()).append(event)
        elif event.action == "CLOSE":
            queue = open_positions.get(key)
            entry_event = queue.popleft() if queue else None
            duration_seconds = None
            entry_timestamp = entry_event.timestamp if entry_event else event.timestamp
            if entry_event:
                duration_seconds = (event.dt - entry_event.dt).total_seconds()
            completed.append(
                CompletedTrade(
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=event.timestamp,
                    coin=event.coin,
                    side=event.side,
                    entry_price=entry_event.price if entry_event else None,
                    exit_price=event.price,
                    quantity=entry_event.quantity if entry_event else event.quantity,
                    pnl=event.pnl,
                    duration_seconds=duration_seconds,
                    leverage=entry_event.leverage if entry_event else event.leverage,
                    confidence=entry_event.confidence if entry_event else event.confidence,
                    entry_reason=entry_event.reason if entry_event else "",
                    exit_reason=event.reason,
                )
            )

    for queue in open_positions.values():
        while queue:
            entry_event = queue.popleft()
            completed.append(
                CompletedTrade(
                    entry_timestamp=entry_event.timestamp,
                    exit_timestamp=None,
                    coin=entry_event.coin,
                    side=entry_event.side,
                    entry_price=entry_event.price,
                    exit_price=None,
                    quantity=entry_event.quantity,
                    pnl=None,
                    duration_seconds=None,
                    leverage=entry_event.leverage,
                    confidence=entry_event.confidence,
                    entry_reason=entry_event.reason,
                    exit_reason="Open position",
                )
            )

    completed.sort(key=lambda trade: trade.exit_timestamp or trade.entry_timestamp)
    return completed


def safe_currency(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def safe_percent(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def compute_stats(
    portfolio_points: List[PortfolioPoint],
    trades: List[TradeEvent],
    completed_trades: List[CompletedTrade],
) -> dict:
    def values_from_points(attribute: str) -> List[float]:
        vals = []
        for point in portfolio_points:
            val = getattr(point, attribute, None)
            if val is None:
                continue
            vals.append(val)
        return vals

    def first_equity_value() -> float | None:
        for point in portfolio_points:
            candidate = point.equity if point.equity is not None else point.balance
            if candidate is not None:
                return candidate
        return None

    def last_equity_value() -> float | None:
        for point in reversed(portfolio_points):
            candidate = point.equity if point.equity is not None else point.balance
            if candidate is not None:
                return candidate
        return None

    equities = [
        point.equity if point.equity is not None else point.balance
        for point in portfolio_points
    ]
    equities = [val for val in equities if val is not None]

    max_drawdown_pct = None
    if equities:
        peak = equities[0]
        worst = 0.0
        for val in equities:
            if val > peak:
                peak = val
            drawdown = (peak - val) / peak if peak else 0.0
            if drawdown > worst:
                worst = drawdown
        max_drawdown_pct = worst * 100.0

    start_equity = first_equity_value()
    end_equity = last_equity_value()

    net_return_pct = None
    if start_equity not in (None, 0) and end_equity is not None:
        net_return_pct = ((end_equity / start_equity) - 1.0) * 100.0

    hodl_values = values_from_points("hodl_equity")
    hodl_start = hodl_values[0] if hodl_values else None
    hodl_end = hodl_values[-1] if hodl_values else None
    hodl_return_pct = None
    if hodl_start not in (None, 0) and hodl_end is not None:
        hodl_return_pct = ((hodl_end / hodl_start) - 1.0) * 100.0

    alpha_vs_hodl = None
    if net_return_pct is not None and hodl_return_pct is not None:
        alpha_vs_hodl = net_return_pct - hodl_return_pct

    pnl_values = [trade.pnl for trade in completed_trades if trade.pnl is not None]
    win_rate = None
    if pnl_values:
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        win_rate = (wins / len(pnl_values)) * 100.0

    runtime_hours = None
    if portfolio_points:
        duration = (
            portfolio_points[-1].dt - portfolio_points[0].dt
        ).total_seconds() / 3600.0
        runtime_hours = max(duration, 0.0)

    return {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "net_return_pct": net_return_pct,
        "hodl_end": hodl_end,
        "hodl_return_pct": hodl_return_pct,
        "alpha_vs_hodl": alpha_vs_hodl,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": len(completed_trades),
        "win_rate": win_rate,
        "runtime_hours": runtime_hours,
    }


def render_html(
    portfolio_points: List[PortfolioPoint],
    trades: List[TradeEvent],
    completed_trades: List[CompletedTrade],
    data_label: str,
    stats: dict,
) -> str:
    portfolio_payload = json.dumps([point.to_payload() for point in portfolio_points])
    trade_payload = json.dumps([trade.to_payload() for trade in trades])
    completed_payload = json.dumps([trade.to_payload() for trade in completed_trades])
    max_frame = max(len(portfolio_points) - 1, 0)

    final_equity = safe_currency(stats.get("end_equity"))
    net_return = safe_percent(stats.get("net_return_pct"))
    hodl_equity = safe_currency(stats.get("hodl_end"))
    hodl_return = safe_percent(stats.get("hodl_return_pct"))
    max_dd = safe_percent(stats.get("max_drawdown_pct"), signed=False)
    win_rate = safe_percent(stats.get("win_rate"))
    alpha_return = safe_percent(stats.get("alpha_vs_hodl"))
    total_trades = stats.get("total_trades")
    runtime_hours = stats.get("runtime_hours")
    runtime_text = "—"
    if runtime_hours is not None:
        if runtime_hours > 72:
            runtime_text = f"{runtime_hours/24:.1f} days"
        else:
            runtime_text = f"{runtime_hours:.1f} hrs"
    max_dd_display = "—" if max_dd == "—" else f"-{max_dd}"
    total_trades_text = "—" if total_trades is None else f"{total_trades:,}"
    alpha_display = alpha_return

    return dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Trade Replay</title>
          <link rel="preconnect" href="https://fonts.googleapis.com">
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
          <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
          <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
          
          
          <style>
            :root {{
              color-scheme: dark;
              font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
              background-color: #010409;
              color: #f8fafc;
            }}
            * {{
              box-sizing: border-box;
            }}
            body {{
              margin: 0;
              min-height: 100vh;
              display: flex;
              justify-content: center;
              align-items: flex-start;
              padding: 2.5rem clamp(1.25rem, 4vw, 3rem);
              background:
                radial-gradient(circle at 20% 20%, rgba(1,195,141,0.18), transparent 55%),
                radial-gradient(circle at 80% 10%, rgba(13,191,203,0.15), transparent 45%),
                #010409;
              overflow-x: hidden;
            }}
            .aurora {{
              position: fixed;
              inset: 0;
              pointer-events: none;
              z-index: 0;
              mix-blend-mode: screen;
              opacity: 0.35;
              filter: blur(120px);
              animation: drift 24s linear infinite;
            }}
            .aurora-1 {{
              background: radial-gradient(circle at 20% 20%, rgba(1,195,141,0.8), transparent 60%);
            }}
            .aurora-2 {{
              background: radial-gradient(circle at 80% 10%, rgba(13,191,203,0.7), transparent 55%);
              animation-duration: 28s;
            }}
            .aurora-3 {{
              background: radial-gradient(circle at 60% 80%, rgba(251,191,36,0.5), transparent 60%);
              animation-duration: 32s;
            }}
            @keyframes drift {{
              0% {{ transform: translate3d(0,0,0) scale(1); }}
              50% {{ transform: translate3d(-5%, -3%, 0) scale(1.1); }}
              100% {{ transform: translate3d(0,0,0) scale(1); }}
            }}
            .app {{
              width: min(1280px, 100%);
              display: flex;
              flex-direction: column;
              gap: 2rem;
              position: relative;
              z-index: 1;
            }}
            .card {{
              background: rgba(7,12,24,0.9);
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 24px;
              padding: clamp(1.25rem, 2vw, 2rem);
              box-shadow: 0 20px 60px rgba(2,6,23,0.7);
              backdrop-filter: blur(18px);
            }}
            .hero {{
              display: flex;
              flex-direction: column;
              gap: 1.5rem;
              position: relative;
              overflow: hidden;
            }}
            .hero::after {{
              content: "";
              position: absolute;
              inset: 0;
              background: linear-gradient(135deg, rgba(1,195,141,0.08), rgba(13,191,203,0));
              opacity: 0.8;
              pointer-events: none;
            }}
            .hero-badge {{
              display: inline-flex;
              align-items: center;
              gap: 0.4rem;
              padding: 0.3rem 0.9rem;
              border-radius: 999px;
              border: 1px solid rgba(1,195,141,0.5);
              color: #8fffe0;
              font-size: 0.85rem;
              letter-spacing: 0.05em;
              text-transform: uppercase;
            }}
            .hero h1 {{
              margin: 0;
              font-size: clamp(1.9rem, 3vw, 2.8rem);
            }}
            .hero p {{
              margin: 0;
              color: rgba(248,250,252,0.78);
              max-width: 640px;
            }}
            .stat-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 1rem;
              position: relative;
              z-index: 1;
            }}
            .stat-card {{
              padding: 1rem 1.2rem;
              border-radius: 18px;
              border: 1px solid rgba(255,255,255,0.08);
              background: rgba(13,19,35,0.85);
              display: flex;
              flex-direction: column;
              gap: 0.3rem;
            }}
            .stat-card.primary {{
              background: linear-gradient(135deg, rgba(1,195,141,0.35), rgba(13,191,203,0.15));
              border-color: rgba(1,195,141,0.5);
              box-shadow: 0 15px 40px rgba(1,195,141,0.25);
            }}
            .stat-card span {{
              font-size: 0.75rem;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: rgba(248,250,252,0.65);
            }}
            .stat-card strong {{
              font-size: 1.6rem;
              font-weight: 600;
            }}
            .stat-card small {{
              font-size: 0.85rem;
              color: rgba(248,250,252,0.7);
            }}
            .chart-card {{
              position: relative;
              overflow: hidden;
            }}
            .chart-card::after {{
              content: "";
              position: absolute;
              inset: 0;
              pointer-events: none;
              background: radial-gradient(circle at 30% 0%, rgba(255,255,255,0.05), transparent 55%);
            }}
            canvas {{
              width: 100% !important;
              height: 460px !important;
            }}
            .controls {{
              display: flex;
              flex-wrap: wrap;
              gap: 1rem;
              align-items: center;
              margin-top: 1.25rem;
            }}
            button {{
              background: linear-gradient(135deg, #01c38d, #0dbfcb);
              border: none;
              color: #05060b;
              font-weight: 600;
              padding: 0.7rem 1.6rem;
              border-radius: 999px;
              cursor: pointer;
              transition: transform 0.15s ease, box-shadow 0.15s ease;
              box-shadow: 0 12px 30px rgba(13,191,203,0.28);
            }}
            button:disabled {{
              opacity: 0.4;
              cursor: not-allowed;
              box-shadow: none;
            }}
            button:not(:disabled):active {{
              transform: scale(0.96);
            }}
            input[type="range"] {{
              flex: 1 1 240px;
              accent-color: #01c38d;
            }}
            select {{
              background: #0f1625;
              border-radius: 999px;
              border: 1px solid rgba(255,255,255,0.12);
              padding: 0.45rem 0.9rem;
              color: inherit;
            }}
            .details {{
              display: grid;
              gap: 1.5rem;
              grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            }}
            .trade-card {{
              display: flex;
              flex-direction: column;
              gap: 1rem;
            }}
            .status-panel {{
              display: grid;
              gap: 0.8rem;
              grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
              background: linear-gradient(145deg, rgba(1,195,141,0.08), rgba(2,6,23,0.9));
              border: 1px solid rgba(1,195,141,0.25);
            }}
            .status-block {{
              display: flex;
              flex-direction: column;
              gap: 0.2rem;
            }}
            .status-block span {{
              font-size: 0.75rem;
              color: rgba(255,255,255,0.65);
              text-transform: uppercase;
              letter-spacing: 0.07em;
            }}
            .status-value {{
              font-size: 1.3rem;
              font-weight: 600;
            }}
            .status-meta {{
              font-size: 0.85rem;
              color: rgba(255,255,255,0.65);
            }}
            .trade-log {{
              max-height: 300px;
              overflow: auto;
              border-top: 1px solid rgba(255,255,255,0.08);
            }}
            .trade-item {{
              display: flex;
              justify-content: space-between;
              border-bottom: 1px solid rgba(255,255,255,0.04);
              padding: 0.6rem 0;
              gap: 1rem;
            }}
            .trade-item:last-child {{
              border-bottom: none;
            }}
            .trade-item .label {{
              font-weight: 600;
              font-size: 0.95rem;
            }}
            .trade-item .meta {{
              font-size: 0.85rem;
              color: rgba(255,255,255,0.65);
            }}
            .pnl-value {{
              font-size: 1.2rem;
              font-weight: 600;
              display: flex;
              align-items: center;
            }}
            .pnl-value.pos {{
              color: #00f5a0;
            }}
            .pnl-value.neg {{
              color: #ff5f8f;
            }}
            .trade-log::-webkit-scrollbar {{
              width: 6px;
            }}
            .trade-log::-webkit-scrollbar-thumb {{
              background: rgba(255,255,255,0.18);
              border-radius: 999px;
            }}
            .time-label {{
              font-size: 0.95rem;
              color: rgba(255,255,255,0.75);
            }}
            @media (max-width: 720px) {{
              body {{
                padding: 1rem;
              }}
              canvas {{
                height: 320px !important;
              }}
              .controls {{
                flex-direction: column;
                align-items: stretch;
              }}
              button,
              select {{
                width: 100%;
                justify-content: center;
              }}
              .stat-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              }}
            }}
          </style>


        </head>
        <body>
          <div class="aurora aurora-1"></div>
          <div class="aurora aurora-2"></div>
          <div class="aurora aurora-3"></div>
          <main class="app">
            <header class="hero card">
              <div class="hero-badge">Session Replay</div>
              <div>
                <h1>Backtest Trade Replay</h1>
                <p>Generated locally from the data in <code>{data_label}</code>. Use the controls to scrub through the session.</p>
              </div>
              <div class="stat-grid">
                <div class="stat-card primary">
                  <span>Final Equity</span>
                  <strong>{final_equity}</strong>
                  <small>Net return {net_return}</small>
                </div>
                <div class="stat-card">
                  <span>BTC HODL Benchmark</span>
                  <strong>{hodl_equity}</strong>
                  <small>HODL return {hodl_return}</small>
                </div>
                <div class="stat-card">
                  <span>Alpha vs HODL</span>
                  <strong>{alpha_display}</strong>
                  <small>Strategy edge vs buy-and-hold</small>
                </div>
                <div class="stat-card">
                  <span>Trades Executed</span>
                  <strong>{total_trades_text}</strong>
                  <small>Win rate {win_rate}</small>
                </div>
                <div class="stat-card">
                  <span>Max Drawdown</span>
                  <strong>{max_dd_display}</strong>
                  <small>Peak-to-valley damage</small>
                </div>
                <div class="stat-card">
                  <span>Session Duration</span>
                  <strong>{runtime_text}</strong>
                  <small>Captured timeline</small>
                </div>
              </div>
            </header>
            <section class="card chart-card">
              <canvas id="replayChart"></canvas>
              <div class="controls">
                <button id="playButton">Play</button>
                <input id="frameSlider" type="range" min="0" max="{max_frame}" value="0">
                <label>
                  Speed
                  <select id="speedSelect">
                    <option value="1">1x</option>
                    <option value="2">2x</option>
                    <option value="4">4x</option>
                    <option value="8">8x</option>
                    <option value="10">10x</option>
                    <option value="50">50x</option>
                    <option value="100">100x</option>
                  </select>
                </label>
                <span id="timeLabel" class="time-label"></span>
              </div>
            </section>
            <section class="details">
              <div class="card status-panel" id="statusPanel">
                <div class="status-block">
                  <span>Time</span>
                  <div class="status-value" id="statusTime">–</div>
                </div>
                <div class="status-block">
                  <span>Equity</span>
                  <div class="status-value" id="statusEquity">–</div>
                </div>
                <div class="status-block">
                  <span>BTC HODL</span>
                  <div class="status-value" id="statusHodl">–</div>
                </div>
                <div class="status-block">
                  <span>Last Trade</span>
                  <div class="status-value" id="statusTrade">No trades yet</div>
                </div>
              </div>
              <div class="card trade-card">
                <h2 style="margin-top:0;">Timeline</h2>
                <div class="trade-log" id="tradeLog"></div>
              </div>
            </section>
          </main>


          <script>
            const portfolioPoints = {portfolio_payload};
            const tradeEvents = {trade_payload};
            const completedTrades = {completed_payload};
            const ctx = document.getElementById('replayChart').getContext('2d');

            const equityDataset = {{
              type: 'line',
              label: 'Equity',
              data: [],
              borderColor: '#00e6a8',
              backgroundColor: 'rgba(0, 230, 168, 0.08)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
              borderWidth: 2
            }};

            const balanceDataset = {{
              type: 'line',
              label: 'Balance',
              data: [],
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59,130,246,0.05)',
              fill: false,
              tension: 0.15,
              pointRadius: 0,
              borderWidth: 1.2,
              borderDash: [6, 3]
            }};

            const hodlDataset = {{
              type: 'line',
              label: 'BTC HODL',
              data: [],
              borderColor: '#fbbf24',
              backgroundColor: 'rgba(251,191,36,0.08)',
              fill: false,
              tension: 0.25,
              pointRadius: 0,
              borderWidth: 1.6,
              borderDash: [3, 3]
            }};

            const tradesDataset = {{
              type: 'scatter',
              label: 'Trades',
              data: [],
              parsing: false,
              pointRadius: ctx => ctx.raw?.action === 'CLOSE' ? 6 : 4,
              pointHoverRadius: 8,
              pointBackgroundColor: ctx => {{
                if (!ctx.raw) return '#ffffff';
                if (ctx.raw.action === 'ENTRY') return ctx.raw.side === 'LONG' ? '#22d3ee' : '#f97316';
                return ctx.raw.pnl >= 0 ? '#00ff9d' : '#ff4d6d';
              }},
              pointBorderColor: 'rgba(0,0,0,0.6)',
              pointBorderWidth: 1,
            }};

            const chart = new Chart(ctx, {{
              data: {{
                datasets: [equityDataset, balanceDataset, hodlDataset, tradesDataset]
              }},
              options: {{
                responsive: true,
                animation: false,
                interaction: {{
                  mode: 'nearest',
                  intersect: false,
                }},
                scales: {{
                  x: {{
                    type: 'time',
                    time: {{
                      tooltipFormat: 'yyyy-MM-dd HH:mm'
                    }},
                    grid: {{
                      color: 'rgba(255,255,255,0.05)'
                    }},
                    ticks: {{
                      color: 'rgba(255,255,255,0.7)'
                    }}
                  }},
                  y: {{
                    title: {{
                      display: true,
                      text: 'USD'
                    }},
                    grid: {{
                      color: 'rgba(255,255,255,0.05)'
                    }},
                    ticks: {{
                      color: 'rgba(255,255,255,0.7)'
                    }}
                  }}
                }},
                plugins: {{
                  legend: {{
                    position: 'top',
                    labels: {{
                      usePointStyle: true,
                    }}
                  }},
                  tooltip: {{
                    callbacks: {{
                      label: ctx => {{
                        if (ctx.dataset.type === 'scatter') {{
                          const raw = ctx.raw;
                          return `${{raw.action}} ${{raw.coin}} @ ${{raw.price ?? '—'}} (PnL ${{formatUsd(raw.pnl)}})`;
                        }}
                        return `${{ctx.dataset.label}}: ${{formatUsd(ctx.parsed.y)}}`;
                      }}
                    }}
                  }}
                }}
              }}
            }});

            const playButton = document.getElementById('playButton');
            const frameSlider = document.getElementById('frameSlider');
            const speedSelect = document.getElementById('speedSelect');
            const timeLabel = document.getElementById('timeLabel');
            const statusTime = document.getElementById('statusTime');
            const statusEquity = document.getElementById('statusEquity');
            const statusHodl = document.getElementById('statusHodl');
            const statusTrade = document.getElementById('statusTrade');
            const tradeLog = document.getElementById('tradeLog');

            let currentFrame = 0;
            let playing = false;
            let rafId = null;
            const maxFrame = Number(frameSlider.max);

            const speeds = {{
              1: 650,
              2: 420,
              4: 220,
              8: 120,
              10: 80,
              50: 30,
              100: 15
            }};

            function formatUsd(value) {{
              if (value == null || isNaN(value)) return '—';
              return Number(value).toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
            }}

            function formatUsdDisplay(value) {{
              const formatted = formatUsd(value);
              return formatted === '—' ? '—' : '$' + formatted;
            }}

            function formatTimestamp(value) {{
              if (!value) return '—';
              return new Date(value).toLocaleString();
            }}

            function formatDuration(seconds) {{
              if (seconds == null || isNaN(seconds)) return '—';
              const abs = Math.abs(seconds);
              if (abs >= 86400) {{
                return `${{(abs / 86400).toFixed(1)}} d`;
              }}
              if (abs >= 3600) {{
                return `${{(abs / 3600).toFixed(1)}} h`;
              }}
              return `${{(abs / 60).toFixed(0)}} m`;
            }}

            function updateStatus(point, latestTrade) {{
              statusTime.textContent = new Date(point.timestamp).toLocaleString();
              statusEquity.textContent = formatUsdDisplay(point.equity ?? point.balance);
              statusHodl.textContent = formatUsdDisplay(point.hodl_equity);
              if (latestTrade) {{
                const pnlStr = formatUsdDisplay(latestTrade.pnl);
                const durationStr = formatDuration(latestTrade.duration_seconds);
                statusTrade.innerHTML = `<strong>${{latestTrade.coin}} ${{latestTrade.side}}</strong><br><span class="status-meta">PnL ${{pnlStr}} · ${{durationStr}}</span>`;
              }} else {{
                statusTrade.textContent = 'No trades yet';
              }}
            }}

            function renderTradeLog(latestTime) {{
              const visible = completedTrades.filter(trade => {{
                const marker = trade.exit_timestamp || trade.entry_timestamp;
                return new Date(marker) <= latestTime;
              }});
              tradeLog.innerHTML = visible
                .slice(-25)
                .reverse()
                .map(trade => {{
                  const pnlClass = trade.pnl == null ? '' : (trade.pnl >= 0 ? 'pos' : 'neg');
                  return `
                    <div class="trade-item">
                      <div>
                        <div class="label">${{trade.coin}} ${{trade.side}}</div>
                        <div class="meta">Entry ${{formatTimestamp(trade.entry_timestamp)}} @ ${{formatUsdDisplay(trade.entry_price)}}</div>
                        <div class="meta">Exit ${{formatTimestamp(trade.exit_timestamp)}} @ ${{formatUsdDisplay(trade.exit_price)}} · Duration ${{formatDuration(trade.duration_seconds)}}</div>
                      </div>
                      <div class="pnl-value ${{pnlClass}}">${{formatUsdDisplay(trade.pnl)}}</div>
                    </div>
                  `;
                }}).join('');
            }}

            function sliceSeries(frameIdx) {{
              const segment = portfolioPoints.slice(0, frameIdx + 1);
              equityDataset.data = segment
                .filter(point => point.equity != null)
                .map(point => ({{ x: point.timestamp, y: point.equity }}));
              balanceDataset.data = segment
                .filter(point => point.balance != null)
                .map(point => ({{ x: point.timestamp, y: point.balance }}));
              const cutoff = new Date(segment[segment.length - 1].timestamp);
              hodlDataset.data = segment
                .filter(point => point.hodl_equity != null)
                .map(point => ({{ x: point.timestamp, y: point.hodl_equity }}));
              tradesDataset.data = tradeEvents
                .filter(evt => new Date(evt.timestamp) <= cutoff)
                .map(evt => ({{ x: evt.timestamp, y: evt.plot_value, ...evt }}));
              chart.update('none');
              const latestTrade = (() => {{
                for (let i = completedTrades.length - 1; i >= 0; i--) {{
                  const trade = completedTrades[i];
                  const marker = trade.exit_timestamp || trade.entry_timestamp;
                  if (new Date(marker) <= cutoff) {{
                    return trade;
                  }}
                }}
                return null;
              }})();
              updateStatus(segment[segment.length - 1], latestTrade);
              renderTradeLog(cutoff);
              timeLabel.textContent = cutoff.toLocaleString();
            }}

            function step(timestamp) {{
              if (!playing) return;
              const delay = speeds[speedSelect.value] ?? speeds[1];
              if (!step.lastTime || timestamp - step.lastTime >= delay) {{
                currentFrame = Math.min(currentFrame + 1, maxFrame);
                frameSlider.value = currentFrame;
                sliceSeries(currentFrame);
                step.lastTime = timestamp;
                if (currentFrame === maxFrame) {{
                  playing = false;
                  playButton.textContent = 'Replay';
                  return;
                }}
              }}
              rafId = requestAnimationFrame(step);
            }}

            playButton.addEventListener('click', () => {{
              if (playing) {{
                playing = false;
                playButton.textContent = 'Play';
                if (rafId) cancelAnimationFrame(rafId);
                return;
              }}
              if (currentFrame === maxFrame) {{
                currentFrame = 0;
                frameSlider.value = 0;
                sliceSeries(0);
              }}
              playing = true;
              playButton.textContent = 'Pause';
              step.lastTime = null;
              rafId = requestAnimationFrame(step);
            }});

            frameSlider.addEventListener('input', (event) => {{
              currentFrame = Number(event.target.value);
              sliceSeries(currentFrame);
            }});

            window.addEventListener('keydown', (event) => {{
              if (event.code === 'Space') {{
                event.preventDefault();
                playButton.click();
              }}
            }});

            sliceSeries(0);
            renderTradeLog(new Date(portfolioPoints[0].timestamp));
          </script>
        </body>
        </html>
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the trade replay webpage.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory that contains portfolio_state/trade_history files (default: replay/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write the HTML file (default: replay/index.html)",
    )
    args = parser.parse_args()

    data_dir = args.data.expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist.")

    portfolio_records = load_records(data_dir / "portfolio_state")
    trade_records = load_records(data_dir / "trade_history")

    portfolio_points = build_portfolio_points(portfolio_records)
    trades = build_trade_events(trade_records, portfolio_points)
    completed_trades = pair_trade_events(trades)
    stats = compute_stats(portfolio_points, trades, completed_trades)

    data_label = data_dir.name or str(data_dir)
    html = render_html(portfolio_points, trades, completed_trades, data_label, stats)
    args.output.write_text(html, encoding="utf-8")
    print(f"Trade replay written to {args.output}")


if __name__ == "__main__":
    main()
