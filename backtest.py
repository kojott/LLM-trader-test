#!/usr/bin/env python3
"""
Backtesting harness for the DeepSeek Multi-Asset Trading Bot.

It replays historical Binance data, calls the LLM for decisions on each bar,
and reuses the live trading execution and logging pipeline while writing output
to an isolated data directory per run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# Columns returned by Binance kline endpoints
KLINE_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]

DEFAULT_INTERVAL = "3m"
LONG_CONTEXT_INTERVAL = "4h"
INTERVAL_TO_DELTA = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
}
SUPPORTED_INTERVALS = tuple(INTERVAL_TO_DELTA.keys())
WARMUP_BARS = {
    "1m": 500,
    DEFAULT_INTERVAL: 300,
    "5m": 240,
    "15m": 200,
    "30m": 180,
    "1h": 150,
    LONG_CONTEXT_INTERVAL: 120,
}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKTEST_DIR = PROJECT_ROOT / "data-backtest"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_datetime(value: Optional[str], fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        parsed = pd.to_datetime(value, utc=True)
    except Exception as exc:  # pragma: no cover - parsing guard
        logging.warning("Failed to parse datetime '%s': %s; using fallback %s", value, exc, fallback)
        return fallback
    if isinstance(parsed, pd.Series):
        parsed = parsed.iloc[0]
    if isinstance(parsed, pd.Timestamp):
        parsed = parsed.to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def interval_to_timedelta(interval: str) -> timedelta:
    try:
        return INTERVAL_TO_DELTA[interval]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported interval '{interval}'") from exc


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    normalized = df.copy()
    for col in ("timestamp", "close_time", "trades"):
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    float_columns = ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]
    for col in float_columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized.dropna(subset=["timestamp"], inplace=True)
    normalized["timestamp"] = normalized["timestamp"].astype(np.int64)
    normalized["close_time"] = normalized["close_time"].fillna(normalized["timestamp"]).astype(np.int64)
    normalized["trades"] = normalized["trades"].fillna(0).astype(int)
    normalized.sort_values("timestamp", inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


@dataclass
class BacktestConfig:
    start: datetime
    end: datetime
    interval: str
    base_dir: Path
    run_dir: Path
    cache_dir: Path
    run_id: str
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    thinking: Optional[str]
    system_prompt: Optional[str]
    system_prompt_file: Optional[str]
    start_capital: Optional[float]
    disable_telegram: bool

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)

    @staticmethod
    def from_environment() -> "BacktestConfig":
        now_utc = datetime.now(timezone.utc)
        default_end = now_utc
        default_start = default_end - timedelta(days=7)

        start = ensure_utc(parse_datetime(os.getenv("BACKTEST_START"), default_start))
        end = ensure_utc(parse_datetime(os.getenv("BACKTEST_END"), default_end))
        if start >= end:
            raise ValueError("BACKTEST_START must be earlier than BACKTEST_END")

        interval = os.getenv("BACKTEST_INTERVAL", DEFAULT_INTERVAL).lower()
        if interval not in SUPPORTED_INTERVALS:
            logging.warning(
                "Interval %s not explicitly supported; defaulting to %s",
                interval,
                DEFAULT_INTERVAL,
            )
            interval = DEFAULT_INTERVAL

        base_dir_raw = os.getenv("BACKTEST_DATA_DIR")
        if base_dir_raw:
            base_dir = Path(base_dir_raw).expanduser()
            if not base_dir.is_absolute():
                base_dir = (PROJECT_ROOT / base_dir).resolve()
        else:
            base_dir = DEFAULT_BACKTEST_DIR
        cache_dir = base_dir / "cache"

        run_id = os.getenv("BACKTEST_RUN_ID")
        if not run_id:
            run_id = f"run-{now_utc.strftime('%Y%m%d-%H%M%S')}"
        run_dir = base_dir / run_id

        model = os.getenv("BACKTEST_LLM_MODEL")
        if model is None:
            model = os.getenv("BACKTEST_MODEL")

        temperature_raw = os.getenv("BACKTEST_TEMPERATURE")
        temperature = None
        if temperature_raw:
            try:
                temperature = float(temperature_raw)
            except ValueError:
                logging.warning("Invalid BACKTEST_TEMPERATURE '%s'; ignoring.", temperature_raw)

        max_tokens_raw = os.getenv("BACKTEST_MAX_TOKENS")
        max_tokens = None
        if max_tokens_raw:
            try:
                max_tokens = int(max_tokens_raw)
            except ValueError:
                logging.warning("Invalid BACKTEST_MAX_TOKENS '%s'; ignoring.", max_tokens_raw)

        thinking_raw = os.getenv("BACKTEST_LLM_THINKING")
        if thinking_raw is None:
            thinking_raw = os.getenv("BACKTEST_THINKING")
        thinking = None
        if thinking_raw is not None:
            thinking_raw = thinking_raw.strip()
            if thinking_raw:
                thinking = thinking_raw

        system_prompt_file_raw = os.getenv("BACKTEST_SYSTEM_PROMPT_FILE")
        system_prompt_file = None
        if system_prompt_file_raw:
            prompt_path = Path(system_prompt_file_raw).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = (PROJECT_ROOT / prompt_path).resolve()
            system_prompt_file = str(prompt_path)

        system_prompt = os.getenv("BACKTEST_SYSTEM_PROMPT")

        start_capital_raw = os.getenv("BACKTEST_START_CAPITAL")
        start_capital = None
        if start_capital_raw:
            try:
                start_capital = float(start_capital_raw)
            except ValueError:
                logging.warning("Invalid BACKTEST_START_CAPITAL '%s'; ignoring.", start_capital_raw)

        disable_telegram = os.getenv("BACKTEST_DISABLE_TELEGRAM", "true").strip().lower() in {"1", "true", "yes", "on"}

        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        return BacktestConfig(
            start=start,
            end=end,
            interval=interval,
            base_dir=base_dir,
            run_dir=run_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
            start_capital=start_capital,
            disable_telegram=disable_telegram,
        )


def ensure_cached_klines(
    client: Client,
    cfg: BacktestConfig,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    warmup = WARMUP_BARS.get(interval, 0)
    interval_delta = interval_to_timedelta(interval)
    start_with_buffer = cfg.start - interval_delta * warmup
    start_with_buffer = ensure_utc(start_with_buffer)
    end_with_buffer = ensure_utc(cfg.end)

    cache_path = cfg.cache_dir / f"{symbol}_{interval}.csv"
    if cache_path.exists():
        cached = normalize_kline_dataframe(pd.read_csv(cache_path))
    else:
        cached = pd.DataFrame(columns=KLINE_COLUMNS)

    start_ms_required = int(start_with_buffer.timestamp() * 1000)
    end_ms_required = int(end_with_buffer.timestamp() * 1000)

    have_coverage = False
    if not cached.empty:
        cached_start = int(cached["timestamp"].min())
        cached_end = int(cached["timestamp"].max())
        have_coverage = cached_start <= start_ms_required and cached_end >= end_ms_required

    if not have_coverage:
        start_str = start_with_buffer.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_with_buffer.strftime("%Y-%m-%d %H:%M:%S")
        logging.info("Downloading %s %s klines from Binance (%s → %s)...", symbol, interval, start_str, end_str)
        # Pass millisecond timestamps to avoid ambiguous string date parsing in python-binance.
        klines = client.get_historical_klines(symbol, interval, start_ms_required, end_ms_required)
        fetched = normalize_kline_dataframe(pd.DataFrame(klines, columns=KLINE_COLUMNS))
        if cached.empty:
            cached = fetched
        else:
            cached = pd.concat([cached, fetched], ignore_index=True)
            cached.drop_duplicates(subset="timestamp", keep="last", inplace=True)
            cached.sort_values("timestamp", inplace=True)
            cached.reset_index(drop=True, inplace=True)
        cached.to_csv(cache_path, index=False)

    trimmed = cached[cached["timestamp"] <= end_ms_required].copy()
    trimmed.reset_index(drop=True, inplace=True)
    return trimmed


class HistoricalBinanceClient:
    """Minimal Binance client shim that replays cached klines."""

    def __init__(self, frames: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        self._frames = frames
        self._current_timestamp_ms: Optional[int] = None
        self._indices: Dict[str, Dict[str, Optional[int]]] = {
            symbol: {interval: None for interval in intervals}
            for symbol, intervals in frames.items()
        }

    def set_current_timestamp(self, timestamp_ms: int) -> None:
        self._current_timestamp_ms = timestamp_ms
        for symbol, interval_frames in self._frames.items():
            for interval, df in interval_frames.items():
                timestamps = df["timestamp"].to_numpy(dtype=np.int64)
                idx = np.searchsorted(timestamps, timestamp_ms, side="right") - 1
                if 0 <= idx < len(timestamps):
                    self._indices[symbol][interval] = int(idx)
                else:
                    self._indices[symbol][interval] = None

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List[float]]:
        if symbol not in self._frames or interval not in self._frames[symbol]:
            return []
        idx = self._indices[symbol][interval]
        if idx is None:
            return []
        df = self._frames[symbol][interval]
        start_idx = max(0, idx - max(0, limit - 1))
        subset = df.iloc[start_idx : idx + 1]
        return subset[KLINE_COLUMNS].values.tolist()

    def futures_open_interest_hist(self, symbol: str, period: str, limit: int = 30) -> List[Dict[str, float]]:
        return []

    def futures_funding_rate(self, symbol: str, limit: int = 30) -> List[Dict[str, float]]:
        return []

    @property
    def current_timestamp_ms(self) -> Optional[int]:
        return self._current_timestamp_ms

    @property
    def current_datetime(self) -> Optional[datetime]:
        if self._current_timestamp_ms is None:
            return None
        return datetime.fromtimestamp(self._current_timestamp_ms / 1000, tz=timezone.utc)


def compute_max_drawdown(equity_values: Iterable[float]) -> Optional[float]:
    values = np.array([v for v in equity_values if np.isfinite(v)], dtype=float)
    if values.size < 2:
        return None
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / peaks
    return float(drawdowns.max()) if drawdowns.size else None


def summarize_trades(trades_path: Path) -> Dict[str, Optional[float]]:
    empty_stats = {
        "total_trades": 0,
        "closed_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": None,
        "net_realized_pnl": 0.0,
    }

    if not trades_path.exists():
        return dict(empty_stats)

    try:
        df = pd.read_csv(trades_path)
    except Exception as exc:  # pragma: no cover - defensive against bad CSVs
        logging.warning("Unable to load trade history from %s: %s", trades_path, exc)
        return dict(empty_stats)

    if df.empty or "action" not in df:
        return dict(empty_stats)

    actions = df["action"].astype(str).str.upper().str.strip()
    entries_mask = actions == "ENTRY"
    closes_mask = actions == "CLOSE"

    total_trades = int(entries_mask.sum())

    close_trades = df.loc[closes_mask].copy()
    if close_trades.empty:
        return {
            **empty_stats,
            "total_trades": total_trades,
        }

    close_trades["pnl"] = pd.to_numeric(close_trades["pnl"], errors="coerce")
    close_trades = close_trades[np.isfinite(close_trades["pnl"])]

    closed = int(len(close_trades))
    winning = int((close_trades["pnl"] > 0).sum())
    losing = int((close_trades["pnl"] < 0).sum())
    win_rate = (winning / closed) * 100 if closed else None
    net_realized = float(close_trades["pnl"].sum()) if closed else 0.0

    return {
        "total_trades": total_trades,
        "closed_trades": closed,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate_pct": float(win_rate) if win_rate is not None else None,
        "net_realized_pnl": net_realized,
    }


def configure_environment(cfg: BacktestConfig) -> None:
    os.environ["TRADEBOT_DATA_DIR"] = str(cfg.run_dir)
    os.environ["HYPERLIQUID_LIVE_TRADING"] = "false"
    if cfg.start_capital is not None:
        os.environ["PAPER_START_CAPITAL"] = str(cfg.start_capital)
    if cfg.model:
        os.environ["TRADEBOT_LLM_MODEL"] = cfg.model
    if cfg.temperature is not None:
        os.environ["TRADEBOT_LLM_TEMPERATURE"] = str(cfg.temperature)
    if cfg.max_tokens is not None:
        os.environ["TRADEBOT_LLM_MAX_TOKENS"] = str(cfg.max_tokens)
    if cfg.thinking is not None:
        os.environ["TRADEBOT_LLM_THINKING"] = cfg.thinking
    if cfg.system_prompt_file:
        os.environ["TRADEBOT_SYSTEM_PROMPT_FILE"] = cfg.system_prompt_file
        os.environ.pop("TRADEBOT_SYSTEM_PROMPT", None)
    elif cfg.system_prompt is not None:
        os.environ["TRADEBOT_SYSTEM_PROMPT"] = cfg.system_prompt
        os.environ.pop("TRADEBOT_SYSTEM_PROMPT_FILE", None)
    if cfg.disable_telegram:
        os.environ["TELEGRAM_BOT_TOKEN"] = ""
        os.environ["TELEGRAM_CHAT_ID"] = ""


def main() -> None:
    configure_logging()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

    cfg = BacktestConfig.from_environment()
    logging.info("Backtest LLM override from env: %s", cfg.model)
    print(f"Backtest LLM override from env: {cfg.model}")
    configure_environment(cfg)

    import bot  # pylint: disable=import-error
    if hasattr(bot, "refresh_llm_configuration_from_env"):
        bot.refresh_llm_configuration_from_env()
    if hasattr(bot, "log_system_prompt_info"):
        bot.log_system_prompt_info("Backtest system prompt")
        print(f"System prompt for this backtest: {bot.describe_system_prompt_source()}")

    if getattr(bot, "INTERVAL", None) != cfg.interval:
        logging.info("Aligning bot interval with backtest interval: %s → %s", getattr(bot, "INTERVAL", None), cfg.interval)
        bot.INTERVAL = cfg.interval
        if hasattr(bot, "_INTERVAL_TO_SECONDS"):
            bot.CHECK_INTERVAL = bot._INTERVAL_TO_SECONDS[cfg.interval]  # type: ignore[attr-defined]

    logging.info("Backtest configured with LLM model: %s", bot.LLM_MODEL_NAME)
    print(f"LLM model for this backtest: {bot.LLM_MODEL_NAME}")

    if bot.hyperliquid_trader.is_live:
        logging.warning("Hyperliquid trader reports live mode; forcing paper mode for backtest.")
        bot.hyperliquid_trader._requested_live = False  # type: ignore[attr-defined]

    api_key = os.getenv("BN_API_KEY") or None
    api_secret = os.getenv("BN_SECRET") or None
    binance_client = Client(api_key, api_secret, testnet=False)

    intervals_needed = {cfg.interval, LONG_CONTEXT_INTERVAL}
    symbol_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in bot.SYMBOLS:
        symbol_frames[symbol] = {}
        for interval in intervals_needed:
            frame = ensure_cached_klines(binance_client, cfg, symbol, interval)
            symbol_frames[symbol][interval] = frame

    historical_client = HistoricalBinanceClient(symbol_frames)
    bot.client = historical_client  # type: ignore[assignment]

    primary_symbol = bot.SYMBOLS[0]
    primary_interval_frame = symbol_frames[primary_symbol][cfg.interval]
    timeline_mask = (primary_interval_frame["timestamp"] >= cfg.start_ms) & (
        primary_interval_frame["timestamp"] <= cfg.end_ms
    )
    timeline = primary_interval_frame.loc[timeline_mask, "timestamp"].astype(np.int64).tolist()
    if not timeline:
        logging.error("No data available for %s between %s and %s", cfg.interval, cfg.start, cfg.end)
        return

    time_holder = {"value": int(timeline[0])}

    def simulated_time() -> datetime:
        return datetime.fromtimestamp(time_holder["value"] / 1000, tz=timezone.utc)

    bot.set_time_provider(simulated_time)
    bot.reset_state(cfg.start_capital)
    bot.init_csv_files()
    bot.register_equity_snapshot(bot.START_CAPITAL)

    interval_seconds = int(interval_to_timedelta(cfg.interval).total_seconds())

    logging.info("LLM model used for this backtest: %s", bot.LLM_MODEL_NAME)
    print(f"LLM model used for this backtest: {bot.LLM_MODEL_NAME}")

    for idx, timestamp_ms in enumerate(timeline, start=1):
        time_holder["value"] = int(timestamp_ms)
        historical_client.set_current_timestamp(int(timestamp_ms))
        bot.iteration_counter += 1
        bot.current_iteration_messages = []

        bot.check_stop_loss_take_profit()
        prompt = bot.format_prompt_for_deepseek()
        decisions = bot.call_deepseek_api(prompt)

        if not decisions:
            logging.warning("Iteration %d: no decisions returned by LLM.", idx)
        else:
            bot.process_ai_decisions(decisions)

        total_equity = bot.calculate_total_equity()
        bot.register_equity_snapshot(total_equity)
        bot.log_portfolio_state()
        bot.save_state()

        current_dt = simulated_time()
        logging.info(
            "Processed bar %d/%d at %s | Equity: %.2f | Positions: %d",
            idx,
            len(timeline),
            current_dt.isoformat(),
            total_equity,
            len(bot.positions),
        )

    final_equity = bot.calculate_total_equity()
    total_return_pct = ((final_equity - bot.START_CAPITAL) / bot.START_CAPITAL) * 100 if bot.START_CAPITAL else 0.0
    sortino = bot.calculate_sortino_ratio(bot.equity_history, interval_seconds, bot.RISK_FREE_RATE)
    max_drawdown = compute_max_drawdown(bot.equity_history)
    trade_stats = summarize_trades(bot.TRADES_CSV)

    results = {
        "run_id": cfg.run_id,
        "run_directory": str(cfg.run_dir),
        "cache_directory": str(cfg.cache_dir),
        "timeframe": {
            "start": cfg.start.isoformat(),
            "end": cfg.end.isoformat(),
            "interval": cfg.interval,
            "bars": len(timeline),
        },
        "symbols": list(bot.SYMBOL_TO_COIN.values()),
        "capital": {
            "start": bot.START_CAPITAL,
            "final_balance": bot.balance,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": (max_drawdown * 100) if max_drawdown is not None else None,
            "sortino_ratio": sortino,
        },
        "llm": {
            "model": bot.LLM_MODEL_NAME,
            "temperature": bot.LLM_TEMPERATURE,
            "max_tokens": bot.LLM_MAX_TOKENS,
            "thinking": bot.LLM_THINKING_PARAM,
            "system_prompt": {
                "source": (
                    "file"
                    if cfg.system_prompt_file
                    else ("env" if cfg.system_prompt is not None else "default")
                ),
                "file": cfg.system_prompt_file,
                "override": bool(cfg.system_prompt_file or cfg.system_prompt),
                "preview": bot.TRADING_RULES_PROMPT[:200],
                "full": bot.TRADING_RULES_PROMPT,
            },
        },
        "trading": trade_stats,
        "generated_at": simulated_time().isoformat(),
    }

    results_path = cfg.run_dir / "backtest_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logging.info("Backtest complete. Results written to %s", results_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - interactive guard
        logging.info("Backtest interrupted by user.")
