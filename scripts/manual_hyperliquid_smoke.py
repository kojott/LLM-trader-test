#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual smoke test for Hyperliquid live trading.

This script places a tiny live trade (default ~2 USD notional on BTC) using the
environment credentials, waits briefly, and then closes the position. It is
intended for manual verification only—do NOT add it to automated test suites.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Tuple

from dotenv import load_dotenv

# Ensure project root is on sys.path so local modules resolve when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperliquid_client import HyperliquidTradingClient

DEFAULT_COIN = "BTC"
DEFAULT_NOTIONAL = Decimal("2")  # ~2 USD
DEFAULT_LEVERAGE = 1.0
DEFAULT_WAIT_SECONDS = 15
DEFAULT_SL_BPS = 200  # 2% below entry
DEFAULT_TP_BPS = 200  # 2% above entry


def _parse_decimal(value: str, *, name: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid decimal value for {name}: {value}") from exc


def _extract_price(level: Any) -> float:
    """Extract price from various Hyperliquid L2 level formats."""
    if isinstance(level, (list, tuple)):
        if len(level) == 0:
            raise RuntimeError("Empty level encountered.")
        first = level[0]
        if isinstance(first, (list, tuple)):
            return float(first[0])
        return float(first)
    if isinstance(level, dict):
        for key in ("px", "price", 0):
            if key in level:
                return float(level[key])
        raise RuntimeError(f"Unable to extract price from level dict: {level}")
    raise RuntimeError(f"Unsupported level format: {level}")


def parse_market_input(name: str) -> Tuple[str, str]:
    """
    Return (coin_name, market_label) for API use and logging respectively.

    Hyperliquid expects just the coin (e.g. "BTC") for exchange/info calls, but the
    UI labels markets as "BTC-USDC". We default to USDC pairs unless a different
    suffix is provided explicitly.
    """
    raw = name.strip().upper()
    if not raw:
        raise ValueError("Coin/market name cannot be blank.")
    if "-" in raw:
        base = raw.split("-", 1)[0]
        return base, raw
    return raw, f"{raw}-USDC"


def determine_order_params(
    trader: HyperliquidTradingClient,
    coin: str,
    notional: Decimal,
    leverage: float,
    sl_bps: int,
    tp_bps: int,
) -> Tuple[float, float, float, float]:
    info = trader.info
    if info is None:
        raise RuntimeError("Hyperliquid info client not initialized.")

    size_step = Decimal("0.000001")
    min_size = size_step

    try:
        coin_lookup = coin
        if coin_lookup not in info.coin_to_asset and coin_lookup in info.name_to_coin:
            coin_lookup = info.name_to_coin[coin_lookup]

        asset_id = info.coin_to_asset.get(coin_lookup)
        if asset_id is not None:
            sz_decimals = info.asset_to_sz_decimals.get(asset_id)
            if sz_decimals is not None:
                try:
                    size_step = Decimal("1").scaleb(-int(sz_decimals))
                except (TypeError, ValueError):
                    logging.warning("Unexpected szDecimals '%s' for %s; defaulting to %.6f", sz_decimals, coin, size_step)
            else:
                logging.warning("Size decimals missing for %s; defaulting to %.6f", coin, size_step)

            # Hyperliquid minimum size is typically 1 step for perps like BTC.
            min_size = max(size_step, Decimal("0.001"))
        else:
            logging.warning("Unable to map coin '%s' to asset id; using default sizing.", coin)
            min_size = Decimal("0.001")
    except Exception as exc:
        logging.warning("Failed to derive size metadata for %s: %s", coin, exc)
        min_size = Decimal("0.001")

    logging.debug("Sizing parameters for %s: step=%s, min_size=%s", coin, size_step, min_size)

    snapshot = info.l2_snapshot(coin)
    levels = snapshot.get("levels", [])
    bids = levels[0] if len(levels) > 0 else []
    asks = levels[1] if len(levels) > 1 else []

    if not asks and not bids:
        raise RuntimeError("Order book snapshot is empty; cannot determine price.")

    if asks:
        price = _extract_price(asks[0])
    else:
        price = _extract_price(bids[0]) * 1.001  # Use slight premium if only bid side exists

    logging.debug("Top of book for %s — bid=%s ask=%s", coin, bids[:1], asks[:1])

    price_dec = Decimal(str(price))
    raw_size = (notional / price_dec) * Decimal(str(leverage))

    quant_step = max(size_step, min_size)
    size_dec = raw_size.quantize(quant_step, rounding=ROUND_DOWN)
    if size_dec < min_size:
        logging.info("Adjusted order size up to exchange minimum (%s).", min_size)
        size_dec = min_size

    size = float(size_dec)
    if size <= 0:
        raise RuntimeError(f"Computed order size {size} is non-positive; increase notional.")

    base = Decimal("1")

    try:
        resolved_step = trader.get_price_step(coin)
        logging.debug("Price step for %s resolved to %s", coin, resolved_step)
    except Exception as exc:
        logging.warning("Unable to resolve price step for %s: %s", coin, exc)
        resolved_step = 0.01

    sl_raw = float(price_dec * (base - Decimal(sl_bps) / Decimal("10000")))
    tp_raw = float(price_dec * (base + Decimal(tp_bps) / Decimal("10000")))
    entry_raw = float(price_dec)  # use best quote directly; IOC ensures taker execution

    stop_loss_price = trader.normalize_price(coin, sl_raw, direction="floor")
    take_profit_price = trader.normalize_price(coin, tp_raw, direction="floor")
    entry_limit_price = trader.normalize_price(coin, entry_raw, direction="ceil")

    if stop_loss_price <= 0 or take_profit_price <= 0 or entry_limit_price <= 0:
        raise RuntimeError("Resolved prices contain non-positive values; check market data.")

    return (
        size,
        stop_loss_price,
        take_profit_price,
        entry_limit_price,
    )


def run_smoke_test(
    coin: str,
    notional: Decimal,
    leverage: float,
    wait_seconds: int,
    sl_bps: int,
    tp_bps: int,
) -> None:
    load_dotenv(override=True)

    wallet = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    secret = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

    coin_name, market_label = parse_market_input(coin)

    trader = HyperliquidTradingClient(
        live_mode=True,
        wallet_address=wallet,
        secret_key=secret,
    )

    if not trader.is_live:
        raise SystemExit("Hyperliquid live trading is not available. Check credentials and SDK install.")

    logging.info(
        "Running Hyperliquid smoke test on %s with ~%s USD notional at %sx leverage.",
        market_label,
        notional,
        leverage,
    )

    size, stop_loss_price, take_profit_price, entry_limit_price = determine_order_params(
        trader=trader,
        coin=coin_name,
        notional=notional,
        leverage=leverage,
        sl_bps=sl_bps,
        tp_bps=tp_bps,
    )

    logging.info(
        "Placing entry: coin=%s size=%.6f price=%.6f stop=%.6f tp=%.6f",
        market_label,
        size,
        entry_limit_price,
        stop_loss_price,
        take_profit_price,
    )

    entry_receipt = trader.place_entry_with_sl_tp(
        coin=coin_name,
        side="long",
        size=size,
        entry_price=entry_limit_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        leverage=leverage,
        liquidity="taker",
    )

    if not entry_receipt.get("success"):
        logging.error("Entry order failed: %s", entry_receipt.get("entry_result"))
        raise SystemExit(1)

    logging.debug("Entry result payload: %s", entry_receipt.get("entry_result"))
    logging.debug("Stop-loss result payload: %s", entry_receipt.get("stop_loss_result"))
    logging.debug("Take-profit result payload: %s", entry_receipt.get("take_profit_result"))

    entry_statuses = None
    entry_result_payload = entry_receipt.get("entry_result")
    if isinstance(entry_result_payload, dict):
        entry_statuses = entry_result_payload.get("statuses")
        if entry_statuses:
            logging.info("Entry statuses: %s", entry_statuses)

    logging.info(
        "Entry placed successfully. OIDs: entry=%s sl=%s tp=%s",
        entry_receipt.get("entry_oid"),
        entry_receipt.get("stop_loss_oid"),
        entry_receipt.get("take_profit_oid"),
    )

    info_client = trader.info
    if info_client is not None:
        try:
            user_state = info_client.user_state(trader.wallet_address)
            logging.debug("User state after entry: %s", user_state)
        except Exception as exc:
            logging.warning("Unable to fetch user state after entry: %s", exc)

    logging.info("Sleeping %d seconds before closing...", wait_seconds)
    time.sleep(wait_seconds)

    close_receipt = trader.close_position(
        coin=coin_name,
        side="long",
        size=size,
        fallback_price=entry_limit_price,
    )

    if not close_receipt.get("success"):
        logging.error("Close order failed: %s", close_receipt.get("close_result"))
        raise SystemExit(1)

    logging.debug("Close result payload: %s", close_receipt.get("close_result"))

    close_result_payload = close_receipt.get("close_result")
    if isinstance(close_result_payload, dict):
        close_statuses = close_result_payload.get("statuses")
        if close_statuses:
            logging.info("Close statuses: %s", close_statuses)

    logging.info("Close order submitted successfully (oid=%s).", close_receipt.get("close_oid"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually smoke-test Hyperliquid live trading with a tiny position."
    )
    parser.add_argument(
        "--coin",
        default=DEFAULT_COIN,
        help="Coin or market on Hyperliquid (e.g. BTC or BTC-USDC). When only the coin is provided, the script assumes the USDC pair (default: %(default)s)",
    )
    parser.add_argument(
        "--notional",
        type=lambda v: _parse_decimal(v, name="notional"),
        default=DEFAULT_NOTIONAL,
        help="Approximate USD notional to deploy (default: %(default)s)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=DEFAULT_LEVERAGE,
        help="Leverage multiplier to use (default: %(default)s)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait before submitting the close order (default: %(default)s)",
    )
    parser.add_argument(
        "--sl-bps",
        type=int,
        default=DEFAULT_SL_BPS,
        help="Stop-loss distance in basis points below entry (default: %(default)s)",
    )
    parser.add_argument(
        "--tp-bps",
        type=int,
        default=DEFAULT_TP_BPS,
        help="Take-profit distance in basis points above entry (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        run_smoke_test(
            coin=args.coin,
            notional=args.notional,
            leverage=args.leverage,
            wait_seconds=args.wait,
            sl_bps=args.sl_bps,
            tp_bps=args.tp_bps,
        )
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        sys.exit(1)
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        logging.exception("Smoke test failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
