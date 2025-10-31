#!/usr/bin/env python3
"""
Recalculate portfolio state from trade history.

Use this when manual edits were made to trade_history.csv
and the persisted state/json needs to be reconciled.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

FEE_PATTERN = re.compile(r"Fees:\s*\$(-?\d+(?:\.\d+)?)")


def _parse_bool_env(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float_env(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def detect_starting_capital() -> float:
    hyperliquid_live = _parse_bool_env(
        os.getenv("HYPERLIQUID_LIVE_TRADING"),
        default=False,
    )
    paper_start = _parse_float_env(os.getenv("PAPER_START_CAPITAL"), 10_000.0)
    hyper_start = _parse_float_env(os.getenv("HYPERLIQUID_CAPITAL"), 500.0)
    return hyper_start if hyperliquid_live else paper_start


def extract_fee(reason: str) -> float:
    if not reason:
        return 0.0
    match = FEE_PATTERN.search(reason)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return 0.0


def clean_reason_text(reason: str) -> str:
    if not reason:
        return ""
    parts = reason.split("|")
    if not parts:
        return reason.strip()
    return parts[0].strip()


def resolve_data_dir(base_dir: Path) -> Path:
    default = base_dir / "data"
    override = os.getenv("TRADEBOT_DATA_DIR")
    if not override:
        return default
    path = Path(override).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass
class Position:
    coin: str
    side: str
    quantity: float
    entry_price: float
    profit_target: float
    stop_loss: float
    leverage: float
    confidence: float
    entry_reason: str
    entry_timestamp: str
    # Derived
    margin: float
    entry_fee: float
    fee_rate: float
    risk_usd: float
    extra: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_trade(cls, row: Dict[str, str]) -> "Position":
        quantity = float(row.get("quantity") or 0.0)
        price = float(row.get("price") or 0.0)
        leverage = float(row.get("leverage") or 1.0) or 1.0
        profit_target = float(row.get("profit_target") or 0.0)
        stop_loss = float(row.get("stop_loss") or 0.0)
        entry_fee = extract_fee(row.get("reason", ""))
        position_value = quantity * price
        margin = position_value / leverage if leverage else position_value
        fee_base = position_value if position_value else 1.0
        fee_rate = entry_fee / fee_base if entry_fee else 0.0
        risk_usd = abs(price - stop_loss) * quantity
        return cls(
            coin=row["coin"],
            side=(row.get("side") or "long").lower(),
            quantity=quantity,
            entry_price=price,
            profit_target=profit_target,
            stop_loss=stop_loss,
            leverage=leverage,
            confidence=float(row.get("confidence") or 0.0),
            entry_reason=clean_reason_text(row.get("reason", "")),
            entry_timestamp=row.get("timestamp", ""),
            margin=margin,
            entry_fee=entry_fee,
            fee_rate=fee_rate,
            risk_usd=risk_usd,
        )

    def to_state_dict(self) -> Dict[str, object]:
        return {
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "leverage": self.leverage,
            "confidence": self.confidence,
            "invalidation_condition": "",
            "margin": self.margin,
            "fees_paid": self.entry_fee,
            "fee_rate": self.fee_rate,
            "liquidity": "taker",
            "entry_justification": self.entry_reason,
            "last_justification": self.entry_reason,
            "risk_usd": self.risk_usd,
            "wait_for_fill": False,
            "entry_oid": -1,
            "tp_oid": -1,
            "sl_oid": -1,
        }


def process_trades(trade_rows: List[Dict[str, str]], starting_balance: float) -> Dict[str, object]:
    balance = starting_balance
    open_positions: Dict[str, Position] = {}
    warnings: List[str] = []

    for row in trade_rows:
        action = (row.get("action") or "").strip().upper()
        coin = row.get("coin") or ""
        if not action or not coin:
            warnings.append(f"Skipping malformed row: {row}")
            continue

        if action == "ENTRY":
            if coin in open_positions:
                warnings.append(f"Duplicate entry for {coin} at {row.get('timestamp')}; replacing previous open position.")
            position = Position.from_trade(row)
            balance -= position.margin
            balance -= position.entry_fee
            open_positions[coin] = position
        elif action == "CLOSE":
            position = open_positions.pop(coin, None)
            if position is None:
                warnings.append(f"CLOSE for {coin} at {row.get('timestamp')} without matching ENTRY; ignored.")
                continue
            try:
                exit_price = float(row.get("price") or 0.0)
            except (TypeError, ValueError):
                exit_price = 0.0
            if exit_price <= 0:
                warnings.append(
                    f"{coin} close @ {row.get('timestamp')} has non-positive exit price ({exit_price}); "
                    "check trade history."
                )
            total_fees = extract_fee(row.get("reason", ""))
            if total_fees <= 0 and position.entry_fee > 0:
                # Fallback: assume exit fee is zero, total fees equals entry fee
                total_fees = position.entry_fee
            if position.side == "long":
                gross = (exit_price - position.entry_price) * position.quantity
            else:
                gross = (position.entry_price - exit_price) * position.quantity
            net = gross - total_fees
            balance += position.margin
            balance += net
        else:
            warnings.append(f"Unknown action '{action}' for {coin} at {row.get('timestamp')}")

    total_margin = sum(pos.margin for pos in open_positions.values())
    state = {
        "balance": balance,
        "positions": {coin: pos.to_state_dict() for coin, pos in open_positions.items()},
        "total_margin": total_margin,
        "warnings": warnings,
    }
    return state


def load_trades(trades_path: Path) -> List[Dict[str, str]]:
    with trades_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    rows.sort(key=lambda r: r.get("timestamp") or "")
    return rows


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = resolve_data_dir(base_dir)

    parser = argparse.ArgumentParser(description="Recalculate portfolio state from trade history.")
    parser.add_argument("--trades", type=Path, default=data_dir / "trade_history.csv", help="Path to trade_history.csv")
    parser.add_argument("--state-json", type=Path, default=data_dir / "portfolio_state.json", help="Path to write portfolio_state.json")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just display results.")
    parser.add_argument("--start-capital", type=float, default=None, help="Override starting capital.")
    args = parser.parse_args()

    if not args.trades.exists():
        raise FileNotFoundError(f"Trade history not found at {args.trades}")

    starting_capital = args.start_capital if args.start_capital is not None else detect_starting_capital()
    trades = load_trades(args.trades)
    result = process_trades(trades, starting_capital)

    print("=== Portfolio Reconstruction ===")
    print(f"Trades processed : {len(trades)}")
    print(f"Starting balance : ${starting_capital:,.2f}")
    print(f"Available balance: ${result['balance']:,.2f}")
    print(f"Open positions  : {len(result['positions'])}")
    for coin, pos in result["positions"].items():
        print(
            f"  - {coin} {pos['side']} size {pos['quantity']:.6f} "
            f"entry ${pos['entry_price']:.2f} margin ${pos['margin']:.2f}"
        )
    if result["warnings"]:
        print("\nWarnings:")
        for message in result["warnings"]:
            print(f"  * {message}")

    now_iso = datetime.now(timezone.utc).isoformat()
    iteration = 0
    if args.state_json.exists():
        try:
            with args.state_json.open("r") as fh:
                existing = json.load(fh)
            iteration = int(existing.get("iteration", 0))
        except Exception:
            iteration = 0

    state_payload = {
        "balance": result["balance"],
        "positions": result["positions"],
        "iteration": iteration,
        "updated_at": now_iso,
    }

    if args.dry_run:
        print("\n-- Dry run: state file not updated --")
        print(json.dumps(state_payload, indent=2))
        return

    args.state_json.parent.mkdir(parents=True, exist_ok=True)
    with args.state_json.open("w") as fh:
        json.dump(state_payload, fh, indent=2)
        fh.write("\n")
    print(f"\nState written to {args.state_json}")


if __name__ == "__main__":
    main()
