"""Hyperliquid mainnet execution helper."""

from __future__ import annotations

import logging
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_UP,
    InvalidOperation,
)
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from hyperliquid.exchange import Exchange as HLExchange
    from hyperliquid.info import Info as HLInfo
    from eth_account.signers.local import LocalAccount


class HyperliquidTradingClient:
    """Submit live orders to Hyperliquid mainnet when enabled."""

    def __init__(self, live_mode: bool, wallet_address: str, secret_key: str) -> None:
        self._requested_live = live_mode
        self.wallet_address = (wallet_address or "").strip()
        self._secret_key = (secret_key or "").strip()
        self.info: Optional["HLInfo"] = None
        self.exchange: Optional["HLExchange"] = None
        self._local_account: Optional["LocalAccount"] = None
        self._initialized = False
        self._price_step_cache: Dict[str, Decimal] = {}

        if not self._requested_live:
            return

        if not self.wallet_address or not self._secret_key:
            raise ValueError(
                "Hyperliquid live trading requested but HYPERLIQUID_WALLET_ADDRESS or "
                "HYPERLIQUID_PRIVATE_KEY is missing."
            )

        try:
            from eth_account import Account
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants as hl_constants
        except ImportError as exc:
            raise RuntimeError(
                "hyperliquid-python-sdk (and its Ethereum dependencies) must be installed to enable live trading."
            ) from exc

        account: "LocalAccount" = Account.from_key(self._secret_key)
        original_address = self.wallet_address
        if account.address.lower() != original_address.lower():
            logging.warning(
                "Hyperliquid wallet address %s does not match derived account %s; using derived address.",
                original_address,
                account.address,
            )
        self.wallet_address = account.address
        self._local_account = account

        try:
            self.info = cast("HLInfo", Info(hl_constants.MAINNET_API_URL, skip_ws=True))
            self.exchange = cast(
                "HLExchange",
                Exchange(
                    wallet=account,
                    base_url=hl_constants.MAINNET_API_URL,
                    vault_address=None,
                    account_address=account.address,
                ),
            )
            self._initialized = True
            logging.info(
                "Hyperliquid live trading initialized for wallet %s",
                self._mask_wallet(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Hyperliquid client: {exc}"
            ) from exc

    @property
    def is_live(self) -> bool:
        """Return True when live trading is active and initialized."""
        return self._requested_live and self._initialized

    @property
    def masked_wallet(self) -> str:
        """Return masked wallet address for status messages."""
        return self._mask_wallet()

    def place_entry_with_sl_tp(
        self,
        coin: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        leverage: float,
        liquidity: str,
    ) -> Dict[str, Any]:
        """
        Submit an entry order and attach optional stop-loss / take-profit triggers.
        Returns a dictionary containing raw exchange responses.
        """
        response: Dict[str, Any] = {
            "success": False,
            "entry_result": None,
            "stop_loss_result": None,
            "take_profit_result": None,
            "entry_oid": None,
            "stop_loss_oid": None,
            "take_profit_oid": None,
        }

        if not self.is_live:
            return response

        is_buy = side.lower() == "long"
        exchange = self.exchange
        if exchange is None:
            raise RuntimeError("Hyperliquid exchange client not initialized.")
        exchange_any = cast(Any, exchange)
        try:
            leverage_value = int(leverage)
            exchange_any.update_leverage(leverage_value, coin, is_cross=False)
        except Exception as exc:
            logging.warning("Failed to set leverage %s for %s: %s", leverage_value, coin, exc)

        tif = "Gtc" if liquidity.lower() == "maker" else "Ioc"
        limit_px = entry_price
        if limit_px is None:
            limit_px = self._compute_market_price(coin, is_buy, entry_price)
        if limit_px and limit_px > 0:
            limit_px = self._normalize_price(coin, limit_px, rounding="ceil" if is_buy else "floor")
            if tif == "Ioc":
                step = self.get_price_step(coin)
                if step > 0:
                    adjustment = limit_px + step if is_buy else limit_px - step
                    limit_px = self._normalize_price(
                        coin,
                        adjustment,
                        rounding="ceil" if is_buy else "floor",
                    )
        order_type: Dict[str, Any] = {"limit": {"tif": tif}}

        try:
            entry_result = exchange_any.order(
                name=coin,
                is_buy=is_buy,
                sz=size,
                limit_px=limit_px,
                order_type=order_type,
                reduce_only=False,
            )
        except Exception as exc:
            logging.error("Hyperliquid entry order failed for %s: %s", coin, exc)
            response["entry_result"] = {"status": "error", "exception": str(exc)}
            return response

        response["entry_result"] = entry_result
        entry_statuses = self._extract_statuses(entry_result)
        entry_errors = [status.get("error") for status in entry_statuses if isinstance(status, dict) and status.get("error")]
        entry_filled = any("filled" in status for status in entry_statuses)
        entry_resting = any("resting" in status for status in entry_statuses)
        entry_success = bool(entry_result.get("status") == "ok" and not entry_errors)
        if entry_success and entry_statuses:
            entry_success = entry_filled or entry_resting
        response["success"] = entry_success
        response["entry_oid"] = self._find_first_oid(entry_result)

        if entry_errors:
            logging.error("Hyperliquid entry order errors for %s: %s", coin, entry_errors)

        if not entry_success:
            logging.error("Hyperliquid entry order rejected for %s: %s", coin, entry_result)
            return response
        if entry_filled:
            if stop_loss_price and stop_loss_price > 0:
                sl_result = self._place_trigger_order(
                    coin=coin,
                    is_buy=not is_buy,
                    size=size,
                    trigger_price=stop_loss_price,
                    tpsl="sl",
                )
                response["stop_loss_result"] = sl_result
                response["stop_loss_oid"] = self._find_first_oid(sl_result)

            if take_profit_price and take_profit_price > 0:
                tp_result = self._place_trigger_order(
                    coin=coin,
                    is_buy=not is_buy,
                    size=size,
                    trigger_price=take_profit_price,
                    tpsl="tp",
                )
                response["take_profit_result"] = tp_result
                response["take_profit_oid"] = self._find_first_oid(tp_result)
        else:
            logging.info("Entry did not fill immediately; skipping SL/TP placement.")

        return response

    def close_position(
        self,
        coin: str,
        side: str,
        size: Optional[float] = None,
        fallback_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close an open position using a reduce-only IOC order.

        Args:
            coin: Market symbol (e.g. "BTC").
            side: Current position side ("long"/"short").
            size: Quantity to close (defaults to full position).
            fallback_price: Price to use if order book snapshot is unavailable.
        """
        result: Dict[str, Any] = {
            "success": False,
            "close_result": None,
            "close_oid": None,
        }

        exchange = self.exchange
        if not self.is_live or exchange is None:
            return result
        exchange_any = cast(Any, exchange)

        position_size, live_side = self._lookup_live_position(coin)
        close_size = size or abs(position_size)

        if close_size <= 0:
            logging.info("No live Hyperliquid position detected for %s; nothing to close.", coin)
            result["success"] = True
            return result

        if position_size != 0:
            is_buy = position_size < 0
        else:
            is_buy = side.lower() == "short" if live_side is None else live_side == "short"

        limit_px = self._compute_market_price(coin, is_buy, fallback_price)

        try:
            close_resp = exchange_any.order(
                name=coin,
                is_buy=is_buy,
                sz=close_size,
                limit_px=limit_px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,
            )
        except Exception as exc:
            logging.error("Hyperliquid close order failed for %s: %s", coin, exc)
            result["close_result"] = {"status": "error", "exception": str(exc)}
            return result

        result["close_result"] = close_resp
        result["close_oid"] = self._find_first_oid(close_resp)
        result["success"] = close_resp.get("status") == "ok"

        if not result["success"]:
            logging.error("Hyperliquid close order rejected for %s: %s", coin, close_resp)

        return result

    def _compute_market_price(
        self,
        coin: str,
        is_buy: bool,
        fallback_price: Optional[float],
    ) -> float:
        """Return a price likely to fill immediately when using IOC orders."""
        price = fallback_price or 0.0
        info = self.info
        if not self.is_live or info is None:
            return price
        info_any = cast(Any, info)

        try:
            l2_data = info_any.l2_snapshot(coin)
            bids = l2_data.get("levels", [[], []])[0] or []
            asks = l2_data.get("levels", [[], []])[1] or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None

            if is_buy:
                if best_ask is not None:
                    price = best_ask
                elif best_bid is not None:
                    price = best_bid * 1.001
            else:
                if best_bid is not None:
                    price = best_bid
                elif best_ask is not None:
                    price = best_ask * 0.999
        except Exception as exc:
            logging.warning("Failed to pull L2 snapshot for %s: %s", coin, exc)

        if not price or price <= 0:
            if fallback_price and fallback_price > 0:
                adjustment = 1.001 if is_buy else 0.999
                price = fallback_price * adjustment
            else:
                return 0.0
        step = Decimal(str(self.get_price_step(coin)))
        try:
            price_dec = Decimal(str(price))
        except (InvalidOperation, ValueError):
            return self._normalize_price(coin, price, rounding="ceil" if is_buy else "floor")

        if step > 0:
            if is_buy:
                price_dec += step
            else:
                candidate = price_dec - step
                if candidate > 0:
                    price_dec = candidate

        return self._normalize_price(coin, float(price_dec), rounding="ceil" if is_buy else "floor")

    def get_price_step(self, coin: str) -> float:
        """Return the resolved price step for the given coin as a float."""
        return float(self._price_step_decimal(coin))

    def normalize_price(self, coin: str, price: float, direction: str = "floor") -> float:
        """Public helper to snap a price to the exchange tick size."""
        return self._normalize_price(coin, price, rounding=direction)

    def _normalize_price(
        self,
        coin: str,
        price: Optional[float],
        *,
        rounding: str = "floor",
    ) -> float:
        if price is None:
            return 0.0
        if price <= 0:
            return float(price)
        try:
            price_dec = Decimal(str(price))
        except (InvalidOperation, ValueError):
            return float(price)

        step = self._price_step_decimal(coin)
        if step <= 0:
            return float(price_dec)

        try:
            quotient = price_dec / step
            if rounding == "ceil":
                units = quotient.to_integral_value(rounding=ROUND_CEILING)
            elif rounding == "nearest":
                units = quotient.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            else:
                units = quotient.to_integral_value(rounding=ROUND_FLOOR)
        except (InvalidOperation, ArithmeticError):
            return float(price_dec)

        normalized = units * step
        if normalized <= 0:
            return float(price_dec)
        return float(normalized.normalize())

    def _price_step_decimal(self, coin: str) -> Decimal:
        key = self._canonical_coin_key(coin)
        cached = self._price_step_cache.get(key)
        if cached is not None:
            return cached

        step = self._resolve_price_step(key)
        if step is None or step <= 0:
            step = Decimal("0.01")

        self._price_step_cache[key] = step
        original_key = (coin or "").upper()
        if original_key != key:
            self._price_step_cache[original_key] = step
        return step

    def _resolve_price_step(self, coin: str) -> Optional[Decimal]:
        info = self.info
        if info is None:
            return None
        info_any = cast(Any, info)
        coin_lookup = coin
        try:
            if coin_lookup not in info_any.coin_to_asset and coin_lookup in info_any.name_to_coin:
                coin_lookup = info_any.name_to_coin[coin_lookup]
        except Exception:
            pass

        meta_step = self._price_step_from_meta(info_any, coin_lookup)
        if meta_step:
            return meta_step

        book_step = self._price_step_from_l2(info_any, coin_lookup)
        if book_step:
            return book_step

        return None

    def _price_step_from_meta(self, info_any: Any, coin: str) -> Optional[Decimal]:
        try:
            meta = info_any.meta()
        except Exception as exc:
            logging.debug("Failed to fetch Hyperliquid meta for %s: %s", coin, exc)
            return None

        target = (coin or "").upper()
        for asset_info in meta.get("universe", []):
            name = str(asset_info.get("name", "")).upper()
            if name != target:
                continue

            tick = asset_info.get("priceTick") or asset_info.get("tickSize")
            if tick:
                try:
                    return Decimal(str(tick)).normalize()
                except (InvalidOperation, ValueError):
                    logging.debug("Invalid tick size '%s' for %s", tick, coin)

            px_decimals = asset_info.get("pxDecimals")
            if px_decimals is not None:
                try:
                    return Decimal("1").scaleb(-int(px_decimals)).normalize()
                except (InvalidOperation, TypeError, ValueError):
                    logging.debug("Unexpected pxDecimals '%s' for %s", px_decimals, coin)
            break

        return None

    def _price_step_from_l2(self, info_any: Any, coin: str) -> Optional[Decimal]:
        try:
            snapshot = info_any.l2_snapshot(coin)
        except Exception as exc:
            logging.debug("Failed to fetch L2 snapshot for %s while resolving tick: %s", coin, exc)
            return None

        levels = snapshot.get("levels", [])
        prices: List[Decimal] = []
        for side in levels or []:
            if not side:
                continue
            for level in side:
                try:
                    px = Decimal(str(self._extract_price_from_level(level)))
                    prices.append(px)
                except (TypeError, ValueError, InvalidOperation):
                    continue

        if len(prices) < 2:
            return None

        unique_prices = sorted(set(prices))
        candidates: List[Decimal] = []
        for idx in range(1, len(unique_prices)):
            diff = unique_prices[idx] - unique_prices[idx - 1]
            if diff > 0:
                candidates.append(diff.normalize())

        if not candidates:
            return None

        step = min(candidates)
        if step <= 0:
            return None
        return step

    @staticmethod
    def _extract_price_from_level(level: Any) -> float:
        if isinstance(level, (list, tuple)):
            if not level:
                raise ValueError("Empty level encountered.")
            first = level[0]
            if isinstance(first, (list, tuple)):
                if not first:
                    raise ValueError("Empty nested level encountered.")
                first = first[0]
            return float(first)
        if isinstance(level, dict):
            for key in ("px", "price", 0):
                if key in level:
                    return float(level[key])
        raise ValueError(f"Unsupported order book level format: {level!r}")

    def _canonical_coin_key(self, coin: str) -> str:
        normalized = (coin or "").upper()
        info = self.info
        if info is None:
            return normalized
        info_any = cast(Any, info)
        try:
            if normalized not in info_any.coin_to_asset and normalized in info_any.name_to_coin:
                normalized = str(info_any.name_to_coin[normalized]).upper()
        except Exception:
            pass
        return normalized

    def _place_trigger_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        trigger_price: float,
        tpsl: str,
    ) -> Dict[str, Any]:
        """Submit a trigger order (stop-loss or take-profit)."""
        exchange = self.exchange
        if exchange is None:
            raise RuntimeError("Hyperliquid exchange client not initialized.")
        exchange_any = cast(Any, exchange)
        rounding = "ceil" if is_buy else "floor"
        normalized_trigger = self._normalize_price(coin, trigger_price, rounding=rounding)
        if normalized_trigger <= 0:
            logging.error(
                "Unable to normalize trigger price %.8f for %s %s order.",
                trigger_price,
                coin,
                tpsl,
            )
            return {"status": "error", "message": "Invalid trigger price after normalization."}

        try:
            response = exchange_any.order(
                name=coin,
                is_buy=is_buy,
                sz=size,
                limit_px=normalized_trigger,
                order_type={
                    "trigger": {
                        "isMarket": True,
                        "triggerPx": normalized_trigger,
                        "tpsl": tpsl,
                    }
                },
                reduce_only=True,
            )
            if response.get("status") != "ok":
                logging.error(
                    "Hyperliquid %s order rejected for %s: %s", tpsl, coin, response
                )
            return response
        except Exception as exc:
            logging.error(
                "Failed to submit Hyperliquid %s order for %s: %s",
                tpsl,
                coin,
                exc,
            )
            return {"status": "error", "exception": str(exc)}

    def _lookup_live_position(self, coin: str) -> Tuple[float, Optional[str]]:
        """Return (size, side) tuple for the current live position."""
        info = self.info
        if not self.is_live or info is None:
            return 0.0, None
        try:
            info_any = cast(Any, info)
            user_state = info_any.user_state(self.wallet_address)
        except Exception as exc:
            logging.error("Failed to fetch Hyperliquid user state: %s", exc)
            return 0.0, None

        for asset_pos in user_state.get("assetPositions", []):
            position = asset_pos.get("position", {})
            if position.get("coin") != coin:
                continue
            try:
                size = float(position.get("szi", 0.0))
            except (TypeError, ValueError):
                size = 0.0
            side = "long" if size > 0 else "short" if size < 0 else None
            return size, side

        return 0.0, None

    @staticmethod
    def _find_first_oid(payload: Any) -> Optional[Any]:
        """Recursively search for the first 'oid' field in a nested payload."""
        if isinstance(payload, dict):
            if "oid" in payload:
                return payload["oid"]
            for value in payload.values():
                found = HyperliquidTradingClient._find_first_oid(value)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = HyperliquidTradingClient._find_first_oid(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _extract_statuses(payload: Any) -> List[Dict[str, Any]]:
        """Return list of status dicts extracted from exchange response."""
        if not isinstance(payload, dict):
            return []
        response = payload.get("response")
        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, dict):
                statuses = data.get("statuses")
                if isinstance(statuses, list):
                    return [cast(Dict[str, Any], status) for status in statuses if isinstance(status, dict)]
        statuses = payload.get("statuses")
        if isinstance(statuses, list):
            return [cast(Dict[str, Any], status) for status in statuses if isinstance(status, dict)]
        return []

    def _mask_wallet(self) -> str:
        """Return a partially masked wallet address for logs."""
        if not self.wallet_address or len(self.wallet_address) < 10:
            return self.wallet_address
        return f"{self.wallet_address[:6]}…{self.wallet_address[-4:]}"
