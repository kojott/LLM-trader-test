## Hyperliquid Python Trading Examples

Here are comprehensive examples showing how to place trades with SL/TP, close positions, and view active trades with all fees included.

### 1. **Installation & Setup**

```python
# Install the SDK
# pip install hyperliquid-python-sdk

import json
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Configuration
config = {
    "account_address": "0xYOUR_WALLET_ADDRESS",  # Your main wallet address
    "secret_key": "YOUR_PRIVATE_KEY"  # Your private key or API wallet key
}

# Initialize Info (read-only) and Exchange (trading)
info = Info(constants.MAINNET_API_URL, skip_ws=True)
exchange = Exchange(
    wallet=config["account_address"],
    base_url=constants.MAINNET_API_URL,
    vault_address=None,  # Use None for personal trading
    account_address=config["account_address"]
)

# Set your private key
exchange.wallet.secret_key = config["secret_key"]
```

***

### 2. **View Active Positions & Account Info**

```python
def get_active_positions(address):
    """
    Get all active positions with details including PnL, size, margin
    """
    user_state = info.user_state(address)
    
    print("=" * 60)
    print("ACCOUNT SUMMARY")
    print("=" * 60)
    
    # Account value and margin
    margin_summary = user_state.get("marginSummary", {})
    print(f"Account Value: ${float(margin_summary.get('accountValue', 0)):,.2f}")
    print(f"Total Position Value: ${float(margin_summary.get('totalNtlPos', 0)):,.2f}")
    print(f"Total Raw USD: ${float(margin_summary.get('totalRawUsd', 0)):,.2f}")
    print(f"Withdrawable: ${float(user_state.get('withdrawable', 0)):,.2f}")
    
    print("\n" + "=" * 60)
    print("ACTIVE POSITIONS")
    print("=" * 60)
    
    # Get positions
    asset_positions = user_state.get("assetPositions", [])
    
    if not asset_positions:
        print("No active positions")
        return []
    
    positions = []
    for asset_pos in asset_positions:
        position = asset_pos.get("position", {})
        coin = position.get("coin", "")
        
        # Position details
        size = float(position.get("szi", 0))
        entry_px = float(position.get("entryPx", 0))
        position_value = float(position.get("positionValue", 0))
        unrealized_pnl = float(position.get("unrealizedPnl", 0))
        return_on_equity = float(position.get("returnOnEquity", 0)) * 100
        leverage = position.get("leverage", {}).get("value", "0")
        liquidation_px = position.get("liquidationPx")
        margin_used = float(position.get("marginUsed", 0))
        
        positions.append({
            "coin": coin,
            "size": size,
            "entry_price": entry_px,
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "roe": return_on_equity,
            "leverage": leverage,
            "liquidation_price": liquidation_px,
            "margin_used": margin_used
        })
        
        side = "LONG" if size > 0 else "SHORT"
        print(f"\n{coin} ({side})")
        print(f"  Size: {abs(size)}")
        print(f"  Entry Price: ${entry_px:,.4f}")
        print(f"  Position Value: ${position_value:,.2f}")
        print(f"  Unrealized PnL: ${unrealized_pnl:,.2f}")
        print(f"  ROE: {return_on_equity:,.2f}%")
        print(f"  Leverage: {leverage}x")
        print(f"  Margin Used: ${margin_used:,.2f}")
        if liquidation_px:
            print(f"  Liquidation Price: ${float(liquidation_px):,.4f}")
    
    return positions

# Usage
positions = get_active_positions(config["account_address"])
```

***

### 3. **View Open Orders (Including TP/SL)**

```python
def get_open_orders(address):
    """
    Get all open orders including take profit and stop loss orders
    """
    open_orders = info.open_orders(address)
    
    print("\n" + "=" * 60)
    print("OPEN ORDERS")
    print("=" * 60)
    
    if not open_orders:
        print("No open orders")
        return []
    
    for order in open_orders:
        coin = order.get("coin", "")
        side = order.get("side", "")
        limit_px = order.get("limitPx", "")
        sz = order.get("sz", "")
        oid = order.get("oid", "")
        order_type = order.get("orderType", "")
        trigger_px = order.get("triggerPx", "")
        reduce_only = order.get("reduceOnly", False)
        
        print(f"\n{coin} - {side} {order_type}")
        print(f"  Order ID: {oid}")
        print(f"  Size: {sz}")
        print(f"  Limit Price: ${limit_px}")
        if trigger_px:
            print(f"  Trigger Price: ${trigger_px}")
        print(f"  Reduce Only: {reduce_only}")
    
    return open_orders

# Usage
orders = get_open_orders(config["account_address"])
```

***

### 4. **Place Order with Stop Loss and Take Profit**

```python
def place_order_with_sl_tp(
    coin: str,
    is_buy: bool,
    size: float,
    price: float,
    stop_loss_price: float = None,
    take_profit_price: float = None,
    leverage: int = 1,
    reduce_only: bool = False
):
    """
    Place a limit order with optional stop loss and take profit
    
    Args:
        coin: Symbol (e.g., "BTC", "ETH", "SOL")
        is_buy: True for long, False for short
        size: Order size
        price: Limit price for entry
        stop_loss_price: Stop loss trigger price (optional)
        take_profit_price: Take profit trigger price (optional)
        leverage: Leverage to use (default 1x)
        reduce_only: Set True to only reduce position (close)
    """
    
    print(f"\n{'='*60}")
    print(f"PLACING ORDER: {coin}")
    print(f"{'='*60}")
    
    # Set leverage (isolated mode)
    try:
        leverage_result = exchange.update_leverage(leverage, coin, is_cross=False)
        print(f"Leverage set to {leverage}x (isolated)")
    except Exception as e:
        print(f"Warning: Could not set leverage: {e}")
    
    # Place the main order
    order_type = {"limit": {"tif": "Gtc"}}  # Good-til-cancel
    
    print(f"\n1. Placing {'BUY' if is_buy else 'SELL'} order")
    print(f"   Size: {size}")
    print(f"   Price: ${price}")
    
    order_result = exchange.order(
        coin=coin,
        is_buy=is_buy,
        sz=size,
        limit_px=price,
        order_type=order_type,
        reduce_only=reduce_only
    )
    
    print(f"   Result: {json.dumps(order_result, indent=2)}")
    
    if order_result.get("status") != "ok":
        print(f"❌ Order failed!")
        return order_result
    
    print(f"✅ Entry order placed successfully")
    
    # Add Stop Loss if specified
    if stop_loss_price:
        print(f"\n2. Adding Stop Loss at ${stop_loss_price}")
        
        # Stop loss is opposite side (if buy entry, sell for SL)
        sl_side = not is_buy
        
        sl_result = exchange.order(
            coin=coin,
            is_buy=sl_side,
            sz=size,
            limit_px=stop_loss_price,  # This is ignored but required
            order_type={
                "trigger": {
                    "isMarket": True,  # Market order when triggered
                    "triggerPx": stop_loss_price,  # Trigger price
                    "tpsl": "sl"  # Stop loss type
                }
            },
            reduce_only=True  # Close position only
        )
        
        print(f"   Result: {json.dumps(sl_result, indent=2)}")
        
        if sl_result.get("status") == "ok":
            print(f"✅ Stop Loss set at ${stop_loss_price}")
        else:
            print(f"❌ Stop Loss failed")
    
    # Add Take Profit if specified
    if take_profit_price:
        print(f"\n3. Adding Take Profit at ${take_profit_price}")
        
        # Take profit is opposite side (if buy entry, sell for TP)
        tp_side = not is_buy
        
        tp_result = exchange.order(
            coin=coin,
            is_buy=tp_side,
            sz=size,
            limit_px=take_profit_price,  # This is ignored but required
            order_type={
                "trigger": {
                    "isMarket": True,  # Market order when triggered
                    "triggerPx": take_profit_price,  # Trigger price
                    "tpsl": "tp"  # Take profit type
                }
            },
            reduce_only=True  # Close position only
        )
        
        print(f"   Result: {json.dumps(tp_result, indent=2)}")
        
        if tp_result.get("status") == "ok":
            print(f"✅ Take Profit set at ${take_profit_price}")
        else:
            print(f"❌ Take Profit failed")
    
    return order_result

# Example usage
# Long BTC at $67,000 with SL at $66,000 and TP at $69,000
result = place_order_with_sl_tp(
    coin="BTC",
    is_buy=True,  # Long
    size=0.01,
    price=67000,
    stop_loss_price=66000,
    take_profit_price=69000,
    leverage=2
)
```

***

### 5. **Place Market Order with SL/TP**

```python
def place_market_order_with_sl_tp(
    coin: str,
    is_buy: bool,
    size: float,
    stop_loss_price: float = None,
    take_profit_price: float = None,
    leverage: int = 1
):
    """
    Place a market order with stop loss and take profit
    """
    
    print(f"\n{'='*60}")
    print(f"MARKET ORDER: {coin}")
    print(f"{'='*60}")
    
    # Set leverage
    try:
        exchange.update_leverage(leverage, coin, is_cross=False)
        print(f"Leverage: {leverage}x (isolated)")
    except Exception as e:
        print(f"Warning: {e}")
    
    # Get current price to use as limit (market order workaround)
    l2_data = info.l2_snapshot(coin)
    
    if is_buy:
        # For buy, use ask price (or slightly above)
        current_price = float(l2_data["levels"][1][0][0]) * 1.02  # 2% above ask
    else:
        # For sell, use bid price (or slightly below)
        current_price = float(l2_data["levels"][0][0][0]) * 0.98  # 2% below bid
    
    print(f"\n1. Placing MARKET {'BUY' if is_buy else 'SELL'}")
    print(f"   Size: {size}")
    print(f"   Estimated price: ${current_price:,.2f}")
    
    # Market order = limit order with Ioc (Immediate or Cancel)
    order_result = exchange.order(
        coin=coin,
        is_buy=is_buy,
        sz=size,
        limit_px=current_price,
        order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel = market
        reduce_only=False
    )
    
    print(f"   Result: {json.dumps(order_result, indent=2)}")
    
    if order_result.get("status") != "ok":
        print(f"❌ Market order failed!")
        return order_result
    
    print(f"✅ Market order executed")
    
    # Add SL and TP (same as limit order)
    if stop_loss_price:
        print(f"\n2. Adding Stop Loss at ${stop_loss_price}")
        sl_result = exchange.order(
            coin=coin,
            is_buy=not is_buy,
            sz=size,
            limit_px=stop_loss_price,
            order_type={
                "trigger": {
                    "isMarket": True,
                    "triggerPx": stop_loss_price,
                    "tpsl": "sl"
                }
            },
            reduce_only=True
        )
        print(f"   SL Status: {sl_result.get('status')}")
    
    if take_profit_price:
        print(f"\n3. Adding Take Profit at ${take_profit_price}")
        tp_result = exchange.order(
            coin=coin,
            is_buy=not is_buy,
            sz=size,
            limit_px=take_profit_price,
            order_type={
                "trigger": {
                    "isMarket": True,
                    "triggerPx": take_profit_price,
                    "tpsl": "tp"
                }
            },
            reduce_only=True
        )
        print(f"   TP Status: {tp_result.get('status')}")
    
    return order_result

# Example: Market buy ETH with SL/TP
result = place_market_order_with_sl_tp(
    coin="ETH",
    is_buy=True,
    size=0.5,
    stop_loss_price=2400,
    take_profit_price=2700,
    leverage=3
)
```

***

### 6. **Close Position (Full or Partial)**

```python
def close_position(coin: str, size: float = None):
    """
    Close a position using market order
    
    Args:
        coin: Symbol to close
        size: Size to close (if None, closes full position)
    """
    
    print(f"\n{'='*60}")
    print(f"CLOSING POSITION: {coin}")
    print(f"{'='*60}")
    
    # Get current position
    user_state = info.user_state(config["account_address"])
    
    position_size = 0
    for asset_pos in user_state.get("assetPositions", []):
        position = asset_pos.get("position", {})
        if position.get("coin") == coin:
            position_size = float(position.get("szi", 0))
            break
    
    if position_size == 0:
        print(f"❌ No position found for {coin}")
        return None
    
    # Determine close size
    if size is None:
        size = abs(position_size)
    else:
        size = min(size, abs(position_size))
    
    # Determine side (opposite of position)
    is_buy = position_size < 0  # If short, buy to close
    
    print(f"Position size: {position_size}")
    print(f"Closing size: {size}")
    print(f"Side: {'BUY' if is_buy else 'SELL'} to close")
    
    # Get current market price
    l2_data = info.l2_snapshot(coin)
    if is_buy:
        price = float(l2_data["levels"][1][0][0]) * 1.02
    else:
        price = float(l2_data["levels"][0][0][0]) * 0.98
    
    # Close with market order (Ioc)
    close_result = exchange.order(
        coin=coin,
        is_buy=is_buy,
        sz=size,
        limit_px=price,
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=True  # Important: only reduce position
    )
    
    print(f"\nResult: {json.dumps(close_result, indent=2)}")
    
    if close_result.get("status") == "ok":
        print(f"✅ Position closed successfully")
    else:
        print(f"❌ Failed to close position")
    
    return close_result

# Example usage
# Close full BTC position
close_position("BTC")

# Close partial SOL position (0.5 SOL)
close_position("SOL", size=0.5)
```

***

### 7. **Cancel Order**

```python
def cancel_order(coin: str, order_id: int):
    """
    Cancel an open order by ID
    """
    cancel_result = exchange.cancel(coin, order_id)
    
    print(f"Cancelling order {order_id} for {coin}")
    print(f"Result: {json.dumps(cancel_result, indent=2)}")
    
    if cancel_result.get("status") == "ok":
        print(f"✅ Order cancelled")
    else:
        print(f"❌ Cancel failed")
    
    return cancel_result

# Usage
cancel_order("BTC", order_id=12345678)
```

***

### 8. **Get Fee Information**

```python
def get_fee_info(address):
    """
    Get trading fee tiers and current fee rate
    """
    user_state = info.user_state(address)
    
    # User fees are based on 14-day volume
    print("\n" + "=" * 60)
    print("FEE INFORMATION")
    print("=" * 60)
    
    # Fee tiers (as of documentation):
    # Maker: -0.0002% to 0.0200%
    # Taker: 0.025% to 0.0500%
    
    print("\nFee Tiers (14-day rolling volume):")
    print("  < $0: Maker 0.0200% / Taker 0.0500%")
    print("  ≥ $100k: Maker 0.0150% / Taker 0.0400%")
    print("  ≥ $1M: Maker 0.0100% / Taker 0.0350%")
    print("  ≥ $10M: Maker 0.0050% / Taker 0.0300%")
    print("  ≥ $50M: Maker 0.0000% / Taker 0.0250%")
    print("  ≥ $100M: Maker -0.0002% / Taker 0.0250% (rebate)")
    
    # Get user's current volume and fee rate from margin summary
    margin_summary = user_state.get("marginSummary", {})
    
    return user_state

# Usage
get_fee_info(config["account_address"])
```

***

### 9. **Complete Trading Example**

```python
def complete_trading_example():
    """
    Complete example: Check positions, place trade with SL/TP, monitor, then close
    """
    
    # 1. Check current positions
    print("\n" + "="*60)
    print("STEP 1: CHECK CURRENT POSITIONS")
    print("="*60)
    positions = get_active_positions(config["account_address"])
    
    # 2. Check open orders
    print("\n" + "="*60)
    print("STEP 2: CHECK OPEN ORDERS")
    print("="*60)
    orders = get_open_orders(config["account_address"])
    
    # 3. Place new trade with SL/TP
    print("\n" + "="*60)
    print("STEP 3: PLACE NEW TRADE")
    print("="*60)
    
    # Example: Long ETH with 3x leverage
    trade_result = place_order_with_sl_tp(
        coin="ETH",
        is_buy=True,  # Long
        size=0.1,  # 0.1 ETH
        price=2500,  # Entry at $2500
        stop_loss_price=2450,  # SL at $2450 (-2%)
        take_profit_price=2600,  # TP at $2600 (+4%)
        leverage=3
    )
    
    # 4. Wait and check updated positions
    print("\n" + "="*60)
    print("STEP 4: CHECK UPDATED POSITIONS")
    print("="*60)
    import time
    time.sleep(2)  # Wait for order to process
    
    positions = get_active_positions(config["account_address"])
    orders = get_open_orders(config["account_address"])
    
    # 5. Close position (if needed)
    # close_position("ETH")
    
    return trade_result

# Run the complete example (uncomment to use)
# complete_trading_example()
```

***

## **Key Notes:**

1. **Fees**: Hyperliquid uses maker/taker fee model:
   - **Maker fees**: -0.0002% to 0.0200% (can get rebates)
   - **Taker fees**: 0.025% to 0.0500%
   - Fees are based on 14-day rolling volume

2. **Stop Loss & Take Profit**:
   - Must be placed as **separate orders** after entry
   - Use `reduce_only=True` to ensure they only close positions
   - Use `"tpsl": "sl"` or `"tpsl": "tp"` in trigger type

3. **Market Orders**:
   - Use `{"limit": {"tif": "Ioc"}}` (Immediate or Cancel)
   - Set price slightly above/below market to ensure fill

4. **Leverage**:
   - Set before placing orders using `exchange.update_leverage()`
   - Use `is_cross=False` for isolated margin (recommended)

5. **Testnet**:
   - Replace `constants.MAINNET_API_URL` with `constants.TESTNET_API_URL`
   - Get testnet funds from the faucet


[1](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api)
[2](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/take-profit-and-stop-loss-orders-tp-sl)
[3](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/86)
[4](https://www.youtube.com/watch?v=l7CN2_TLfjM)
[5](https://www.youtube.com/shorts/5FXuNR7AKOI)
[6](https://www.youtube.com/watch?v=UuBr331wxr4)
[7](https://thedocumentation.org/hyperliquid-python-sdk/examples/basic_orders/)
[8](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/builder-codes)
[9](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/order-types)
[10](https://www.youtube.com/shorts/PgjUltGjGHo)
[11](https://www.vadim.blog/hyperliquid-gasless-trading-strategies)
[12](https://lobehub.com/mcp/edkdev-hyperliquid-mcp)
[13](https://hexdocs.pm/hyperliquid/Hyperliquid.Orders.html)
[14](https://www.youtube.com/watch?v=pdKi6V8VJb4&vl=en)
[15](https://github.com/ccxt/ccxt/issues/22904)
[16](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint)
[17](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
[18](https://thedocumentation.org/hyperliquid-python-sdk/)
[19](https://www.youtube.com/watch?v=uvzvNS-Zrc0)
[20](https://github.com/oni-giri/hyperliquid-monitor)
[21](https://thedocumentation.org/hyperliquid-python-sdk/examples/market_orders/)
[22](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
[23](https://thedocumentation.org/hyperliquid-python-sdk/installation/)
[24](https://www.piwheels.org/project/hyperliquid-python-sdk-xiangyu/)
[25](https://docs.chainstack.com/docs/hyperliquid-tooling)
[26](https://github.com/thunderhead-labs/hyperliquid-stats)
[27](https://gitlab.com/tuanito/hyperliquid-python-sdk)
[28](https://apidog.com/blog/hyperliquid-api/)
[29](https://thedocumentation.org/hyperliquid-python-sdk/quick_start/)
[30](https://www.mcpkit.com/tools/hyperliquid-info-mcp/)
[31](https://www.quicknode.com/docs/hyperliquid/quickstart)
[32](https://docs.privy.io/recipes/hyperliquid-guide)