# DeepSeek Paper Trading Bot

This repository contains a paper-trading bot (with optional Hyperliquid mainnet execution) that runs against the Binance REST API while leveraging DeepSeek for trade decision-making. Inspired by the https://nof1.ai/ challenge. A live deployment is available at [llmtest.coininspector.pro](https://llmtest.coininspector.pro/), where you can access the dashboard and review the complete bot conversation log.

The app persists its runtime data (portfolio state, AI messages, and trade history) inside a dedicated `data/` directory so it can be mounted as a volume when running in Docker.

---

## 🚀 Development Roadmap

**Support next-gen features through community sponsorship!** Each tier unlocks focused capabilities. Tiers must be funded in order.

| Tier | Feature | Progress |
|------|---------|----------|
| 🔒 **Tier 1** | Hyperliquid Live Execution | **$0 / $1,000** |
| 🔒 **Tier 2** | Emergency Controls & Monitoring | **$0 / $1,000** |
| 🔒 **Tier 3** | Smart Position Sizing | **$0 / $1,000** |
| 🔒 **Tier 4** | Portfolio Risk Limits | **$0 / $1,000** |
| 🔒 **Tier 5** | Multi-LLM Support | **$0 / $1,000** |
| 🔒 **Tier 6** | Strategy Voting System | **$0 / $1,000** |
| 🔒 **Tier 7** | Basic Backtesting | **$0 / $1,000** |
| 🔒 **Tier 8** | Advanced Backtesting | **$0 / $1,000** |
| 🔒 **Tier 9** | Performance Analytics | **$0 / $1,000** |
| 🔒 **Tier 10** | Smart Alerting & Reports | **$0 / $1,000** |

💰 **Sponsor:** Send $1,000 to unlock the next tier → [Details below](#development-roadmap--sponsorship)

---

## Dashboard Preview

The Streamlit dashboard provides real-time monitoring of the trading bot's performance, displaying portfolio metrics, equity curves benchmarked against BTC buy-and-hold, trade history, and AI decision logs.

### DeepSeek Trading Bot Dashboard
![DeepSeek Trading Bot Dashboard](examples/dashboard.png)

### DeepSeek Trading Bot Console
![DeepSeek Trading Bot Console](examples/screenshot.png)

## How It Works (Multi-Timeframe System)

### Timeframe Analysis
The bot uses a hierarchical 3-timeframe approach:

- **15-Minute (Execution)**: Precise entry timing, RSI14, MACD crossovers
- **1-Hour (Structure)**: Swing highs/lows, pullback identification, support/resistance
- **4-Hour (Trend)**: Overall bias (bullish/bearish/neutral), major EMAs, ATR for stops

### Trading Loop (Every 15 minutes)
1. **Fetch Market Data**: Retrieves 200× 15m candles, 100× 1h candles, 100× 4h candles
2. **Calculate Indicators**: EMA 20/50/200, RSI14, MACD, ATR, volume analysis
3. **Build Rich Prompt**: Formats multi-timeframe data with clear hierarchy
4. **AI Decision**: DeepSeek analyzes using system prompt rules
5. **Execute Trades**: Validates AI decisions against risk management rules
6. **Monitor Positions**: Checks for stop loss, take profit, or structural breaks

### Entry Types
- **Type A (With-Trend)**: 4H trend + 1H pullback + 15M reversal signal (2% risk)
- **Type B (Counter-Trend)**: 4H extreme RSI + major level + strong reversal (1% risk)
- **Type C (Range)**: Neutral 4H market, trade swing_high/swing_low (1% risk)

### Exit Rules
Positions close ONLY when:
1. Stop loss or take profit is hit
2. 1H structure breaks (closes beyond swing_high/swing_low)
3. 4H major trend reverses (closes beyond EMA50 + MACD flip)
4. Within 20% of stop loss distance = **DO NOT manually close** (let SL work)

### What Changed from 3m System
- ❌ Removed: 3-minute noise, RSI7, subjective "weak momentum" exits
- ✅ Added: 1-hour structure layer, mechanical exit rules, 20% proximity rule
- ✅ Improved: Clearer timeframe hierarchy, confluence requirements, risk scaling by trade type

## System Prompt & Decision Contract
DeepSeek is primed with a risk-first system prompt that stresses:
- Never risking more than 1–2% of capital on a trade
- Mandatory stop-loss orders and pre-defined exit plans
- Favouring trend-following setups, patience, and written trade plans
- Thinking in probabilities while keeping position sizing under control

Each iteration DeepSeek receives the live portfolio snapshot and must answer **only** with JSON resembling:

```json
{
  "ETH": {
    "signal": "entry",
    "side": "long",
    "quantity": 0.5,
    "profit_target": 3150.0,
    "stop_loss": 2880.0,
    "leverage": 5,
    "confidence": 0.72,
    "risk_usd": 150.0,
    "invalidation_condition": "If price closes below 4h EMA20",
    "justification": "Momentum + RSI reset on support"
  }
}
```

If DeepSeek responds with `hold`, the bot still records unrealised PnL, accumulated fees, and the rationale in `ai_decisions.csv`.

Need to iterate on the playbook? Set `TRADEBOT_SYSTEM_PROMPT` directly in `.env`, or point `TRADEBOT_SYSTEM_PROMPT_FILE` at a text file to swap the default rules. The backtester honours `BACKTEST_SYSTEM_PROMPT` and `BACKTEST_SYSTEM_PROMPT_FILE` so you can trial alternative prompts without touching live settings.

## Telegram Notifications
Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to receive a message after every iteration. The notification mirrors the console output (positions opened/closed, portfolio summary, and any warnings) so you can follow progress without tailing logs.

Additionally you can set a dedicated signals group for trade-entry/exit signals using `TELEGRAM_SIGNALS_CHAT_ID`. When this is set the bot will send rich, Markdown-formatted ENTRY and CLOSE signals (only) to that chat — these messages include:
- **ENTRY signals**: Asset, direction, leverage, entry price, position size, margin, risk, profit targets, stop-loss levels, R/R ratio, liquidity type, confidence percentage, entry fees, and AI reasoning
- **CLOSE signals**: Asset, direction, size, entry/exit prices, price change %, gross/net P&L, fees paid, ROI %, updated balance, and exit reasoning

The signals use emojis (🟢 for LONG, 🔴 for SHORT, ✅ for profit, ❌ for loss) and structured Markdown formatting for easy reading on mobile devices. If `TELEGRAM_SIGNALS_CHAT_ID` is not set, ENTRY/CLOSE signals will not be sent to a separate group (the general `TELEGRAM_CHAT_ID` remains used for iteration summaries and errors).

Leave the variables empty to run without Telegram.

## Performance Metrics

The console summary and dashboard track both realized and unrealized performance:

- **Sharpe ratio** (dashboard) is computed from closed trades using balance snapshots after each exit.
- **Sortino ratio** (bot + dashboard) comes from the equity curve and penalises downside volatility only, making it more informative when the sample size is small.

By default the Sortino ratio assumes a 0% risk-free rate. Override it by defining `SORTINO_RISK_FREE_RATE` (annualized decimal, e.g. `0.03` for 3%) or, as a fallback, `RISK_FREE_RATE` in your `.env`.

## Prerequisites

- Docker 24+ (any engine capable of building Linux/AMD64 images)
- A `.env` file with the required API credentials:
  - `BN_API_KEY` / `BN_SECRET` for Binance access
  - `OPENROUTER_API_KEY` for DeepSeek requests
  - Optional: `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` for push notifications
  - Optional: Hyperliquid live-trading variables (see below)

## Hyperliquid Live Trading (Optional)

The bot runs in paper-trading mode by default and never touches live capital. To forward fills to Hyperliquid mainnet:

- Install the extra dependency (`pip install hyperliquid-python-sdk`) or rely on the updated `requirements.txt`.
- Set the following variables in `.env`:
  - `HYPERLIQUID_LIVE_TRADING=true`
  - `HYPERLIQUID_WALLET_ADDRESS=0xYourWallet`
  - `HYPERLIQUID_PRIVATE_KEY=your_private_key_or_vault_key`
  - `HYPERLIQUID_CAPITAL=500` (used for position sizing / risk limits)
- Optionally adjust `PAPER_START_CAPITAL` to keep a separate paper account value when live trading is disabled.
- To perform a tiny live round-trip sanity check, run `python scripts/manual_hyperliquid_smoke.py --coin BTC --notional 2 --leverage 1`. Passing `BTC-USDC` works as well; the script automatically maps both forms to the correct Hyperliquid market, opens a ~2 USD taker position, attaches TP/SL, waits briefly, and closes the trade.

When live mode is active the bot submits IOC (market-like) entry/exit orders and attaches reduce-only stop-loss / take-profit triggers on Hyperliquid mainnet using isolated leverage. If initialization fails (missing SDK, credentials, etc.) the bot falls back to paper trading and logs a warning. Treat your private key with care—avoid checking it into version control and prefer a dedicated trading wallet.

## Build the Image

```bash
docker build -t tradebot .
```

## Prepare Local Data Storage

Create a directory on the host that will receive the bot's CSV/JSON artifacts:

```bash
mkdir -p ./data
```

The container stores everything under `/app/data`. Mounting your host folder to that path keeps trade history and AI logs between runs.

## Run the Bot in Docker

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  tradebot
```

- `--env-file .env` injects API keys into the container.
- The volume mount keeps `portfolio_state.csv`, `portfolio_state.json`, `ai_messages.csv`, `ai_decisions.csv`, and `trade_history.csv` outside the container so you can inspect them locally.
- By default the app writes to `/app/data`. To override, set `TRADEBOT_DATA_DIR` and update the volume mount accordingly.

## Optional: Streamlit Dashboard

To launch the monitoring dashboard instead of the trading bot, run:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -p 8501:8501 \
  tradebot \
  streamlit run dashboard.py
```

Then open <http://localhost:8501> to access the UI.

The top-level metrics include Sharpe and Sortino ratios alongside balance, equity, and PnL so you can quickly assess both realised returns and downside-adjusted performance.

---

## Reconcile Portfolio State After Editing Trades

If you manually edit `data/trade_history.csv` (for example, deleting erroneous trades) run the reconciliation helper to rebuild `portfolio_state.json` from the remaining rows:

```bash
python3 scripts/recalculate_portfolio.py
```

- The script replays the trade log from the configured starting capital (respects `PAPER_START_CAPITAL`, `HYPERLIQUID_CAPITAL`, and `HYPERLIQUID_LIVE_TRADING`).
- Open positions are recreated with their margin, leverage, and risk metrics; the resulting balance and positions are written to `data/portfolio_state.json`.
- Use `--dry-run` to inspect the reconstructed state without updating files, or `--start-capital 7500` to override the initial balance.

This keeps the bot's persisted state consistent with the edited trade history before restarting the live loop.

---

## Historical Backtesting

The repository ships with a replay harness (`backtest.py`) so you can evaluate prompts and LLM choices on cached Binance data without touching the live loop.

### 1. Configure the Environment

Add any of the following keys to your `.env` when running a backtest (all are optional and fall back to the live defaults):

- `BACKTEST_DATA_DIR` – root folder for cached candles and run artifacts (default `data-backtest/`)
- `BACKTEST_START` / `BACKTEST_END` – UTC timestamps (`2024-01-01T00:00:00Z` format) that define the evaluation window
- `BACKTEST_INTERVAL` – primary bar size (`3m` by default); a 4h context stream is fetched automatically
- `BACKTEST_LLM_MODEL`, `BACKTEST_TEMPERATURE`, `BACKTEST_MAX_TOKENS`, `BACKTEST_LLM_THINKING`, `BACKTEST_SYSTEM_PROMPT`, `BACKTEST_SYSTEM_PROMPT_FILE` – override the model, sampling parameters, and system prompt without touching your live settings
- `BACKTEST_START_CAPITAL` – initial equity used for balance/equity calculations
- `BACKTEST_DISABLE_TELEGRAM` – set to `true` to silence notifications during the simulation

You can also keep distinct live overrides via `TRADEBOT_LLM_MODEL`, `TRADEBOT_LLM_TEMPERATURE`, `TRADEBOT_LLM_MAX_TOKENS`, `TRADEBOT_LLM_THINKING`, and `TRADEBOT_SYSTEM_PROMPT` / `TRADEBOT_SYSTEM_PROMPT_FILE` if you want different prompts or thinking budgets in production.

### 2. Run the Backtest

```bash
python3 backtest.py
```

The runner automatically:

1. Loads `.env`, forces paper-trading mode, and injects the backtest overrides into the bot.
2. Downloads any missing Binance klines into `data-backtest/cache/` (subsequent runs reuse the cache).
3. Iterates through each bar in the requested window, calling the LLM for fresh decisions at every step.
4. Reuses the live execution engine so position management, fee modelling, and CSV logging behave identically.

#### Option B: Run in Docker

Launch containerised backtests (handy for running several windows in parallel) via the helper script:

```bash
./scripts/run_backtest_docker.sh 2024-01-01T00:00:00Z 2024-01-07T00:00:00Z prompts/system_prompt.txt
```

- Pass start/end timestamps in UTC; provide a prompt file or `-` to reuse the default rules.
- The script ensures the Docker image exists, mounts `data-backtest` so results land in `data-backtest/run-<id>/`, and forwards all relevant env vars into the container.
- Tweak behaviour with `DOCKER_IMAGE`, `DOCKER_ENV_FILE`, `BACKTEST_INTERVAL`, or `BACKTEST_RUN_ID` environment variables before invoking the script.
- Because each run gets its own container name and run id you can kick off multiple tests concurrently without clashing directories.

### 3. Inspect the Results

Each run is written to a timestamped directory (e.g. `data-backtest/run-20240101-120000/`) that mirrors the live layout:

- `portfolio_state.csv`, `trade_history.csv`, `ai_decisions.csv`, `ai_messages.csv` contain the full replay trace.
- `backtest_results.json` summarises the run (final equity, return %, Sortino ratio, max drawdown, realised PnL, trade counts, LLM config, etc.). A fresh JSON file is generated for every run—nothing is overwritten.

Because the backtester drives the same modules as production you can plug the CSVs directly into the Streamlit dashboard (point `TRADEBOT_DATA_DIR` at a run folder) or external analytics tools.

---

## Development Roadmap & Sponsorship

This project evolves through community sponsorship. Each **$1,000 tier** unlocks focused capabilities. Development begins once a tier is fully funded (estimated 1-2 weeks per tier). All code remains open-source.

### Current Status

**🔒 Tier 1 is next** - Hyperliquid Live Execution needs funding to begin development.

### 🎯 Tier 1: Hyperliquid Live Execution
**Goal: $1,000 | Funded: $0**

Core live trading on Hyperliquid mainnet:
- IOC order execution with retry logic
- Basic position tracking

### 🛡️ Tier 2: Emergency Controls & Monitoring
**Goal: $1,000 | Funded: $0**

Safety and transparency:
- Kill-switch (Telegram command + env variable)
- Slippage tracking and audit logging
- Enhanced smoke test suite

### 📊 Tier 3: Smart Position Sizing
**Goal: $1,000 | Funded: $0**

Dynamic risk-based sizing:
- Volatility-adjusted position sizing (ATR)
- Account equity percentage rules
- Trailing stops implementation

### 🔒 Tier 4: Portfolio Risk Limits
**Goal: $1,000 | Funded: $0**

Portfolio-level protection:
- Max total exposure limits
- Correlation analysis between assets
- Daily loss limits with auto-pause
- Risk heat maps in dashboard

### 🤖 Tier 5: Multi-LLM Support
**Goal: $1,000 | Funded: $0**

Compare AI performance:
- Add GPT-5 and Claude support
- Side-by-side LLM comparison
- Per-model performance tracking
- Easy model switching

### 🧠 Tier 6: Strategy Voting System
**Goal: $1,000 | Funded: $0**

Run multiple strategies:
- Multiple prompt personalities (conservative/aggressive/counter-trend)
- Weighted voting on decisions
- Hot-swap strategies without restart
- Individual strategy P&L tracking

### 📈 Tier 7: Basic Backtesting
**Goal: $1,000 | Funded: $0**

Test on historical data:
- Historical OHLCV data pipeline
- Simple simulation engine
- Basic performance metrics
- CSV report generation

### 🔬 Tier 8: Advanced Backtesting
**Goal: $1,000 | Funded: $0**

Professional validation:
- Monte Carlo analysis
- Walk-forward optimization
- Realistic slippage/commissions
- Parameter sensitivity testing

### 📊 Tier 9: Performance Analytics
**Goal: $1,000 | Funded: $0**

Deep insights:
- ML-based anomaly detection
- Advanced metrics (VaR, CVaR, rolling Sharpe/Sortino)
- Profit factor analysis by asset/timeframe
- Market regime detection

### 🚨 Tier 10: Smart Alerting & Reports
**Goal: $1,000 | Funded: $0**

Intelligence layer:
- Context-aware alerting (pattern-based, not just thresholds)
- Automated weekly performance reports
- Multi-channel alerts (Email/Telegram/Discord)
- Custom dashboard exports

### 💰 How to Sponsor

1. **Choose Tier 1** (must fund in order)
2. **Send $1,000** to: `0x4B1bEd654BA86F64441037ad0A7D2ce54321B381` (Ethereum)
3. **Create Issue** with transaction ID
4. **Track Progress** - Development starts once funded

**Sponsor Benefits:**
- Early access to new features
- Listed as project sponsor in README
- Direct input on feature priorities
- Weekly progress updates

### 📧 Contact

Questions about sponsorship? Reach out via:
- **Email:** [kojott@gmail.com]
- **Twitter:** [@kojott]
- **Telegram:** [@kojottchorche]

---

## Disclaimer

This repository is provided strictly for experimental and educational purposes. You alone choose how to use it and you bear 100% of the financial risk. I do not offer trading advice, I make no promises of profitability, and I am not responsible for any losses, damages, or missed opportunities that arise from running this project in any environment.

Please keep the following in mind before you deploy anything derived from this code:

- There is no token, airdrop, or fundraising effort associated with this work; if someone claims otherwise, they are not connected to me.
- The bot does not ship with a complete trading system. Every result depends on your own research, testing, risk controls, and execution discipline.
- Market conditions change quickly. Past backtests, paper trades, or screenshots are not guarantees of future performance.
- No LLM, agent, or automated component can remove the inherent risk from trading. Validate everything yourself before real capital is at stake.

By using this repository you acknowledge that you are solely responsible for configuring, auditing, and running it, and that you accept all associated risks.

## Development Notes

- The Docker image sets `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` for cleaner logging.
- When running locally without Docker, the bot still writes to the `data/` directory next to the source tree (or to `TRADEBOT_DATA_DIR` if set).
- Existing files inside `data/` are never overwritten automatically; if headers or columns change, migrate the files manually.
- The repository already includes sample CSV files in `data/` so you can explore the dashboard immediately. These files will be overwritten as the bot runs.
