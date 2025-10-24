# DeepSeek Trading Bot - Technical Documentation

## 1. Introduction

This document provides detailed technical documentation for the DeepSeek Trading Bot. It is intended for developers who want to understand the bot's inner workings, set up a development environment, and extend its functionality.

## 2. Functionality

The DeepSeek Trading Bot is a paper-trading bot that uses the DeepSeek language model to make trading decisions. It operates on a predefined list of cryptocurrencies and interacts with the Binance API for market data.

### Core Features

-   **Multi-Asset Trading:** The bot can trade multiple cryptocurrencies simultaneously.
-   **AI-Powered Decisions:** Trading decisions are made by the DeepSeek language model based on market data and technical indicators.
-   **Risk Management:** The bot enforces a simple risk management rule of risking no more than 1-2% of capital per trade.
-   **Data Persistence:** The bot's state, trade history, and AI interactions are saved to disk, allowing for later analysis and monitoring.
-   **Telegram Notifications:** The bot can send notifications to a Telegram chat for real-time updates on its activities.
-   **Dashboard:** A Streamlit-based dashboard provides a graphical user interface for monitoring the bot's performance.

## 3. Development Setup

To set up a local development environment, you will need the following:

-   Python 3.8+
-   Docker (optional, but recommended for a consistent environment)
-   A code editor (e.g., VS Code)

### Local Installation (Without Docker)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DeepSeek-Paper-Trading-Bot.git
    cd DeepSeek-Paper-Trading-Bot
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**

    Copy the `.env.example` file to `.env` and fill in your API keys:

    ```bash
    cp .env.example .env
    ```

    ```
    BN_API_KEY="your_binance_api_key"
    BN_SECRET="your_binance_secret_key"
    OPENROUTER_API_KEY="your_openrouter_api_key"
    TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
    TELEGRAM_CHAT_ID="your_telegram_chat_id"
    ```

5.  **Run the bot:**

    ```bash
    python bot.py
    ```

6.  **Run the dashboard:**

    ```bash
    streamlit run dashboard.py
    ```

### Docker-Based Setup

1.  **Build the Docker image:**

    ```bash
    docker build -t tradebot .
    ```

2.  **Run the bot:**

    ```bash
    docker run --rm -it \
      --env-file .env \
      -v "$(pwd)/data:/app/data" \
      tradebot
    ```

3.  **Run the dashboard:**

    ```bash
    docker run --rm -it \
      --env-file .env \
      -v "$(pwd)/data:/app/data" \
      -p 8501:8501 \
      tradebot \
      streamlit run dashboard.py
    ```

## 4. Data Formats

The bot uses several CSV and JSON files to store its data. These files are located in the `data` directory.

### `portfolio_state.json`

This file stores the current state of the bot's portfolio, including the available balance and open positions.

```json
{
  "balance": 9845.12,
  "positions": {
    "ETH": {
      "side": "long",
      "quantity": 0.5,
      "entry_price": 3100.0,
      "profit_target": 3150.0,
      "stop_loss": 2880.0,
      "leverage": 5,
      ...
    }
  }
}
```

### `portfolio_state.csv`

This file logs the portfolio state at each iteration.

| timestamp | total_balance | total_equity | ... |
|---|---|---|---|
| 2023-10-27T10:00:00Z | 9845.12 | 9950.34 | ... |

### `trade_history.csv`

This file logs all executed trades.

| timestamp | coin | action | side | quantity | price | ... |
|---|---|---|---|---|---|---|
| 2023-10-27T10:00:00Z | ETH | ENTRY | long | 0.5 | 3100.0 | ... |

### `ai_decisions.csv`

This file logs the trading decisions made by the AI model.

| timestamp | coin | signal | reasoning | confidence |
|---|---|---|---|---|
| 2023-10-27T10:00:00Z | ETH | entry | Momentum + RSI reset | 0.72 |

### `ai_messages.csv`

This file logs the raw messages exchanged with the AI model.

| timestamp | direction | role | content | metadata |
|---|---|---|---|---|
| 2023-10-27T10:00:00Z | sent | user | ... | ... |
| 2023-10-27T10:00:01Z | received | assistant | ... | ... |

## 5. Extension Points

The bot is designed to be extensible. Here are some of the ways you can extend its functionality:

### Adding New Trading Symbols

To add a new trading symbol, simply add it to the `SYMBOLS` list in `bot.py`.

```python
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT", "NEWCOINUSDT"]
```

You will also need to add a mapping from the symbol to the coin in the `SYMBOL_TO_COIN` dictionary.

```python
SYMBOL_TO_COIN = {
    ...
    "NEWCOINUSDT": "NEWCOIN"
}
```

### Changing the AI Model

The bot uses the OpenRouter API to interact with the DeepSeek model. To use a different model, you will need to change the `model` parameter in the `call_deepseek_api` function in `bot.py`.

```python
response = requests.post(
    ...
    json={
        "model": "your-new-model-name",
        ...
    },
    ...
)
```

You may also need to adjust the prompt formatting in the `format_prompt_for_deepseek` function to match the requirements of the new model.

### Adding New Indicators

To add a new technical indicator, you can modify the `calculate_indicators` function in `bot.py`. You will need to use a library like `pandas-ta` or implement the indicator calculation yourself.

### Changing the Trading Logic

The core trading logic is located in the `main` function in `bot.py`. You can modify this function to implement your own trading strategies. For example, you could add more complex risk management rules, implement a trailing stop-loss, or use a different position sizing algorithm.
