# DeepSeek 交易机器人 - 技术文档

## 1. 简介

本文档为 DeepSeek 交易机器人提供详细的技术文档。它面向希望了解机器人内部工作原理、设置开发环境并扩展其功能的开发者。

## 2. 功能

DeepSeek 交易机器人是一个模拟交易机器人，使用 DeepSeek 语言模型进行交易决策。它针对预定义的加密货币列表进行操作，并与币安 API 交互以获取市场数据。

### 核心特性

-   **多资产交易:** 机器人可以同时交易多种加密货币。
-   **AI 驱动决策:** 交易决策由 DeepSeek 语言模型根据市场数据和技术指标制定。
-   **风险管理:** 机器人执行简单的风险管理规则，即每笔交易的风险不超过资本的 1-2%。
-   **数据持久化:** 机器人的状态、交易历史和 AI 交互被保存到磁盘，以便后续分析和监控。
-   **Telegram 通知:** 机器人可以向 Telegram 聊天发送通知，以实时更新其活动。
-   **仪表盘:** 基于 Streamlit 的仪表盘提供图形用户界面，用于监控机器人的性能。

## 3. 开发设置

要设置本地开发环境，您需要以下内容：

-   Python 3.8+
-   Docker（可选，但建议用于一致的环境）
-   代码编辑器（例如 VS Code）

### 本地安装（不使用 Docker）

1.  **克隆代码库:**

    ```bash
    git clone https://github.com/your-username/DeepSeek-Paper-Trading-Bot.git
    cd DeepSeek-Paper-Trading-Bot
    ```

2.  **创建虚拟环境:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装所需依赖:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **创建 `.env` 文件:**

    将 `.env.example` 文件复制到 `.env` 并填入您的 API 密钥：

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

5.  **运行机器人:**

    ```bash
    python bot.py
    ```

6.  **运行仪表盘:**

    ```bash
    streamlit run dashboard.py
    ```

### 基于 Docker 的设置

1.  **构建 Docker 镜像:**

    ```bash
    docker build -t tradebot .
    ```

2.  **运行机器人:**

    ```bash
    docker run --rm -it \
      --env-file .env \
      -v "$(pwd)/data:/app/data" \
      tradebot
    ```

3.  **运行仪表盘:**

    ```bash
    docker run --rm -it \
      --env-file .env \
      -v "$(pwd)/data:/app/data" \
      -p 8501:8501 \
      tradebot \
      streamlit run dashboard.py
    ```

## 4. 数据格式

机器人使用多个 CSV 和 JSON 文件来存储其数据。这些文件位于 `data` 目录中。

### `portfolio_state.json`

此文件存储机器人投资组合的当前状态，包括可用余额和未平仓头寸。

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

此文件记录每次迭代时的投资组合状态。

| timestamp | total_balance | total_equity | ... |
|---|---|---|---|
| 2023-10-27T10:00:00Z | 9845.12 | 9950.34 | ... |

### `trade_history.csv`

此文件记录所有已执行的交易。

| timestamp | coin | action | side | quantity | price | ... |
|---|---|---|---|---|---|---|
| 2023-10-27T10:00:00Z | ETH | ENTRY | long | 0.5 | 3100.0 | ... |

### `ai_decisions.csv`

此文件记录 AI 模型做出的交易决策。

| timestamp | coin | signal | reasoning | confidence |
|---|---|---|---|---|
| 2023-10-27T10:00:00Z | ETH | entry | Momentum + RSI reset | 0.72 |

### `ai_messages.csv`

此文件记录与 AI 模型的原始消息交换。

| timestamp | direction | role | content | metadata |
|---|---|---|---|---|
| 2023-10-27T10:00:00Z | sent | user | ... | ... |
| 2023-10-27T10:00:01Z | received | assistant | ... | ... |

## 5. 扩展点

该机器人设计为可扩展的。以下是扩展其功能的一些方法：

### 添加新的交易品种

要添加新的交易品种，只需将其添加到 `bot.py` 中的 `SYMBOLS` 列表中。

```python
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT", "NEWCOINUSDT"]
```

您还需要在 `SYMBOL_TO_COIN` 字典中添加从交易品种到币种的映射。

```python
SYMBOL_TO_COIN = {
    ...
    "NEWCOINUSDT": "NEWCOIN"
}
```

### 更改 AI 模型

该机器人使用 OpenRouter API 与 DeepSeek 模型进行交互。要使用不同的模型，您需要在 `bot.py` 的 `call_deepseek_api` 函数中更改 `model` 参数。

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

您可能还需要调整 `format_prompt_for_deepseek` 函数中的提示格式，以匹配新模型的要求。

### 添加新指标

要添加新的技术指标，您可以修改 `bot.py` 中的 `calculate_indicators` 函数。您需要使用像 `pandas-ta` 这样的库或自己实现指标计算。

### 更改交易逻辑

核心交易逻辑位于 `bot.py` 的 `main` 函数中。您可以修改此函数以实现您自己的交易策略。例如，您可以添加更复杂的风险管理规则，实施追踪止损，或使用不同的仓位调整算法。
