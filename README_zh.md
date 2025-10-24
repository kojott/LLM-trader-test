# DeepSeek 模拟交易机器人

**文档:** [English](README.md) | [技术评审](TECHNICAL_REVIEW.md) | [技术文档](TECHNICAL_DOCUMENTATION.md) | [技术评审 (中文)](TECHNICAL_REVIEW_zh.md) | [技术文档 (中文)](TECHNICAL_DOCUMENTATION_zh.md)

这个代码库包含一个模拟交易机器人，它通过调用 Binance REST API 运行，同时利用 DeepSeek 进行交易决策。灵感来自 https://nof1.ai/ 挑战。您可以在 [llmtest.coininspector.pro](https://llmtest.coininspector.pro/) 上访问实时部署，在那里您可以访问仪表盘并查看完整的机器人对话日志。

该应用程序将其运行时数据（投资组合状态、AI 消息和交易历史）保存在专用的 `data/` 目录中，以便在 Docker 中运行时可以将其作为卷挂载。

## 运行时界面截图
![DeepSeek 交易机器人仪表盘](examples/screenshot.png)

## 工作原理
- 每三分钟，机器人会获取 `ETH`、`SOL`、`XRP`、`BTC`、`DOGE` 和 `BNB` 的最新 K 线数据，更新 EMA/RSI/MACD 指标，并快照当前持仓。
- 快照会转换成一个详细的 DeepSeek 提示，其中包括余额、未实现盈亏、未结订单和指标值。
- 一个交易规则系统提示（见下文）会与用户提示一起发送，这样模型在做决策前总能收到风险框架。
- DeepSeek 会以 JSON 格式回复每个资产的决策（`hold`、`entry` 或 `close`）。机器人会执行仓位大小控制、下单/平仓并保存结果。
- 投资组合状态、交易历史、AI 请求/响应以及每次迭代的控制台记录都会被写入 `data/` 目录，以供日后检查或在仪表盘中进行可视化。

## 系统提示和决策合约
DeepSeek 使用一个以风险为先的系统提示进行初始化，该提示强调：
- 单次交易风险不超过总资本的 1-2%
- 强制设置止损订单和预定义的退出计划
- 偏好趋势跟踪设置、耐心和书面的交易计划
- 以概率思维方式思考，同时控制仓位大小

每次迭代，DeepSeek 都会收到实时投资组合快照，并且必须 **只** 以类似以下的 JSON 格式进行回复：

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
    "invalidation_condition": "如果价格在 4 小时 EMA20 下方收盘",
    "justification": "动量 + RSI 在支撑位重置"
  }
}
```

如果 DeepSeek 回复 `hold`，机器人仍会在 `ai_decisions.csv` 中记录未实现盈亏、累计费用和理由。

## Telegram 通知
在 `.env` 文件中配置 `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID`，以便在每次迭代后接收消息。通知内容与控制台输出一致（开仓/平仓、投资组合摘要和任何警告），这样您无需查看日志即可跟踪进度。如果留空这些变量，则不会启用 Telegram 通知。

## 性能指标

控制台摘要和仪表盘会跟踪已实现和未实现的性能：

- **夏普比率**（仪表盘）是根据每次平仓后的余额快照计算得出的。
- **索提诺比率**（机器人 + 仪表盘）来自权益曲线，只惩罚下行波动，因此在样本量较小时更具参考价值。

默认情况下，索提诺比率假设无风险利率为 0%。您可以通过在 `.env` 文件中定义 `SORTINO_RISK_FREE_RATE`（年化小数，例如 `0.03` 表示 3%）或作为备用选项 `RISK_FREE_RATE` 来覆盖它。

## 先决条件

- Docker 24+（任何能够构建 Linux/AMD64 镜像的引擎）
- 一个包含所需 API 凭证的 `.env` 文件：
  - `BN_API_KEY` / `BN_SECRET` 用于 Binance 访问
  - `OPENROUTER_API_KEY` 用于 DeepSeek 请求
  - 可选：`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` 用于推送通知

## 构建镜像

```bash
docker build -t tradebot .
```

## 准备本地数据存储

在主机上创建一个目录，用于接收机器人的 CSV/JSON 文件：

```bash
mkdir -p ./data
```

容器会将所有内容存储在 `/app/data` 下。将您的主机文件夹挂载到该路径可以使交易历史和 AI 日志在多次运行之间保持不变。

## 在 Docker 中运行机器人

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  tradebot
```

- `--env-file .env` 将 API 密钥注入容器。
- 卷挂载使 `portfolio_state.csv`、`portfolio_state.json`、`ai_messages.csv`、`ai_decisions.csv` 和 `trade_history.csv` 文件保留在容器外部，以便您可以在本地检查它们。
- 默认情况下，应用程序会写入 `/app/data`。要覆盖此设置，请设置 `TRADEBOT_DATA_DIR` 并相应地更新卷挂载。

## 可选：Streamlit 仪表盘

要启动监控仪表盘而不是交易机器人，请运行：

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -p 8501:8501 \
  tradebot \
  streamlit run dashboard.py
```

然后在浏览器中打开 <http://localhost:8501> 即可访问 UI。

顶层指标包括夏普比率和索提诺比率，以及余额、权益和盈亏，以便您快速评估已实现回报和经下行风险调整后的性能。

## 本地运行（无 Docker）

虽然建议使用 Docker 以获得一致的环境，但您也可以直接在本地计算机上运行机器人和仪表盘。

### 1. 设置虚拟环境

最佳实践是使用虚拟环境来管理依赖项。

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate

# 或者在 Windows 上
# venv\Scripts\activate
```

### 2. 安装依赖

使用 pip 安装所需的 Python 包。

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env` 示例文件并添加您的 API 密钥。

```bash
cp .env.example .env
```
现在编辑 `.env` 文件以添加您的凭据。

### 4. 运行机器人

启动主交易机器人脚本。

```bash
python bot.py
```

### 5. 运行仪表盘

要查看仪表盘，请在单独的终端中运行以下命令（虚拟环境仍需激活）。

```bash
streamlit run dashboard.py
```

## 开发说明

- Docker 镜像设置了 `PYTHONDONTWRITEBYTECODE=1` 和 `PYTHONUNBUFFERED=1` 以获得更清晰的日志。
- 在没有 Docker 的情况下本地运行时，机器人仍会写入源代码树旁边的 `data/` 目录（如果设置了 `TRADEBOT_DATA_DIR`，则写入该目录）。
- `data/` 目录中的现有文件不会被自动覆盖；如果表头或列发生更改，请手动迁移文件。
- 代码库已经在 `data/` 中包含了示例 CSV 文件，因此您可以立即探索仪表盘。这些文件将在机器人运行时被覆盖。
