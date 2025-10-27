# LLM Trader 知识速览

## 功能概述
- **核心组件**：`bot.py` 提供基于 DeepSeek 的币安纸面交易机器人，每 3 分钟刷新一次 `ETH`、`SOL`、`XRP`、`BTC`、`DOGE` 与 `BNB` 的行情，计算 EMA/RSI/MACD 指标并生成账户快照，交由 DeepSeek 按 JSON 协议返回交易决策。
- **风险控制**：系统提示词要求模型遵循风控规则（单笔风险 1–2%、强制止损、提前制定退出计划、控制仓位），最终决策会记录到 `data/` 下的 CSV/JSON 以便审计。
- **数据持久化**：运行时会把组合状态、交易历史、AI 请求/响应、控制台日志写入 `data/` 目录，方便通过仪表盘或离线分析复盘。
- **可视化仪表盘**：`dashboard.py` 使用 Streamlit 展示收益、Sharpe/Sortino 比率、AI 日志与仓位明细，帮助快速了解机器人表现。

## 运行流程（Docker）
1. 准备 `.env`，填入 Binance (`BN_API_KEY`/`BN_SECRET`)、DeepSeek (`OPENROUTER_API_KEY`) 以及可选的 Telegram 通知令牌。
2. `docker build -t tradebot .` 构建镜像。
3. 建立宿主机 `./data` 目录并通过 `-v "$(pwd)/data:/app/data"` 挂载到容器。
4. 运行机器人：
   ```bash
   docker run --rm -it \
     --env-file .env \
     -v "$(pwd)/data:/app/data" \
     tradebot
   ```
5. 启动仪表盘（可选）：
   ```bash
   docker run --rm -it \
     --env-file .env \
     -v "$(pwd)/data:/app/data" \
     -p 8501:8501 \
     tradebot \
     streamlit run dashboard.py
   ```
   浏览器访问 <http://localhost:8501> 查看实时指标。

## Windows 原生环境能否运行？
可以，只需安装与 Docker 环境相同的依赖并指向项目根目录：
1. 安装 Python 3.10+、Git，以及 `requirements.txt` 中的依赖（`pip install -r requirements.txt`）。
2. 在项目根创建 `.env`，填写与 Docker 部署相同的密钥。
3. 直接运行：
   ```powershell
   python bot.py
   ```
   上述命令会启动交易机器人本身，持续拉取行情并写入 `data/`。

   如需查看可视化仪表盘，则单独运行（可与机器人同时运行，建议在第二个终端中执行）：
   ```powershell
   streamlit run dashboard.py
   ```
4. 默认仍会在项目根目录的 `data/` 文件夹写入和读取数据。如需自定义路径，可设置环境变量 `TRADEBOT_DATA_DIR`。

### Windows 环境注意事项
- 请确保使用 PowerShell 或 CMD 时具备访问网络及写入 `data/` 目录的权限。
- Streamlit、pandas、ta-lib 等依赖在 Windows 上可通过 `pip` 安装；若 ta-lib 编译遇到困难，可使用预编译轮子或参考其官方文档。
- 若希望与 Docker 行为保持一致，可在 Windows 上同样创建 `data` 文件夹以保存运行过程中生成的 CSV/JSON。

## 常见问题
- **日志/数据没有写入？** 确认 `data/` 目录存在且当前用户具备写权限，或检查 `TRADEBOT_DATA_DIR` 设置。
- **如何查看机器人决策？** 运行过程中可实时查看控制台输出，或打开 Streamlit 仪表盘的“AI 决策日志”表格。
- **能否推送 Telegram？** 在 `.env` 中配置 `TELEGRAM_BOT_TOKEN` 与 `TELEGRAM_CHAT_ID`，机器人每次迭代会自动推送通知。

> 以上信息整理自仓库 `README.md` 与源码，便于快速了解与部署 LLM Trader 项目。
