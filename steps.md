# LLM Trader 项目部署指南 (Windows)

## 1. 项目概述

本项目是一个基于大型语言模型（LLM）的纸上交易机器人。其核心功能如下：

- **自动化交易决策**: 机器人定期（每3分钟）获取多种加密货币（如 `ETH`, `SOL`, `BTC` 等）的最新市场数据。
- **技术指标分析**: 利用获取的数据计算关键技术指标（EMA, RSI, MACD）。
- **AI 驱动**: 将市场数据和技术指标整合后，发送给 DeepSeek AI 模型，请求交易决策。
- **风险管理**: AI 在提供决策时，会遵循预设的风险管理框架，例如控制单笔交易风险、设置止损等。
- **数据持久化**: 所有的交易历史、账户状态、AI 请求与响应都会被记录在 `data/` 目录中，便于后续分析和复盘。
- **可视化仪表盘**: 项目提供一个基于 Streamlit 的 Web 仪表盘 (`dashboard.py`)，可以实时监控机器人表现、查看收益曲线、交易历史和 AI 决策日志。

## 2. Windows 环境设置与运行步骤

以下是在 Windows 操作系统上从零开始设置并运行此项目的详细步骤。

### 第一步：安装基础环境

1.  **安装 Python**:
    - 下载并安装 [Python 3.10 或更高版本](https://www.python.org/downloads/windows/)。
    - **重要**: 在安装过程中，请务必勾选 **"Add Python to PATH"** 选项，以便在命令行中直接使用 `python` 和 `pip` 命令。

2.  **安装 Git**:
    - 下载并安装 [Git for Windows](https://git-scm.com/download/win)。
    - 安装后，你将可以在 PowerShell 或命令提示符（CMD）中使用 Git 命令。

### 第二步：获取项目代码

打开 PowerShell 或 CMD，进入你希望存放项目的目录，然后运行以下命令克隆项目仓库。
(如果您已经拥有项目文件，可以跳过此步)
```powershell
git clone <repository_url>
cd LLM-trader-test
```

### 第三步：安装项目依赖

在项目的根目录下，使用 `pip` 安装 `requirements.txt` 文件中列出的所有 Python 库。

```powershell
pip install -r requirements.txt
```

**注意**: `ta-lib` 库在 Windows 上直接安装可能会失败。如果遇到问题，请尝试以下方法：
1.  访问 [Python 扩展包的非官方 Windows 二进制文件网站](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)。
2.  根据你的 Python 版本（如 cp310 代表 Python 3.10）和系统架构（`win_amd64` 代表 64 位），下载对应的 `.whl` 文件。
3.  使用 `pip` 安装下载好的文件，例如：
    ```powershell
    pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
    ```

### 第四步：配置环境变量

1.  在项目根目录下，找到 `.env.example` 文件，复制一份并将其重命名为 `.env`。

2.  使用文本编辑器打开 `.env` 文件，填入必要的 API 密钥：
    ```
    # 币安 API (纸上交易也需要)
    BN_API_KEY="YOUR_BINANCE_API_KEY"
    BN_SECRET="YOUR_BINANCE_API_SECRET"

    # 用于访问 DeepSeek 的 OpenRouter API
    OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"

    # (可选) Telegram 推送通知
    TELEGRAM_BOT_TOKEN=""
    TELEGRAM_CHAT_ID=""
    ```

### 第五步：创建数据存储目录

程序运行时会生成数据文件，需要一个 `data` 目录来存放它们。在项目根目录下运行：

```powershell
mkdir data
```

### 第六步：运行程序

1.  **运行交易机器人 (`bot.py`)**
    - 打开一个 PowerShell 或 CMD 终端，进入项目根目录，运行以下命令：
      ```powershell
      python bot.py
      ```
    - 机器人会开始运行，并持续输出日志信息。请保持此终端窗口开启。

2.  **运行可视化仪表盘 (`dashboard.py`)**
    - **打开一个新的** PowerShell 或 CMD 终端（不要关闭正在运行机器人的终端）。
    - 进入项目根目录，运行以下命令：
      ```powershell
      streamlit run dashboard.py
      ```
    - 运行后，你的默认浏览器会自动打开一个新标签页，地址为 `http://localhost:8501`。你可以在此页面上实时监控机器人的各项指标。

现在，您的 LLM 交易机器人已经在 Windows 环境下成功运行！
