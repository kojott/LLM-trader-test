# 钉钉机器人接口说明

## 概述
本项目提供一个简单的 Python 模块 `dingtalk_bot.py`，用于通过钉钉自定义机器人 Webhook 发送文本消息。模块支持从 `dingconfig.ini` 读取 webhook 和加签 secret，自动生成签名并完成消息发送。

## 文件结构
- `dingtalk_bot.py`：钉钉机器人客户端，实现配置读取、签名以及发送文本消息。
- `dingconfig.ini`：配置文件模板，需要填写真实的 webhook 与 secret。
- `test_dingtalk_bot.py`：最小化测试脚本，演示如何快速验证机器人是否可用。

## 环境准备
1. 安装 Python 3.8 及以上版本。
2. 安装依赖库：
   ```bash
   pip install requests
   ```

## 配置说明 (`dingconfig.ini`)
`dingconfig.ini` 默认放在与脚本同目录下，内容示例：
```ini
[dingbot]
webhook = https://oapi.dingtalk.com/robot/send?access_token=替换为你的token
secret = 替换为你的密钥
```
- `webhook`：钉钉后台生成的自定义机器人 webhook，确保包含完整的 `access_token`。
- `secret`：启用“加签”后得到的密钥。如果未开启加签，可将 secret 字段保留为空字符串并在代码中自行调整签名逻辑。

## 用法示例
### 1. 作为脚本直接运行
```bash
python dingtalk_bot.py
```
运行后按照提示输入要发送的文本消息。脚本会读取 `dingconfig.ini` 并发送消息。

### 2. 调用测试脚本
```bash
python test_dingtalk_bot.py
```
`test_dingtalk_bot.py` 会发送一条内置的测试消息（请按需修改内容），可用于快速验证配置是否正确。

### 3. 在其他项目中复用
```python
from dingtalk_bot import DingTalkBot

bot = DingTalkBot.from_config()
bot.send_text("钉钉机器人测试消息", is_at_all=False)
```
也可以传入 `config_path` 指定其他配置位置，或通过 `at_mobiles` / `at_user_ids` 实现定向 @。

## 常见问题
- **请求返回非 0 errcode**：通常是 webhook 或 secret 填写错误，请重新复制粘贴钉钉后台提供的信息。
- **网络超时**：确认服务器可以访问钉钉域名，可适当增大 `send_text` 的 `timeout` 参数。
- **未启用加签**：若机器人未开启加签，钉钉文档要求去掉签名参数。可将 `_signed_webhook` 改为直接返回 `self.webhook`。

## 进一步扩展
- 按钉钉文档可增加 Markdown、链接、图文等消息类型，参考官方字段格式并在 `payload` 中调整。
- 结合定时任务、告警脚本使用时，建议增加日志与异常处理，确保失败可追踪。
