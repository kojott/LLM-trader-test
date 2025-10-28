"""Quick smoke test for DingTalkBot.

Replace the message before running the script.
"""
from dingtalk_bot import ConfigError, DingTalkBot


def main() -> None:
    try:
        bot = DingTalkBot.from_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return

    message = "钉钉机器人测试消息"
    try:
        response = bot.send_text(message)
    except Exception as exc:  # pragma: no cover - manual run helper
        print(f"Send failed: {exc}")
    else:
        print("Send succeeded:", response)


if __name__ == "__main__":
    main()
