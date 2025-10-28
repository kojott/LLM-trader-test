"""Simple DingTalk robot client that reads webhook & secret from a config file.

Usage:
    bot = DingTalkBot.from_config()
    bot.send_text("Hello from Codex")
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote_plus

import requests


_CONFIG_SECTION = "dingbot"


class ConfigError(Exception):
    """Raised when the DingTalk configuration cannot be loaded."""


@dataclass(slots=True)
class DingTalkBot:
    """Client wrapper for a DingTalk incoming webhook robot."""

    webhook: str
    secret: str

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "dingconfig.ini",
        section: str = _CONFIG_SECTION,
    ) -> "DingTalkBot":
        """Create a bot instance from an INI configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigError(f"Config file not found: {config_file}")

        parser = ConfigParser()
        parser.read(config_file, encoding="utf-8")

        if not parser.has_section(section):
            raise ConfigError(f"Config section [{section}] not found in {config_file}")

        try:
            webhook = parser.get(section, "webhook").strip()
            secret = parser.get(section, "secret").strip()
        except Exception as exc:  # pragma: no cover - generic configparser errors
            raise ConfigError("Config file is missing required keys 'webhook' or 'secret'.") from exc

        if not webhook:
            raise ConfigError("Webhook URL must not be empty.")
        if not secret:
            raise ConfigError("Secret must not be empty.")

        return cls(webhook=webhook, secret=secret)

    def send_text(
        self,
        message: str,
        *,
        at_mobiles: Optional[Iterable[str]] = None,
        at_user_ids: Optional[Iterable[str]] = None,
        is_at_all: bool = False,
        timeout: float = 8.0,
    ) -> dict:
        """Send a plain text message through the DingTalk robot."""
        if not message:
            raise ValueError("Message must not be empty.")

        payload = {
            "msgtype": "text",
            "text": {"content": message},
            "at": {
                "atMobiles": list(at_mobiles or []),
                "atUserIds": list(at_user_ids or []),
                "isAtAll": bool(is_at_all),
            },
        }

        url = self._signed_webhook()
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"DingTalk API error {data.get('errcode')}: {data.get('errmsg')}")
        return data

    def _signed_webhook(self) -> str:
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        signature = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        encoded = quote_plus(base64.b64encode(signature).decode("utf-8"))
        return f"{self.webhook}&timestamp={timestamp}&sign={encoded}"


def main() -> None:
    """Simple manual test when running the module directly."""
    try:
        bot = DingTalkBot.from_config()
    except ConfigError as exc:
        print(f"Failed to load config: {exc}")
        return

    message = input("Input message to send via DingTalk: ").strip()
    if not message:
        print("No message provided, aborting.")
        return

    try:
        result = bot.send_text(message)
    except Exception as exc:  # pragma: no cover - manual execution path
        print(f"Send failed: {exc}")
    else:
        print("Send succeeded:", result)


if __name__ == "__main__":
    main()
