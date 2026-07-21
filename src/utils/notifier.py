import json
import os
import sys
import urllib.request
from logging import ERROR
from pathlib import Path
from typing import Optional

from src.utils.logger import log

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_webhook_url_from_env_file() -> Optional[str]:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "NOTIFY_WEBHOOK_URL":
            return value.strip() or None
    return None


def send_webhook_notification(message: str, webhook_url: Optional[str] = None) -> bool:
    webhook_url = webhook_url or os.getenv("NOTIFY_WEBHOOK_URL") or _read_webhook_url_from_env_file()
    if not webhook_url:
        return False

    payload = {"content": message} if "discord" in webhook_url else {"text": message}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        # Discord's edge blocks the default "Python-urllib/x.y" User-Agent with a 403.
        "User-Agent": "RealFLConsumption-Notifier/1.0",
    }
    req = urllib.request.Request(webhook_url, data=data, headers=headers)
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        log(ERROR, f"Failed to send webhook notification: {e}")
        return False


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NOTIFY_MESSAGE", "")
    send_webhook_notification(message)
