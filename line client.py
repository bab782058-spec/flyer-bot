"""
LINE Messaging API の送信と、Flexメッセージの部品(news_bot.py / meat_flyer_bot.py で共用)。
友だち全員へのブロードキャスト配信だけを扱う。通信は http_post に集約してある(テストではここを差し替える)。
"""

import requests

BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"
TIMEOUT_SECONDS = 30
TEXT_LIMIT = 4900  # LINEの上限は5000文字
DEFAULT_TEXT_COLOR = "#1A1A1A"


def http_post(url, headers, payload, timeout):
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


class LineClient:
    def __init__(self, channel_access_token):
        self._token = channel_access_token

    def send_flex(self, bubble, alt_text):
        self._broadcast({"type": "flex", "altText": alt_text, "contents": bubble})

    def send_text(self, text):
        self._broadcast({"type": "text", "text": text[:TEXT_LIMIT]})

    def _broadcast(self, message):
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        response = http_post(BROADCAST_URL, headers, {"messages": [message]}, TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"LINE送信に失敗しました: {response.status_code} {response.text}")


def flex_text(text, size="sm", color=DEFAULT_TEXT_COLOR, **extra):
    return {"type": "text", "text": text, "size": size, "color": color, **extra}


def flex_box(contents, layout="vertical", **extra):
    return {"type": "box", "layout": layout, "contents": contents, **extra}
