"""
Gemini API を呼ぶ共通クライアント(news_bot.py / meat_flyer_bot.py で共用)。

- 使えるモデルは、モデル一覧API(models.list)から自動で取得し、新しい順に試す。
  新しいFlashが出ても、コードの修正は要らない。
- 混雑(503)・上限(429)・提供終了(404)のときは、待って再試行し、だめなら次のモデルへ切り替える。
- 「1日の無料枠を使い切った」429は、待っても回復しないので、再試行せずその実行中は使わない。

環境変数:
- GEMINI_API_KEY  (必須)
- GEMINI_MODELS   (任意。カンマ区切り。指定すると自動取得せず、この順番で試す。例: gemini-3.6-flash,gemini-3.1-flash-lite)
"""

import base64
import json
import os
import re
import sys
import time

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
LIST_MODELS_URL = f"{API_BASE}/models"
GENERATE_URL = f"{API_BASE}/models/{{model}}:generateContent"

# モデル一覧が取得できなかったときの、最後の手段
LAST_RESORT_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
MAX_MODEL_CANDIDATES = 6  # 候補が多すぎると、全滅時の待ち時間が際限なく伸びる

# gemini-3.6-flash / gemini-3.5-flash-lite のような「正式版のFlash」だけを対象にする。
# tts・image・live・日付付き・preview などの派生は、この形に合わないので自然に除外される。
STABLE_FLASH_PATTERN = re.compile(r"^gemini-(?P<version>\d+(?:\.\d+)*)-flash(?P<lite>-lite)?$")

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
SWITCH_MODEL_STATUSES = RETRYABLE_STATUSES | {404}  # 待っても直らなければ、次のモデルを試す価値がある

FIRST_MODEL_RETRY_WAITS = [20, 40, 60, 90, 90]  # 最優先モデルは合計300秒まで粘る
OTHER_MODEL_RETRY_WAITS = [10, 20, 30]  # 予備モデルは短めに(全体が長引かないように)

GENERATE_TIMEOUT_SECONDS = 180
LIST_MODELS_TIMEOUT_SECONDS = 30
GENERATE_METHOD = "generateContent"


class GeminiError(RuntimeError):
    """Gemini APIの呼び出しが、再試行・モデル切り替えを尽くしても失敗した"""


# ---------------------------------------------------------------------------
# HTTP(外部通信はこの2関数に集約。テストではここを差し替える)
# ---------------------------------------------------------------------------


def http_get(url, headers, timeout):
    return requests.get(url, headers=headers, timeout=timeout)


def http_post(url, headers, payload, timeout):
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


# ---------------------------------------------------------------------------
# モデルの選択
# ---------------------------------------------------------------------------


def rank_models(model_names):
    """正式版のFlashだけを残し、バージョンの新しい順(同じバージョンならlite以外が先)に並べる"""
    ranked = []
    for name in dict.fromkeys(model_names):
        match = STABLE_FLASH_PATTERN.match(name)
        if match:
            version = tuple(int(part) for part in match["version"].split("."))
            ranked.append((version, bool(match["lite"]), name))
    ranked.sort(key=lambda entry: (tuple(-v for v in entry[0]), entry[1]))
    return [name for _, _, name in ranked]


def list_generate_models(api_key):
    """このAPIキーで generateContent が使えるモデル名の一覧。取得に失敗したら空リスト"""
    headers = {"x-goog-api-key": api_key}
    names, page_token = [], None
    try:
        while True:
            url = f"{LIST_MODELS_URL}?pageSize=1000" + (f"&pageToken={page_token}" if page_token else "")
            response = http_get(url, headers, LIST_MODELS_TIMEOUT_SECONDS)
            if response.status_code != 200:
                print(f"[WARN] モデル一覧の取得に失敗: HTTP {response.status_code}", file=sys.stderr)
                return []
            body = response.json()
            for model in body.get("models", []):
                if GENERATE_METHOD in model.get("supportedGenerationMethods", []):
                    names.append(model["name"].removeprefix("models/"))
            page_token = body.get("nextPageToken")
            if not page_token:
                return names
    except (requests.RequestException, ValueError, KeyError) as error:
        print(f"[WARN] モデル一覧の取得に失敗: {error}", file=sys.stderr)
        return []


def parse_model_list(value):
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def resolve_models(api_key, pinned_models):
    """試すモデルの順番を決める。指定があればそれを優先、なければ自動取得、取れなければ最後の手段"""
    if pinned_models:
        return pinned_models
    ranked = rank_models(list_generate_models(api_key))[:MAX_MODEL_CANDIDATES]
    if not ranked:
        print(f"[WARN] 使えるモデルを自動取得できなかったため、{LAST_RESORT_MODELS} を使います", file=sys.stderr)
        return LAST_RESORT_MODELS
    return ranked


# ---------------------------------------------------------------------------
# 応答の扱い
# ---------------------------------------------------------------------------


def is_daily_quota_exhausted(response):
    """429のうち「1日あたりの上限」だけは、数十秒待っても回復しない(エラー本文のquotaIdに PerDay が入る)"""
    return response.status_code == 429 and "PerDay" in response.text


def extract_text(response_json):
    try:
        parts = response_json["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as error:
        raise GeminiError(f"Geminiの応答解析に失敗しました: {response_json}") from error


def parse_json_response(raw_text):
    """GeminiのJSON出力をパースする。念のため、```json で囲まれて返ってきた場合にも対応する"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    return json.loads(text)


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# クライアント
# ---------------------------------------------------------------------------


class GeminiClient:
    def __init__(self, api_key, models):
        self._api_key = api_key
        self._models = list(dict.fromkeys(models))
        self._quota_exhausted = set()  # この実行中は使わないモデル(1日の枠を使い切ったもの)

    @classmethod
    def from_env(cls):
        api_key = os.environ["GEMINI_API_KEY"]
        models = resolve_models(api_key, parse_model_list(os.environ.get("GEMINI_MODELS")))
        print(f"[INFO] Geminiモデルの試行順: {models}")
        return cls(api_key, models)

    def generate(self, prompt, image_base64=None, json_mode=False):
        """プロンプト(と任意の画像)を送り、出力テキストを返す"""
        payload = self._build_payload(prompt, image_base64, json_mode)
        failures = []

        usable_models = [model for model in self._models if model not in self._quota_exhausted]
        for index, model in enumerate(usable_models):
            waits = FIRST_MODEL_RETRY_WAITS if index == 0 else OTHER_MODEL_RETRY_WAITS
            response = self._post_with_retry(model, payload, waits)

            if response is not None and response.status_code == 200:
                if index > 0:
                    print(f"[INFO] 予備モデル {model} で成功しました")
                return extract_text(response.json())

            if response is None:
                failures.append(f"{model}: 接続できませんでした")
            elif response.status_code in SWITCH_MODEL_STATUSES:
                failures.append(f"{model}: HTTP {response.status_code} {response.text[:200]}")
                if is_daily_quota_exhausted(response):
                    self._quota_exhausted.add(model)
            else:
                # 400/401/403(キー不正など)はモデルを変えても直らないので、即エラーにする
                raise GeminiError(f"Gemini APIエラー(model={model}): {response.status_code} {response.text[:500]}")
            print(f"[WARN] {failures[-1][:150]} → 次のモデルを試します", file=sys.stderr)

        raise GeminiError("すべてのGeminiモデルで失敗しました: " + (" | ".join(failures) or "使えるモデルがありません"))

    @staticmethod
    def _build_payload(prompt, image_base64, json_mode):
        parts = [{"text": prompt}]
        if image_base64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})
        payload = {"contents": [{"parts": parts}]}
        if json_mode:
            payload["generationConfig"] = {"responseMimeType": "application/json"}
        return payload

    def _post_with_retry(self, model, payload, waits):
        """一時的な不調なら待って再試行する。最後の応答を返す(通信自体に失敗し続けたら None)"""
        url = GENERATE_URL.format(model=model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        response = None

        for attempt, wait in enumerate([*waits, None]):  # 最後の試行の後は待たない
            try:
                response = http_post(url, headers, payload, GENERATE_TIMEOUT_SECONDS)
                if response.status_code not in RETRYABLE_STATUSES:
                    return response
                if is_daily_quota_exhausted(response):
                    print(f"[WARN] {model} は1日の無料枠を使い切っています。待っても回復しないため再試行しません", file=sys.stderr)
                    return response
                reason = f"HTTP {response.status_code}"
            except (requests.ConnectionError, requests.Timeout) as error:
                response = None
                reason = f"通信エラー: {error}"

            if wait is None:
                break
            print(
                f"[WARN] {model} が一時的に使えません({reason})。{wait}秒待って再試行します({attempt + 1}/{len(waits)})",
                file=sys.stderr,
            )
            time.sleep(wait)

        return response
