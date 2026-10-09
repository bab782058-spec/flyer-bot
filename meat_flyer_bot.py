"""
マルエツ 平間店(トクバイ掲載)のチラシから、本日有効な生肉の特売情報だけを
Geminiの画像認識で抽出し、LINEに送るスクリプト。

流れ:
    トクバイの店舗ページ → チラシID一覧 → チラシ画像 → Gemini(生肉だけ抽出) → LINE

環境変数(GitHub Actionsの Secrets / Variables で設定する):
    GEMINI_API_KEY                  Gemini APIキー(news_bot.pyと共通でOK)
    LINE_CHANNEL_ACCESS_TOKEN_MEAT  このbot専用のLINE公式アカウントのトークン
    GEMINI_MODEL                    (任意)主モデル。未指定なら DEFAULT_GEMINI_MODEL
    GEMINI_FALLBACK_MODELS          (任意)カンマ区切りの予備モデル。未指定なら DEFAULT_FALLBACK_MODELS

注意:
    個人利用・1日1回程度のアクセスを想定している。高頻度アクセスやデータの再配布はしないこと。
"""

import base64
import datetime
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------

STORE_NAME = "マルエツ 平間店"
NO_DEALS_MESSAGE = "本日は生肉の特売情報が見つかりませんでした。"
STORE_PAGE_URL = "https://tokubai.co.jp/%E3%83%9E%E3%83%AB%E3%82%A8%E3%83%84/13082"

# gemini-flash-latest は「常に最新のFlash」を指すエイリアス。挙動を固定したいときは
# GitHubの Variables に GEMINI_MODEL を登録して、バージョンを明示する。
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
# 主モデルが混雑(503)・上限(429)・提供終了(404)のときに順に試す予備モデル。
DEFAULT_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"

# 一時的な不調なので、待って再試行する価値があるステータス
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
# 再試行しても直らないが、別のモデルなら成功しうるステータス(404 = モデル提供終了)
FALLBACK_STATUSES = RETRYABLE_STATUSES | {404}
PRIMARY_MODEL_RETRY_WAITS = [10, 20, 40]  # 主モデルは粘る
FALLBACK_MODEL_RETRY_WAITS = [5, 10]  # 予備モデルは短めに(全体が長引かないように)

GEMINI_TIMEOUT_SECONDS = 120
HTTP_TIMEOUT_SECONDS = 30
GEMINI_MAX_OUTPUT_TOKENS = 4096

LEAFLET_ID_PATTERN = re.compile(r"/leaflets/(\d+)")
# & を除外するのは、HTMLエスケープされた &quot; の手前で止めるため
LEAFLET_IMAGE_URL_PATTERN = re.compile(
    r'https://image\.tokubai\.co\.jp/images/[^\s"\'<>()&]+\.jpg(?:\?[^\s"\'<>()&]*)?'
)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

JST = datetime.timezone(datetime.timedelta(hours=9))
WEEKDAYS_JP = ["月", "火", "水", "木", "金", "土", "日"]

# LINEカードの配色
COLOR_HEADER_BG = "#8B3A3A"
COLOR_HEADER_SUBTEXT = "#F0D9D9"
COLOR_ACCENT = "#B23B3B"
COLOR_TEXT = "#1A1A1A"
COLOR_MUTED = "#666666"
COLOR_WHITE = "#FFFFFF"


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------


class GeminiError(RuntimeError):
    """Gemini APIの呼び出しが、リトライ・フォールバックを尽くしても失敗した"""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    line_token: str
    gemini_model: str
    fallback_models: list


@dataclass(frozen=True)
class MeatDeal:
    """チラシから読み取った、生肉1商品分の特売情報"""

    product: str
    price: str
    discount: Optional[str] = None

    @classmethod
    def from_dict(cls, raw):
        return cls(
            product=str(raw.get("product", "")),
            price=str(raw.get("price", "")),
            discount=raw.get("discount") or None,
        )

    def as_text(self):
        suffix = f"({self.discount})" if self.discount else ""
        return f"・{self.product} {self.price}{suffix}"


@dataclass(frozen=True)
class CollectionResult:
    deals: list
    leaflets_total: int
    leaflets_failed: int

    @property
    def all_failed(self):
        return self.leaflets_total > 0 and self.leaflets_failed == self.leaflets_total


def load_settings():
    fallbacks_env = os.environ.get("GEMINI_FALLBACK_MODELS")
    if fallbacks_env is None:
        fallback_models = DEFAULT_FALLBACK_MODELS
    else:
        fallback_models = [m.strip() for m in fallbacks_env.split(",") if m.strip()]

    return Settings(
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        line_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN_MEAT"],
        gemini_model=os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
        fallback_models=fallback_models,
    )


# ---------------------------------------------------------------------------
# 日付
# ---------------------------------------------------------------------------


def format_date_jp(moment):
    return f"{moment.year}年{moment.month}月{moment.day}日({WEEKDAYS_JP[moment.weekday()]})"


def today_label():
    return format_date_jp(datetime.datetime.now(JST))


# ---------------------------------------------------------------------------
# HTTP(外部通信はここ2関数に集約。テストではこの2つを差し替える)
# ---------------------------------------------------------------------------


def http_get(url):
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def http_post(url, headers, payload, timeout):
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


# ---------------------------------------------------------------------------
# トクバイ(チラシの取得)
# ---------------------------------------------------------------------------


def fetch_leaflet_ids():
    """店舗ページから、掲載中のチラシID一覧を返す(重複除去・掲載順)"""
    html = http_get(STORE_PAGE_URL).text
    return list(dict.fromkeys(LEAFLET_ID_PATTERN.findall(html)))


def fetch_leaflet_image_url(leaflet_id):
    """チラシIDから画像ファイルのURLを返す。見つからなければ None"""
    html = http_get(f"{STORE_PAGE_URL}/leaflets/{leaflet_id}/show_for_widget?from=leaflet_widget").text
    match = LEAFLET_IMAGE_URL_PATTERN.search(html)
    return match.group(0).rstrip(").,、。") if match else None


def fetch_image_base64(image_url):
    return base64.b64encode(http_get(image_url).content).decode("ascii")


# ---------------------------------------------------------------------------
# Gemini(画像から生肉の特売を抽出)
# ---------------------------------------------------------------------------


class GeminiClient:
    """主モデルが使えないとき、予備モデルへ自動で切り替えながらGeminiを呼ぶ"""

    def __init__(self, api_key, primary_model, fallback_models):
        self._api_key = api_key
        self._models = list(dict.fromkeys([primary_model, *fallback_models]))

    def generate_with_image(self, prompt, image_base64):
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            },
        }

        failures = []
        for index, model in enumerate(self._models):
            waits = PRIMARY_MODEL_RETRY_WAITS if index == 0 else FALLBACK_MODEL_RETRY_WAITS
            response = self._post_with_retry(model, payload, waits)

            if response is not None and response.status_code == 200:
                if index > 0:
                    print(f"[INFO] 予備モデル {model} で成功しました")
                return self._extract_text(response.json())

            if response is None:
                failures.append(f"{model}: 接続できませんでした")
            elif response.status_code in FALLBACK_STATUSES:
                failures.append(f"{model}: HTTP {response.status_code} {response.text[:200]}")
            else:
                # 400/401/403(キー不正など)はモデルを変えても直らないので、即エラーにする
                raise GeminiError(f"Gemini APIエラー(model={model}): {response.status_code} {response.text[:500]}")
            print(f"[WARN] {failures[-1][:150]} → 次のモデルを試します", file=sys.stderr)

        raise GeminiError("すべてのGeminiモデルで失敗しました: " + " | ".join(failures))

    def _post_with_retry(self, model, payload, waits):
        """一時的な不調なら待って再試行する。最後の応答を返す(通信自体に失敗し続けたら None)"""
        url = GEMINI_ENDPOINT.format(model=model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        response = None

        for attempt, wait in enumerate([*waits, None]):  # 最後の試行の後は待たない
            try:
                response = http_post(url, headers, payload, GEMINI_TIMEOUT_SECONDS)
                if response.status_code not in RETRYABLE_STATUSES:
                    return response
                reason = f"HTTP {response.status_code}"
            except (requests.ConnectionError, requests.Timeout) as error:
                response = None
                reason = f"通信エラー: {error}"

            if wait is None:
                break
            print(
                f"[WARN] {model} が一時的に使えません({reason})。"
                f"{wait}秒待って再試行します({attempt + 1}/{len(waits)})",
                file=sys.stderr,
            )
            time.sleep(wait)

        return response

    @staticmethod
    def _extract_text(response_json):
        try:
            parts = response_json["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as error:
            raise GeminiError(f"Geminiの応答解析に失敗しました: {response_json}") from error


def build_extraction_prompt(date_label):
    return f"""これはスーパーのチラシ画像です。本日の日付は {date_label} です。

## 抽出対象
「生肉」(牛肉・豚肉・鶏肉の精肉全般)の特売情報を抽出してください。
部位の種類は問いません。バラ肉・ロース肉・モモ肉・肩ロース肉はもちろん、
**ひき肉(挽肉)、むね肉、もも肉、ささみなど、見た目が地味で小さく掲載されている商品も見落とさないでください**。
チラシの端や、小さい文字で書かれた箇所も含めて、画像全体を隅々まで確認してください。

## 対象外
ハム・ソーセージ・唐揚げ・加工肉、魚介類、野菜、惣菜、その他の商品は対象外です。

## 日付の扱い(重要)
チラシ内の商品には「10/5限り」「10/4(日)・5(月)2日間」「10/1〜31まで」のように、
適用期間が商品ごと・コーナーごとに書かれていることがあります。
**本日({date_label})が、その適用期間に含まれる商品だけを抽出してください。**
期間の記載が無い商品は、チラシ全体の掲載期間が本日を含む場合のみ対象としてください。
本日が対象期間に含まれない商品は、たとえ目立つ場所にあっても抽出しないでください。

## 出力形式
説明文やMarkdownを含めず、次の形式の**JSON配列のみ**にしてください。
該当する生肉の特売情報が無ければ、空配列 [] を返してください。

[
  {{
    "product": "商品名(例: 国産豚バラ肉)",
    "price": "価格の記載(例: 198円(税込)/100g)",
    "discount": "割引率の記載があれば(例: 20%引き)。無ければ null"
  }}
]
"""


def parse_deals_json(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):  # 念のため、コードブロックで返ってきた場合に備える
        text = text.split("```")[1].removeprefix("json").strip()
    return [MeatDeal.from_dict(item) for item in json.loads(text)]


def extract_meat_deals(gemini, image_base64, date_label):
    raw_text = gemini.generate_with_image(build_extraction_prompt(date_label), image_base64)
    return parse_deals_json(raw_text)


# ---------------------------------------------------------------------------
# LINEのメッセージ組み立て(見た目だけを担当。通信はしない)
# ---------------------------------------------------------------------------


def _text(text, size="sm", color=COLOR_TEXT, **extra):
    return {"type": "text", "text": text, "size": size, "color": color, **extra}


def _box(contents, layout="vertical", **extra):
    return {"type": "box", "layout": layout, "contents": contents, **extra}


def _build_discount_badge(discount):
    # backgroundColor は box にしか指定できないため、text を box で包んでバッジにする
    return _box(
        [_text(discount, size="xxs", color=COLOR_WHITE, align="center")],
        backgroundColor=COLOR_ACCENT,
        cornerRadius="4px",
        paddingAll="2px",
        margin="sm",
    )


def _build_deal_row(deal):
    price_line = [_text(deal.price, size="xs", color=COLOR_ACCENT, weight="bold", flex=0)]
    if deal.discount:
        price_line.append(_build_discount_badge(deal.discount))

    return _box(
        [
            _text(deal.product, weight="bold", wrap=True),
            _box(price_line, layout="horizontal", margin="xs"),
        ],
        margin="md",
    )


def _build_header(date_label):
    return _box(
        [
            _text("🥩 本日の生肉チラシ情報", size="lg", color=COLOR_WHITE, weight="bold"),
            _text(f"{date_label}  {STORE_NAME}", size="xs", color=COLOR_HEADER_SUBTEXT, margin="sm"),
        ],
        backgroundColor=COLOR_HEADER_BG,
        paddingAll="20px",
    )


def _build_body_contents(deals):
    if not deals:
        return [_text(NO_DEALS_MESSAGE, color=COLOR_MUTED, wrap=True)]

    contents = []
    for index, deal in enumerate(deals):
        if index > 0:
            contents.append({"type": "separator", "margin": "md"})
        contents.append(_build_deal_row(deal))
    return contents


def build_flex_message(deals, date_label):
    return {
        "type": "bubble",
        "header": _build_header(date_label),
        "body": _box(_build_body_contents(deals), paddingAll="20px"),
    }


def format_deals_as_text(deals):
    """Flexカードを送れなかったときの、プレーンテキスト版"""
    if not deals:
        return NO_DEALS_MESSAGE
    return "\n".join(["本日の生肉特売情報:", *(deal.as_text() for deal in deals)])


# ---------------------------------------------------------------------------
# LINE(送信)
# ---------------------------------------------------------------------------


class LineClient:
    def __init__(self, channel_access_token):
        self._token = channel_access_token

    def send_flex(self, bubble, alt_text):
        self._broadcast({"type": "flex", "altText": alt_text, "contents": bubble})

    def send_text(self, text):
        self._broadcast({"type": "text", "text": text[:4900]})  # LINEの上限は5000文字

    def _broadcast(self, message):
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        response = http_post(LINE_BROADCAST_URL, headers, {"messages": [message]}, HTTP_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"LINE送信に失敗しました: {response.status_code} {response.text}")


# ---------------------------------------------------------------------------
# 全体の流れ
# ---------------------------------------------------------------------------


def analyze_leaflet(gemini, leaflet_id, date_label):
    print(f"チラシ {leaflet_id} を解析中...")
    image_url = fetch_leaflet_image_url(leaflet_id)
    if image_url is None:
        raise LookupError("画像URLが見つかりませんでした")
    return extract_meat_deals(gemini, fetch_image_base64(image_url), date_label)


def collect_deals(gemini, leaflet_ids, date_label):
    """全チラシを解析して特売情報を集める。1枚の失敗で全体を止めず、失敗数を数えておく"""
    deals, failed = [], 0
    for leaflet_id in leaflet_ids:
        try:
            deals.extend(analyze_leaflet(gemini, leaflet_id, date_label))
        except Exception as error:  # 1枚の失敗で他のチラシまで諦めない
            failed += 1
            print(f"[WARN] チラシ {leaflet_id} の解析に失敗: {error}", file=sys.stderr)
    return CollectionResult(deals=deals, leaflets_total=len(leaflet_ids), leaflets_failed=failed)


def notify(line, result, date_label):
    # 全滅のときに「特売なし」と送ると、本当に無かったのか失敗したのか区別がつかない
    if result.all_failed:
        line.send_text("本日のチラシ解析に失敗しました。GitHub Actionsのログを確認してください。")
        return

    try:
        line.send_flex(build_flex_message(result.deals, date_label), alt_text="本日の生肉チラシ情報")
    except Exception as error:
        print(f"[WARN] Flexメッセージ送信に失敗、テキストで代替送信します: {error}", file=sys.stderr)
        line.send_text(format_deals_as_text(result.deals))


def main():
    settings = load_settings()
    gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.fallback_models)
    line = LineClient(settings.line_token)
    date_label = today_label()
    print(f"本日: {date_label}")

    leaflet_ids = fetch_leaflet_ids()
    print(f"見つかったチラシ数: {len(leaflet_ids)} -> {leaflet_ids}")
    if not leaflet_ids:
        line.send_text(f"本日、{STORE_NAME}のチラシが見つかりませんでした。")
        return 0

    result = collect_deals(gemini, leaflet_ids, date_label)
    print(f"抽出された生肉特売情報: {len(result.deals)}件(解析失敗: {result.leaflets_failed}/{result.leaflets_total}枚)")
    notify(line, result, date_label)

    # 全滅のときはGitHub Actions上でも失敗(赤)にして、気づけるようにする
    return 1 if result.all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
