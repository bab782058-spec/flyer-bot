"""
マルエツ 平間店(トクバイ掲載)のチラシから、本日有効な生肉の特売情報だけを
Geminiの画像認識で抽出し、LINEに送るスクリプト。

流れ:
    トクバイの店舗ページ → チラシID一覧 → チラシ画像 → Gemini(生肉だけ抽出) → LINE

環境変数(GitHub Actionsの Secrets / Variables で設定する):
    GEMINI_API_KEY                  Gemini APIキー(news_bot.pyと共通でOK)
    LINE_CHANNEL_ACCESS_TOKEN_MEAT  このbot専用のLINE公式アカウントのトークン
    GEMINI_MODELS                   (任意)試すモデルの順番(カンマ区切り)。未指定なら自動で選ぶ(gemini_client.py参照)

注意:
    個人利用・1日1回程度のアクセスを想定している。高頻度アクセスやデータの再配布はしないこと。
"""

import datetime
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import requests

from gemini_client import GeminiClient, GeminiError, encode_image, parse_json_response
from line_client import LineClient, flex_box, flex_text

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------

STORE_NAME = "マルエツ 平間店"
NO_DEALS_MESSAGE = "本日は生肉の特売情報が見つかりませんでした。"
RETRY_HINT_MESSAGE = "本日のチラシ解析に失敗しました(Geminiの混雑など)。\n「再実行」と送ると、もう一度試します。"
STORE_PAGE_URL = "https://tokubai.co.jp/%E3%83%9E%E3%83%AB%E3%82%A8%E3%83%84/13082"

HTTP_TIMEOUT_SECONDS = 30

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
COLOR_MUTED = "#666666"
COLOR_WHITE = "#FFFFFF"


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------


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
    return encode_image(http_get(image_url).content)


# ---------------------------------------------------------------------------
# Gemini(画像から生肉の特売を抽出。呼び出しの共通処理は gemini_client.py)
# ---------------------------------------------------------------------------


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
    return [MeatDeal.from_dict(item) for item in parse_json_response(raw_text)]


def extract_meat_deals(gemini, image_base64, date_label):
    raw_text = gemini.generate(build_extraction_prompt(date_label), image_base64=image_base64, json_mode=True)
    return parse_deals_json(raw_text)


# ---------------------------------------------------------------------------
# LINEのメッセージ組み立て(見た目だけを担当。通信はしない)
# ---------------------------------------------------------------------------


def _build_discount_badge(discount):
    # backgroundColor は box にしか指定できないため、text を box で包んでバッジにする
    return flex_box(
        [flex_text(discount, size="xxs", color=COLOR_WHITE, align="center")],
        backgroundColor=COLOR_ACCENT,
        cornerRadius="4px",
        paddingAll="2px",
        margin="sm",
    )


def _build_deal_row(deal):
    price_line = [flex_text(deal.price, size="xs", color=COLOR_ACCENT, weight="bold", flex=0)]
    if deal.discount:
        price_line.append(_build_discount_badge(deal.discount))

    return flex_box(
        [
            flex_text(deal.product, weight="bold", wrap=True),
            flex_box(price_line, layout="horizontal", margin="xs"),
        ],
        margin="md",
    )


def _build_header(date_label):
    return flex_box(
        [
            flex_text("🥩 本日の生肉チラシ情報", size="lg", color=COLOR_WHITE, weight="bold"),
            flex_text(f"{date_label}  {STORE_NAME}", size="xs", color=COLOR_HEADER_SUBTEXT, margin="sm"),
        ],
        backgroundColor=COLOR_HEADER_BG,
        paddingAll="20px",
    )


def _build_body_contents(deals):
    if not deals:
        return [flex_text(NO_DEALS_MESSAGE, color=COLOR_MUTED, wrap=True)]

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
        "body": flex_box(_build_body_contents(deals), paddingAll="20px"),
    }


def format_deals_as_text(deals):
    """Flexカードを送れなかったときの、プレーンテキスト版"""
    if not deals:
        return NO_DEALS_MESSAGE
    return "\n".join(["本日の生肉特売情報:", *(deal.as_text() for deal in deals)])


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
    for index, leaflet_id in enumerate(leaflet_ids):
        try:
            deals.extend(analyze_leaflet(gemini, leaflet_id, date_label))
        except GeminiError as error:
            # 全モデルが使えない状態。残りを試しても同じなので打ち切る(待ち時間の無駄を避ける)
            failed += len(leaflet_ids) - index
            print(f"[WARN] チラシ {leaflet_id} 以降の解析を打ち切ります: {error}", file=sys.stderr)
            break
        except Exception as error:  # 画像URLが取れないなど、1枚だけの失敗で他のチラシまで諦めない
            failed += 1
            print(f"[WARN] チラシ {leaflet_id} の解析に失敗: {error}", file=sys.stderr)
    return CollectionResult(deals=deals, leaflets_total=len(leaflet_ids), leaflets_failed=failed)


def notify(line, result, date_label):
    # 全滅のときに「特売なし」と送ると、本当に無かったのか失敗したのか区別がつかない
    if result.all_failed:
        line.send_text(RETRY_HINT_MESSAGE)
        return

    try:
        line.send_flex(build_flex_message(result.deals, date_label), alt_text="本日の生肉チラシ情報")
    except Exception as error:
        print(f"[WARN] Flexメッセージ送信に失敗、テキストで代替送信します: {error}", file=sys.stderr)
        line.send_text(format_deals_as_text(result.deals))


def main():
    gemini = GeminiClient.from_env()
    line = LineClient(os.environ["LINE_CHANNEL_ACCESS_TOKEN_MEAT"])
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
