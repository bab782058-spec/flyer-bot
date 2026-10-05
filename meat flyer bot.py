"""
マルエツ 平間店(トクバイ掲載)のチラシ画像を取得し、
生肉(牛肉・豚肉・鶏肉などの精肉。加工品は除く)の特売情報だけを
Geminiの画像認識で抽出して、LINEに送るスクリプト。

必要な環境変数(GitHub Actionsのsecretsで設定する):
- GEMINI_API_KEY: Gemini APIキー(news_bot.pyと共通でOK)
- LINE_CHANNEL_ACCESS_TOKEN_MEAT: このbot専用のLINE公式アカウントのチャネルアクセストークン
  (news_bot.pyとは別のLINE公式アカウント・別のトークンを使う)

仕組み:
1. トクバイの店舗ページ(通常のHTTPリクエストで取得できる)から、
   現在掲載中のチラシのリンク(leaflet ID)を抽出
2. 各チラシの show_for_widget ページから、直接の画像URLを抽出
3. 画像をダウンロードし、Geminiに「生肉の特売情報だけ抽出して」と指示
4. 複数チラシ分の結果をまとめて、LINEにFlexメッセージで送信

注意:
- ヘッドレスブラウザは使わず、通常のHTTPリクエストのみで完結する構成
- 個人利用・1日1回程度のアクセスを想定。高頻度アクセスやデータの再配布は行わないこと
"""

import os
import re
import sys
import base64
import datetime
import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN_MEAT"]

# マルエツ 平間店の店舗ページ(トクバイ)
STORE_PAGE_URL = "https://tokubai.co.jp/%E3%83%9E%E3%83%AB%E3%82%A8%E3%83%84/13082"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def get_leaflet_ids():
    """店舗ページから、現在掲載中のチラシID一覧を取得する(重複除去・順序保持)"""
    res = requests.get(STORE_PAGE_URL, headers=HEADERS, timeout=30)
    res.raise_for_status()
    html = res.text

    ids = re.findall(r"/leaflets/(\d+)", html)
    seen = []
    for leaflet_id in ids:
        if leaflet_id not in seen:
            seen.append(leaflet_id)
    return seen


def get_leaflet_image_url(leaflet_id):
    """チラシIDから、そのチラシの直接の画像URLを取得する"""
    url = f"{STORE_PAGE_URL}/leaflets/{leaflet_id}/show_for_widget?from=leaflet_widget"
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    html = res.text

    match = re.search(
        r'https://image\.tokubai\.co\.jp/images/[^\s"\'<>()]+\.jpg(?:\?[^\s"\'<>()]*)?',
        html,
    )
    if not match:
        return None
    # 念のため、末尾に紛れ込みうる記号を除去しておく
    return match.group(0).rstrip(").,、。")


def download_image_as_base64(image_url):
    res = requests.get(image_url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    return base64.b64encode(res.content).decode("utf-8")


def extract_meat_deals_with_gemini(image_base64):
    """チラシ画像から、生肉の特売情報だけをJSONで抽出する"""
    prompt = """これはスーパーのチラシ画像です。
この中から「生肉」(牛肉・豚肉・鶏肉・ひき肉などの精肉)の特売情報だけを抽出してください。
ハム・ソーセージ・唐揚げなどの加工品、魚介類、野菜、その他の商品は対象外です。

出力は、説明文やMarkdownを含めず、次の形式の**JSON配列のみ**にしてください。
該当する生肉の特売情報が無ければ、空配列 [] を返してください。

[
  {
    "product": "商品名(例: 国産豚バラ肉)",
    "price": "価格の記載(例: 198円(税込)/100g)",
    "discount": "割引率の記載があれば(例: 20%引き)。無ければ null"
  }
]
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64,
                        }
                    },
                ]
            }
        ]
    }

    res = requests.post(url, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()

    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Geminiの応答解析に失敗しました: {data}") from e

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned

    import json
    return json.loads(cleaned)


def build_date_header():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    weekday = WEEKDAY_JP[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日({weekday})"


def build_flex_message(meat_deals):
    date_str = build_date_header()

    if not meat_deals:
        body_contents = [
            {
                "type": "text",
                "text": "本日は生肉の特売情報が見つかりませんでした。",
                "size": "sm",
                "color": "#666666",
                "wrap": True,
            }
        ]
    else:
        body_contents = []
        for i, item in enumerate(meat_deals):
            row = {
                "type": "box",
                "layout": "vertical",
                "margin": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": item.get("product", ""),
                        "weight": "bold",
                        "size": "sm",
                        "wrap": True,
                        "color": "#1A1A1A",
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "margin": "xs",
                        "contents": [
                            {
                                "type": "text",
                                "text": item.get("price", ""),
                                "size": "xs",
                                "color": "#B23B3B",
                                "weight": "bold",
                                "flex": 0,
                            }
                        ]
                        + (
                            [
                                {
                                    "type": "text",
                                    "text": item.get("discount"),
                                    "size": "xxs",
                                    "color": "#FFFFFF",
                                    "backgroundColor": "#B23B3B",
                                    "align": "center",
                                    "margin": "sm",
                                }
                            ]
                            if item.get("discount")
                            else []
                        ),
                    },
                ],
            }
            body_contents.append(row)
            if i < len(meat_deals) - 1:
                body_contents.append({"type": "separator", "margin": "md"})

    bubble = {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#8B3A3A",
            "paddingAll": "20px",
            "contents": [
                {
                    "type": "text",
                    "text": "🥩 本日の生肉チラシ情報",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "lg",
                },
                {
                    "type": "text",
                    "text": f"{date_str}  マルエツ 平間店",
                    "color": "#F0D9D9",
                    "size": "xs",
                    "margin": "sm",
                },
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "20px",
            "contents": body_contents,
        },
    }
    return bubble


def send_line_flex(bubble, alt_text):
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"messages": [{"type": "flex", "altText": alt_text, "contents": bubble}]}
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"LINE送信に失敗しました: {res.status_code} {res.text}")


def send_line_text_fallback(text):
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"messages": [{"type": "text", "text": text[:4900]}]}
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"LINE送信に失敗しました: {res.status_code} {res.text}")


def main():
    print("チラシ一覧を取得中...")
    leaflet_ids = get_leaflet_ids()
    print(f"見つかったチラシ数: {len(leaflet_ids)} -> {leaflet_ids}")

    if not leaflet_ids:
        send_line_text_fallback("本日、マルエツ平間店のチラシが見つかりませんでした。")
        return

    all_deals = []
    for leaflet_id in leaflet_ids:
        print(f"チラシ {leaflet_id} の画像URLを取得中...")
        image_url = get_leaflet_image_url(leaflet_id)
        if not image_url:
            print(f"[WARN] チラシ {leaflet_id} の画像URLが見つかりませんでした", file=sys.stderr)
            continue

        print(f"チラシ {leaflet_id} の画像を解析中... ({image_url})")
        try:
            image_b64 = download_image_as_base64(image_url)
            deals = extract_meat_deals_with_gemini(image_b64)
            all_deals.extend(deals)
        except Exception as e:
            print(f"[WARN] チラシ {leaflet_id} の解析に失敗: {e}", file=sys.stderr)
            continue

    print(f"抽出された生肉特売情報: {len(all_deals)}件")

    try:
        bubble = build_flex_message(all_deals)
        send_line_flex(bubble, alt_text="本日の生肉チラシ情報")
    except Exception as e:
        print(f"[WARN] Flexメッセージ送信に失敗、テキストで代替送信します: {e}", file=sys.stderr)
        if all_deals:
            lines = ["本日の生肉特売情報:"]
            for item in all_deals:
                line = f"・{item.get('product','')} {item.get('price','')}"
                if item.get("discount"):
                    line += f"({item['discount']})"
                lines.append(line)
            send_line_text_fallback("\n".join(lines))
        else:
            send_line_text_fallback("本日は生肉の特売情報が見つかりませんでした。")

    print("完了しました。")


if __name__ == "__main__":
    main()
