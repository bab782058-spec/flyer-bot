"""
マルエツ 平間店(トクバイ掲載)のチラシ画像を取得し、
生肉(牛肉・豚肉・鶏肉などの精肉。加工品は除く)の特売情報だけを
Geminiの画像認識で抽出して、LINEに送るスクリプト。

必要な環境変数(GitHub Actionsのsecretsで設定する):
- GEMINI_API_KEY: Gemini APIキー(news_bot.pyと共通でOK)
- LINE_CHANNEL_ACCESS_TOKEN_MEAT: このbot専用のLINE公式アカウントのチャネルアクセストークン
  (news_bot.pyとは別のLINE公式アカウント・別のトークンを使う)
- GEMINI_MODEL(任意): 未指定なら DEFAULT_GEMINI_MODEL
- GEMINI_FALLBACK_MODELS(任意): カンマ区切り。主モデルが使えないとき順に試す予備モデル

仕組み:
1. トクバイの店舗ページ(通常のHTTPリクエストで取得できる)から、
   現在掲載中のチラシのリンク(leaflet ID)を抽出
2. 各チラシの show_for_widget ページから、直接の画像URLを抽出
3. 画像をダウンロードし、Geminiに「生肉の特売情報だけ抽出して」と指示
   (news_bot.pyと同じ、主モデル→予備モデルの自動フォールバック構成)
4. 複数チラシ分の結果をまとめて、LINEにFlexメッセージで送信

注意:
- ヘッドレスブラウザは使わず、通常のHTTPリクエストのみで完結する構成
- 個人利用・1日1回程度のアクセスを想定。高頻度アクセスやデータの再配布は行わないこと
"""

import os
import re
import sys
import time
import json
import base64
import datetime
import requests

# 常に最新のFlashモデルを使うため、エイリアス gemini-flash-latest を既定にしている。
# Googleが新しいFlashを出すと、このエイリアスの指す先が自動で入れ替わる。
# 挙動を固定したいときは、GitHubの Variables に GEMINI_MODEL を登録して、
# gemini-3.6-flash のようにバージョンを明示する。
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
# 主モデルが混雑(503)・無料枠の上限(429)・提供終了(404)のときに順に試す予備モデル。
# 存在しないモデル名だった場合は404としてスキップされるだけで、実行は止まらない。
# GitHubの Variables に GEMINI_FALLBACK_MODELS(カンマ区切り)を登録すると差し替えられる。
DEFAULT_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

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

    # &(HTMLエスケープされた &quot; などの手前)でも止まるようにする
    match = re.search(
        r'https://image\.tokubai\.co\.jp/images/[^\s"\'<>()&]+\.jpg(?:\?[^\s"\'<>()&]*)?',
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


def _post_with_retry(model, headers, payload, waits):
    """1つのモデルに対し、一時的なエラー(混雑・上限・通信エラー)なら待って再試行する。
    最後の応答(通信自体に失敗した場合は None)を返す"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    retry_statuses = {429, 500, 502, 503, 504}
    res = None
    for attempt in range(len(waits) + 1):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=120)
            if res.status_code not in retry_statuses:
                return res
            reason = f"HTTP {res.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            res = None
            reason = f"通信エラー: {e}"
        if attempt < len(waits):
            print(
                f"[WARN] {model} が一時的に使えません({reason})。"
                f"{waits[attempt]}秒待って再試行します({attempt + 1}/{len(waits)})",
                file=sys.stderr,
            )
            time.sleep(waits[attempt])
    return res


def gemini_generate_with_image(prompt, image_base64):
    """Geminiに画像+プロンプトを渡し、出力テキストを返す。
    主モデルが混雑(503など)や提供終了(404)で使えないときは、予備モデルに自動で切り替える"""
    primary = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    fallbacks_env = os.environ.get("GEMINI_FALLBACK_MODELS")
    fallbacks = (
        [m.strip() for m in fallbacks_env.split(",") if m.strip()]
        if fallbacks_env is not None
        else DEFAULT_FALLBACK_MODELS
    )
    models = []
    for m in [primary] + fallbacks:
        if m not in models:
            models.append(m)

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                ]
            }
        ],
        # 商品数が多いチラシでも出力が途中で切れないよう、余裕を持たせる
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 4096,
        },
    }

    failures = []
    res = None
    for i, model in enumerate(models):
        # 主モデルは少し粘り、予備モデルは短めに待つ(全体が長引かないように)
        waits = [10, 20, 40] if i == 0 else [5, 10]
        res = _post_with_retry(model, headers, payload, waits)
        if res is not None and res.status_code == 200:
            if i > 0:
                print(f"[INFO] 予備モデル {model} で成功しました")
            break
        if res is None:
            failures.append(f"{model}: 接続できませんでした")
        elif res.status_code in (429, 500, 502, 503, 504, 404):
            # 混雑・上限・モデル提供終了は、次のモデルで試す価値がある
            failures.append(f"{model}: HTTP {res.status_code} {res.text[:200]}")
        else:
            # 400/401/403(キーが無効など)は、モデルを変えても直らないので即エラー
            raise RuntimeError(f"Gemini APIエラー(model={model}): {res.status_code} {res.text[:500]}")
        print(f"[WARN] {failures[-1][:150]} → 次のモデルを試します", file=sys.stderr)
    else:
        raise RuntimeError("すべてのGeminiモデルで失敗しました: " + " | ".join(failures))

    data = res.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Geminiの応答解析に失敗しました: {data}") from e


def extract_meat_deals_with_gemini(image_base64, target_date_str):
    """チラシ画像から、本日有効な生肉の特売情報だけをJSONで抽出する"""
    prompt = f"""これはスーパーのチラシ画像です。本日の日付は {target_date_str} です。

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
**本日({target_date_str})が、その適用期間に含まれる商品だけを抽出してください。**
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

    raw_text = gemini_generate_with_image(prompt, image_base64)

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        cleaned = cleaned.strip()

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
                                    # backgroundColorは「box」にしか指定できないため、
                                    # textをboxで包んでバッジ風に見せる
                                    "type": "box",
                                    "layout": "vertical",
                                    "backgroundColor": "#B23B3B",
                                    "cornerRadius": "4px",
                                    "paddingAll": "2px",
                                    "margin": "sm",
                                    "contents": [
                                        {
                                            "type": "text",
                                            "text": item.get("discount"),
                                            "size": "xxs",
                                            "color": "#FFFFFF",
                                            "align": "center",
                                        }
                                    ],
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
    today_str = build_date_header()
    print(f"本日: {today_str}")

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
            deals = extract_meat_deals_with_gemini(image_b64, today_str)
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
