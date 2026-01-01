import tweepy
import os
import datetime
from logic import FishingPredictor

# -------------------------------------------
# 1. 認証情報の読み込み (GitHub Secretsから)
# -------------------------------------------
consumer_key = os.environ.get("TWITTER_API_KEY")
consumer_secret = os.environ.get("TWITTER_API_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# -------------------------------------------
# 2. Twitter API v2 クライアントの準備
# -------------------------------------------
client = tweepy.Client(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

# -------------------------------------------
# 3. 予測の実行 (明日の分)
# -------------------------------------------
try:
    predictor = FishingPredictor()
    
    # 明日の日付
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    disp_date = tomorrow.strftime("%m/%d")
    
    points = ["浦安", "若洲", "市原", "東扇島", "大黒", "磯子"]
    
    # 予測実行
    results = predictor.run_prediction(date_str, points)
    
    # スコアが高い順に並び替え
    sorted_results = sorted(results, key=lambda x: x['total_cpue'], reverse=True)
    best_spot = sorted_results[0]
    
    # -------------------------------------------
    # 4. ツイート文の作成
    # -------------------------------------------
    rank_emoji = {'S': '🔥', 'A': '😍', 'B': '😀', 'C': '😐', 'D': '😭'}.get(best_spot['rank'], '🤔')
    
    tweet_text = f"""🤖 東京湾釣り予報AI

📅 {disp_date} のイチオシ！
📍 {best_spot['name']} ({best_spot['weather']})
📊 期待度: {best_spot['rank']} {rank_emoji}
🐟 指数: {best_spot['total_cpue']:.1f}

👇 詳細・他のエリアはこちら
https://tokyo-bay-fishing-ai-ypd33onggtcjxnh69ryijz.streamlit.app/

#釣り #東京湾 #シーバス #アジング
"""

    # -------------------------------------------
    # 5. 投稿実行
    # -------------------------------------------
    client.create_tweet(text=tweet_text)
    print("✅ ツイート成功！")
    print(tweet_text)

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    # エラー時もGithub Actionsを失敗扱いにしないための配慮（必要なら raise e に変更）