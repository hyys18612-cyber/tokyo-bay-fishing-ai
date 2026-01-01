import tweepy
import os
import datetime
import pandas as pd
import joblib
import requests
import numpy as np
import warnings
from geopy.geocoders import Nominatim

warnings.filterwarnings('ignore')

# ==========================================
# 1. 認証情報の読み込み
# ==========================================
consumer_key = os.environ.get("TWITTER_API_KEY")
consumer_secret = os.environ.get("TWITTER_API_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# ==========================================
# 2. 設定エリア
# ==========================================
TARGET_AREAS = [
    ("浦安", "浦安（夢の島・若洲）"),
    ("大黒", "横浜（みなとみらい・大黒）"),
    ("市原", "千葉（千葉港/内房）")
]

STATIONS = {
    "kawasaki": {"name": "川崎人工島", "lat": 35.49028, "lon": 139.83389, "file": "kawasaki_environment.xlsx"},
    "1goto": {"name": "1号灯標", "lat": 35.53694, "lon": 139.95417, "file": "1goto_environment.xlsx"}
}

KNOWN_LOCATIONS = {
    "1号灯標": (35.5369, 139.9542), "川崎人工島": (35.4903, 139.8339),
    "川崎": (35.4903, 139.8339), "磯子": (35.4055, 139.6453),
    "本牧": (35.4285, 139.6873), "大黒": (35.4487, 139.6945),
    "市原": (35.5350, 140.0750), "浦安": (35.6400, 139.9417),
    "検見川沖": (35.6108, 140.0233)
}
CANDIDATE_FACILITIES = ["本牧", "大黒", "磯子", "市原"]

MODELS_CONFIG = {
    "G1": {"model": "fish_catch_model_G1.pkl", "encoder": "label_encoders_G1.pkl"},
    "G2": {"model": "fish_catch_model_G2.pkl", "encoder": "label_encoders_G2.pkl"},
    "G3": {"model": "fish_catch_model_G3.pkl", "encoder": "label_encoders_G3.pkl"},
    "G4": {"model": "fish_catch_model_G4.pkl", "encoder": "label_encoders_G4.pkl"},
    "water": "sub/water_temp_model.pkl", "turbidity": "sub/turbidity_model.pkl",
    "salt": "sub/salt_model.pkl", "do": "sub/do_model.pkl"
}

# ==========================================
# 3. ロジック関数群
# ==========================================
def get_short_reason(row_data, g_cpue_dict):
    """
    ツイート用に短くパンチのある理由を生成する
    """
    wind = row_data['風速(m/s)']
    rain = row_data.get('降水量(mm)', 0)
    total_cpue = row_data['★総釣果(CPUE)']
    
    # 1. ネガティブ要因（最優先）
    if wind >= 8.0: return "⚠️強風！安全第一で"
    if rain >= 5.0: return "☔雨天注意"
    if total_cpue <= 1.5: return "🙏修行の予感…"

    # 2. ポジティブ要因（魚種別）
    # G1: アジ・イワシ・サバ
    if g_cpue_dict.get('G1', 0) >= 8.0: return "🐟アジ・サバ爆釣!?"
    # G2: シーバス・タチウオ
    if g_cpue_dict.get('G2', 0) >= 0.5: return "🔥シーバス狙い目"
    # G3: カレイ・キス・カサゴ
    if g_cpue_dict.get('G3', 0) >= 1.5: return "🎣底物が熱い！"
    
    # 3. その他
    if total_cpue >= 10.0: return "✨全体的に高活性"
    
    return "🧐ワンチャンあるかも"

def evaluate_cpue_rank(val):
    if val >= 20.0: return "S"
    if val >= 10.0: return "A"
    if val >= 4.0: return "B"
    if val >= 1.2: return "C"
    return "D"

# --- 以下、共通ロジック (省略なし) ---
def match_features(model, available_data):
    try:
        if hasattr(model, 'feature_name_'): required_cols = model.feature_name_
        elif hasattr(model, 'feature_name'): required_cols = model.feature_name()
        else: required_cols = []
    except: required_cols = []
    
    if len(required_cols) == 0: return pd.DataFrame([available_data])
    input_data = {}
    for col in required_cols:
        val = available_data.get(col)
        if val is None:
            for k, v in available_data.items():
                if k in col or col in k: val = v; break
        input_data[col] = [val if val is not None else 0]
    return pd.DataFrame(input_data)

def get_coordinates(place_name):
    for key, val in KNOWN_LOCATIONS.items():
        if key in str(place_name): return val
    try:
        geolocator = Nominatim(user_agent="fishing_predictor_bot")
        loc = geolocator.geocode(place_name)
        if loc: return (loc.latitude, loc.longitude)
    except: pass
    return None

def calculate_moon_age(dt):
    base = datetime.datetime(2000, 1, 6, 12, 0)
    diff = dt - base
    return round((diff.total_seconds() / 86400) % 29.53058867, 1)

def get_weather_code_label(code):
    if code <= 1: return "晴れ"
    if code <= 48: return "曇り"
    return "雨"

def fetch_weather_forecast_range(lat, lon, start_dt, end_dt):
    fetch_start = start_dt - datetime.timedelta(days=5)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, 
        "start_date": fetch_start.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"), 
        "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum,pressure_msl_mean,weather_code",
        "timezone": "Asia/Tokyo", "wind_speed_unit": "ms"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if "daily" in data: return pd.DataFrame(data["daily"])
    except: return None
    return None

def find_best_substitute(target_weather_row, date_str, candidates_weather_cache):
    best_facility = "本牧"
    min_score = float('inf')
    t_wind = target_weather_row.get('wind_speed_10m_max', 0)
    t_temp = target_weather_row.get('temperature_2m_mean', 15)
    for facility in CANDIDATE_FACILITIES:
        df_cand = candidates_weather_cache.get(facility)
        if df_cand is None: continue
        cand_row = df_cand[df_cand['time'] == pd.to_datetime(date_str)]
        if len(cand_row) == 0: continue
        cand_row = cand_row.iloc[0]
        diff = abs(t_wind - cand_row['wind_speed_10m_max']) * 2.0 + abs(t_temp - cand_row['temperature_2m_mean'])
        if diff < min_score: min_score = diff; best_facility = facility
    return best_facility

def get_latest_marine_data(target_lat, target_lon):
    def calc_dist(lat1, lon1, lat2, lon2): return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    dk = calc_dist(target_lat, target_lon, STATIONS["kawasaki"]["lat"], STATIONS["kawasaki"]["lon"])
    d1 = calc_dist(target_lat, target_lon, STATIONS["1goto"]["lat"], STATIONS["1goto"]["lon"])
    st = STATIONS["kawasaki"] if dk < d1 else STATIONS["1goto"]
    
    if not os.path.exists(st['file']):
        return None, None
    try:
        df = pd.read_excel(st['file'])
        lr = df.iloc[-1]
        vals = {"water_temp": lr['水温(上層)(℃)'], "turbidity": lr['濁度(上層)(NTU)'], "do": lr['DO(上層)(mg/L)'], "salt": lr['塩分(上層)(-)']}
        return vals, pd.to_datetime(lr.iloc[0])
    except: return None, None

def safe_encode(encoder, val):
    try: return encoder.transform([val])[0]
    except: return 0 

# ==========================================
# 4. メイン処理 (テキスト生成 -> ツイート)
# ==========================================
try:
    print("📂 モデル読み込み中...")
    models = {}; encoders = {}; 
    for key, path in MODELS_CONFIG.items():
        if isinstance(path, dict):
            if os.path.exists(path["model"]):
                models[key] = joblib.load(path["model"])
                encoders[key] = joblib.load(path["encoder"])
        else:
            if os.path.exists(path): models[key] = joblib.load(path)

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    TARGET_DATE_STR = tomorrow.strftime("%Y-%m-%d")
    
    # 曜日の日本語表記
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    weekday_str = weekdays[tomorrow.weekday()]
    
    forecast_results = []
    
    for place_name, display_name in TARGET_AREAS:
        coords = get_coordinates(place_name)
        if not coords: continue

        current, last_marine_date = get_latest_marine_data(coords[0], coords[1])
        if current is None:
            current = {"water_temp": 12.0, "turbidity": 2.5, "salt": 31.5, "do": 9.5}
            last_marine_date = datetime.datetime.now() - datetime.timedelta(days=1)

        target_start_dt = pd.to_datetime(TARGET_DATE_STR)
        sim_start_dt = last_marine_date + datetime.timedelta(days=1)
        if target_start_dt < sim_start_dt: target_start_dt = sim_start_dt

        df_w = fetch_weather_forecast_range(coords[0], coords[1], sim_start_dt, target_start_dt)
        if df_w is None: continue
        
        c_cache = {}
        for f in CANDIDATE_FACILITIES:
            cc = get_coordinates(f)
            if cc:
                d = fetch_weather_forecast_range(cc[0], cc[1], sim_start_dt, target_start_dt)
                if d is not None: d['time'] = pd.to_datetime(d['time']); c_cache[f] = d

        df_w['time'] = pd.to_datetime(df_w['time'])
        target_row = df_w[df_w['time'].dt.strftime('%Y-%m-%d') == TARGET_DATE_STR]
        
        if not target_row.empty:
            row = target_row.iloc[0]
            date = row['time']
            w_label = get_weather_code_label(row['weather_code'])
            
            pool = {
                '気温': row['temperature_2m_mean'], '風速': row.get('wind_speed_10m_max', 0),
                '降水量': row.get('precipitation_sum', 0), '気圧': row.get('pressure_msl_mean', 1013),
                '日付': date.dayofyear, '日付(365)': date.dayofyear, '月齢': calculate_moon_age(date),
                '前日の水温': current['water_temp'], '水温': current['water_temp'],
                '前日の濁度': current['turbidity'], '濁度': current['turbidity'],
                '前日の塩分': current['salt'], '塩分': current['salt'], '前日のDO': current['do'], 'DO': current['do'],
                '平均気温': row['temperature_2m_mean'], '5日平均気温': row['temperature_2m_mean']
            }
            try:
                pw = models['water'].predict(match_features(models['water'], pool))[0] if 'water' in models else current['water_temp']
                pt = models['turbidity'].predict(match_features(models['turbidity'], pool))[0] if 'turbidity' in models else current['turbidity']
                ps = models['salt'].predict(match_features(models['salt'], pool))[0] if 'salt' in models else current['salt']
                pd_val = models['do'].predict(match_features(models['do'], pool))[0] if 'do' in models else current['do']
                pt = max(0.1, pt)
                pool.update({'予測水温': pw, '水温': pw, '前日との水温差': pw - current['water_temp'], '濁度': pt, '塩分': ps, 'DO': pd_val})
            except: pw, pt, ps, pd_val = current.values()

            sub_place = find_best_substitute(row, TARGET_DATE_STR, c_cache)
            g1_total = 0
            g_cpue_sums = {}
            total_all_cpue = 0
            
            for g_name in ["G1", "G2", "G3", "G4"]:
                if g_name in models:
                    m, e = models[g_name], encoders[g_name]
                    pool['施設名'] = safe_encode(e['施設名'], sub_place)
                    pool['天気'] = safe_encode(e['天気'], w_label)
                    g_sum = 0
                    for fish in e['魚種'].classes_:
                        pool['魚種'] = safe_encode(e['魚種'], fish)
                        pred = max(0, m.predict(match_features(m, pool))[0])
                        g_sum += pred
                    g_cpue_sums[g_name] = g_sum
                    total_all_cpue += g_sum
            
            # 結果格納
            rank = evaluate_cpue_rank(total_all_cpue)
            reason = get_short_reason({
                '風速(m/s)': row.get('wind_speed_10m_max', 0),
                '降水量(mm)': row.get('precipitation_sum', 0),
                '★総釣果(CPUE)': total_all_cpue
            }, g_cpue_sums)
            
            # 短い場所名を作る (浦安（夢の島...） -> 浦安)
            short_name = place_name 
            
            forecast_results.append({
                "name": short_name,
                "full_name": display_name,
                "rank": rank,
                "reason": reason,
                "cpue": total_all_cpue
            })

    # --- ツイート本文生成 ---
    if forecast_results:
        # CPUEが高い順に並び替え
        forecast_results.sort(key=lambda x: x['cpue'], reverse=True)
        best_spot = forecast_results[0]
        
        # 本文組み立て
        tweet_text = f"【釣行判断AI｜東京湾】\n\n"
        tweet_text += f"明日（{tomorrow.strftime('%m/%d')}・{weekday_str}）\n"
        tweet_text += f"釣りに行くか迷ってる人へ\n\n"
        
        for res in forecast_results:
            # ランクの絵文字
            rank_emoji = {'S':'🔥', 'A':'◎', 'B':'〇', 'C':'△', 'D':'☔'}.get(res['rank'], '・')
            tweet_text += f"📍{res['name']}\n"
            tweet_text += f"→ {res['rank']} ({res['reason']})\n\n"
        
        # 締めの言葉
        tweet_text += f"明日は{best_spot['name']}がおすすめ！🐟\n"
        tweet_text += f"👇詳細予報\n"
        tweet_text += f"https://tokyo-bay-fishing-ai-ypd33onggtcjxnh69ryijz.streamlit.app/"

        print("📝 生成されたツイート:")
        print(tweet_text)

        # v2でツイート
        client = tweepy.Client(
            consumer_key=consumer_key, consumer_secret=consumer_secret,
            access_token=access_token, access_token_secret=access_token_secret
        )
        client.create_tweet(text=tweet_text)
        print("✅ ツイート成功！")
        
    else:
        print("❌ 予測データが生成されませんでした")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    raise e
