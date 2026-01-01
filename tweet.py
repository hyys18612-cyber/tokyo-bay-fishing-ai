import tweepy
import os
import datetime
import pandas as pd
import joblib
import requests
import numpy as np
import json
import time
from requests_oauthlib import OAuth1
from geopy.geocoders import Nominatim
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ==========================================
# 1. 認証情報の読み込み
# ==========================================
consumer_key = os.environ.get("TWITTER_API_KEY")
consumer_secret = os.environ.get("TWITTER_API_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# ==========================================
# 2. X API v2 手動アップロード関数 (強化版)
# ==========================================
def upload_media_v2(filename, consumer_key, consumer_secret, access_token, access_token_secret):
    """
    Tweepyを使わず、requestsで直接API v2のエンドポイントを叩いて画像をアップロードする関数
    (待機処理強化版)
    """
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
    file_size = os.path.getsize(filename)
    
    url = "https://upload.twitter.com/1.1/media/upload.json"
    
    # --- Step 1: INIT ---
    init_data = {
        "command": "INIT",
        "total_bytes": file_size,
        "media_type": "image/png",
        "media_category": "tweet_image"
    }
    
    print("📤 [v2] Upload Step 1: INIT")
    res_init = requests.post(url, data=init_data, auth=auth)
    
    if res_init.status_code not in [200, 202]:
        print(f"❌ INIT Failed: {res_init.text}")
        raise Exception(f"Media Upload INIT Failed: {res_init.status_code}")
        
    media_id = res_init.json()['media_id_string']
    print(f"   Media ID Issued: {media_id}")
    
    # --- Step 2: APPEND ---
    print(f"📤 [v2] Upload Step 2: APPEND")
    
    with open(filename, 'rb') as f:
        segment_id = 0
        while True:
            chunk = f.read(4 * 1024 * 1024) # 4MB chunk
            if not chunk:
                break
            
            append_data = {
                "command": "APPEND",
                "media_id": media_id,
                "segment_index": segment_id
            }
            files = {'media': chunk}
            
            res_append = requests.post(url, data=append_data, files=files, auth=auth)
            
            if res_append.status_code < 200 or res_append.status_code >= 300:
                print(f"❌ APPEND Failed: {res_append.text}")
                raise Exception("Media Upload APPEND Failed")
            
            segment_id += 1

    # --- Step 3: FINALIZE ---
    print("📤 [v2] Upload Step 3: FINALIZE")
    finalize_data = {
        "command": "FINALIZE",
        "media_id": media_id
    }
    
    res_fin = requests.post(url, data=finalize_data, auth=auth)
    
    if res_fin.status_code < 200 or res_fin.status_code >= 300:
        print(f"❌ FINALIZE Failed: {res_fin.text}")
        raise Exception("Media Upload FINALIZE Failed")
    
    # --- Step 4: STATUS CHECK (ここを強化) ---
    print(f"⏳ Waiting for media processing (ID: {media_id})...")
    
    # 念のため最初に強制待機
    time.sleep(5)
    
    check_url = "https://upload.twitter.com/1.1/media/upload.json"
    status_params = {
        "command": "STATUS",
        "media_id": media_id
    }
    
    # 最大10回 (約30秒) 確認するループ
    for i in range(10):
        res_status = requests.get(check_url, params=status_params, auth=auth)
        
        # ステータス情報が取得できない場合（即時完了）も考慮
        if res_status.status_code != 200:
             # エラーではなく、情報がないだけなら成功とみなして抜ける手もあるが、念のため待つ
             print(f"   Status check {i+1}: No info yet, waiting...")
             time.sleep(3)
             continue
             
        status_data = res_status.json()
        processing_info = status_data.get('processing_info', {})
        state = processing_info.get('state')
        
        print(f"   Status check {i+1}: {state if state else 'succeeded (completed)'}")
        
        if state == 'succeeded' or not state:
            print("✅ Media processing completed!")
            return media_id
        
        elif state == 'failed':
            error = processing_info.get('error', {})
            raise Exception(f"Media processing failed: {error}")
            
        elif state == 'in_progress' or state == 'pending':
            wait_secs = processing_info.get('check_after_secs', 2)
            time.sleep(wait_secs)
        else:
            time.sleep(3)
            
    # ループを抜けてもここに来る場合はタイムアウトだが、IDは返す
    print("⚠️ Warning: Status check timed out, but proceeding...")
    return media_id

# ==========================================
# 3. 設定エリア & ロジック
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

# --- ヘルパー関数 ---
def get_angler_comment(row_data, g_cpue_dict):
    wind = row_data['風速(m/s)']
    rain = row_data.get('降水量(mm)', 0)
    temp_diff = row_data.get('前日水温差', 0)
    total_cpue = row_data['★総釣果(CPUE)']
    if wind >= 8.0: return "⚠ 強風予報！安全第一で撤退も勇気"
    if rain >= 5.0: return "☔ 本降り予報。雨具必須、足元注意"
    if total_cpue >= 20.0: return "★爆釣警報！クーラー満タンの準備を"
    if g_cpue_dict.get('G2', 0) >= 0.5: return "大物チャンス！ルアー・泳がせで攻めろ"
    if g_cpue_dict.get('G1', 0) >= 8.0: return "◎ アジ・イワシ回遊！サビキで手堅く"
    if g_cpue_dict.get('G3', 0) >= 1.5: return "底物が熱い！投げ釣りでじっくり探れ"
    if temp_diff <= -0.5: return "水温低下中。活性低いなら深場・ボトムへ"
    if temp_diff >= 0.5: return "水温上昇！浅場の高活性な個体を狙え"
    if total_cpue >= 10.0: return "好条件！色々な魚種が狙える一日"
    if total_cpue <= 3.0: return "我慢の展開。潮の変わり目に集中しよう"
    return "エンジョイフィッシング！一発逆転を狙え"

def evaluate_cpue_total_scaled(val):
    if val >= 20.0: return "S (爆釣)"
    if val >= 10.0: return "A (好調)"
    if val >= 4.0: return "B (普通)"
    if val >= 1.2: return "C (渋い)"
    return "D (激渋)"

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

# --- 画像生成関数 ---
def generate_fishing_card(card_data_list, target_date_str):
    print("\n🎨 予報カード画像を生成中...")
    
    font_path = "ipaexg.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'IPAexGothic'
    else:
        plt.rcParams['font.family'] = 'sans-serif'
            
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('#f0f8ff')
    ax.set_facecolor('#f0f8ff')
    ax.axis('off')

    dt = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")
    date_display = dt.strftime("%Y/%m/%d (%a)")
    
    plt.text(0.5, 0.93, '東京湾 釣果予測AI', ha='center', va='center', fontsize=22, fontweight='bold', color='#003366')
    plt.text(0.5, 0.85, f'Target Date: {date_display}', ha='center', va='center', fontsize=13, color='#444444')

    y_positions = [0.65, 0.40, 0.15]
    colors = {'A (好調)': '#ffcccc', 'B (普通)': '#fff5cc', 'C (渋い)': '#e6f2ff', 'D (激渋)': '#f0f0f0', 'S (爆釣)': '#ff9999'}
    text_colors = {'A (好調)': '#cc0000', 'B (普通)': '#996600', 'C (渋い)': '#003399', 'D (激渋)': '#666666', 'S (爆釣)': '#cc0000'}

    for i, item in enumerate(card_data_list):
        if i >= 3: break
        y = y_positions[i]
        area_label = item['area_label']
        row_data = item['data']
        comment = item['ai_comment']
        
        rect = patches.FancyBboxPatch((0.05, y - 0.1), 0.9, 0.2, boxstyle="round,pad=0.02", linewidth=1, edgecolor='#cccccc', facecolor='white')
        ax.add_patch(rect)
        plt.text(0.1, y + 0.03, area_label, fontsize=16, fontweight='bold', color='#333333', va='center')
        judge = row_data['総合判定']
        bg_c = colors.get(judge, '#ffffff')
        txt_c = text_colors.get(judge, '#000000')
        v_rect = patches.FancyBboxPatch((0.55, y - 0.08), 0.35, 0.16, boxstyle="round,pad=0.02", linewidth=0, facecolor=bg_c)
        ax.add_patch(v_rect)
        judge_short = judge.split(' ')[0]
        judge_jp = judge.split(' ')[1].replace('(', '').replace(')', '')
        plt.text(0.725, y + 0.03, f"{judge_short} {judge_jp}", ha='center', va='center', fontsize=20, fontweight='bold', color=txt_c)
        details = f"天気: {row_data['天気']} | 風: {row_data['風速(m/s)']}m | 水温: {row_data['水温(℃)']}℃ | 総合CPUE: {row_data['★総釣果(CPUE)']}"
        plt.text(0.1, y - 0.05, details, fontsize=11, color='#555555', va='center')
        plt.text(0.725, y - 0.04, comment, ha='center', va='center', fontsize=11, fontweight='bold', color='#d9534f')

    plt.text(0.5, 0.02, 'Powered by Python & Fishing Forecast Model', ha='center', va='center', fontsize=10, color='#888888')
    plt.tight_layout()
    filename = 'fishing_forecast_card.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

# ==========================================
# 4. メイン処理
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
    
    card_data_list = []
    
    for place_name, display_name in TARGET_AREAS:
        print(f"\n🚀 {place_name} の予測開始...")
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
            fish_preds = {}
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
                        fish_preds[fish] = pred
                        g_sum += pred
                    g_cpue_sums[g_name] = g_sum
                    total_all_cpue += g_sum
            
            grade = evaluate_cpue_total_scaled(total_all_cpue)
            result_row = {
                "日付": TARGET_DATE_STR, "天気": w_label, 
                "風速(m/s)": round(row.get('wind_speed_10m_max', 0), 1),
                "水温(℃)": round(pw, 1), "前日水温差": round(pw - current['water_temp'], 1),
                "総合判定": grade, "★総釣果(CPUE)": round(total_all_cpue, 1)
            }
            comment = get_angler_comment(result_row, g_cpue_sums)
            card_data_list.append({"area_label": display_name, "data": result_row, "ai_comment": comment})

    if card_data_list:
        # 1. 画像生成
        image_file = generate_fishing_card(card_data_list, TARGET_DATE_STR)
        
        # 2. 手動アップロード (Tweepyを使わずv2エンドポイントを叩く)
        media_id = upload_media_v2(image_file, consumer_key, consumer_secret, access_token, access_token_secret)
        
        print(f"✅ Ready to tweet with Media ID: {media_id}")

        # 3. v2でツイート (media_idを添付)
        client = tweepy.Client(
            consumer_key=consumer_key, consumer_secret=consumer_secret,
            access_token=access_token, access_token_secret=access_token_secret
        )
        
        tweet_text = f"""📊 東京湾釣果予測 ({tomorrow.strftime('%m/%d')})

【釣行判断AI】
明日釣りに行こうか迷っている方へAIがアドバイス!!
画像で詳細をチェック👇

Web版: https://tokyo-bay-fishing-ai-ypd33onggtcjxnh69ryijz.streamlit.app/

#釣り #東京湾 #シーバス #アジング
"""
        client.create_tweet(text=tweet_text, media_ids=[media_id])
        print("✅ カード画像付きツイート成功！ (v2 Manual Upload)")
    else:
        print("❌ 予測データが生成されませんでした")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    # GitHub Actionsでエラーとして落とすため
    raise e
