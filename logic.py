import joblib
import datetime
import requests
import pandas as pd
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 設定・定数定義
# ==========================================
MAP_EXTENT = [139.48, 140.32, 35.22, 35.74]

VISUAL_OFFSETS = {
    "市原": {"lon": 0.06, "lat": 0.04},
    "磯子": {"lon": -0.02, "lat": 0.0},
    "大黒": {"lon": -0.01, "lat": 0.01},
    "浦安": {"lon": 0.0, "lat": 0.0},
    "若洲": {"lon": 0.0, "lat": 0.0},
    "東扇島": {"lon": 0.0, "lat": 0.0}
}

STATIONS = {
    "kawasaki": {
        "name": "川崎人工島",
        "lat": 35.49028,
        "lon": 139.83389,
        "file": "kawasaki_environment.xlsx"
    },
    "1goto": {
        "name": "1号灯標",
        "lat": 35.53694,
        "lon": 139.95417,
        "file": "1goto_environment.xlsx"
    },
    "chibaharo": {
        "name": "千葉灯標",
        "lat": 35.61083,
        "lon": 140.02333,
        "file": "chibaharo_environment.xlsx"
    },
    "urayasu": {
        "name": "浦安",
        "lat": 35.64000,
        "lon": 139.94167,
        "file": "urayasu_environment.xlsx"
    }
}

KNOWN_LOCATIONS = {
    "1号灯標": (35.5369, 139.9542), "川崎人工島": (35.4903, 139.8339),
    "川崎": (35.4903, 139.8339), "磯子": (35.4055, 139.6453),
    "本牧": (35.4285, 139.6873), "大黒": (35.4487, 139.6945),
    "市原": (35.5350, 140.0750), "浦安": (35.6400, 139.9417),
    "若洲": (35.6175, 139.8385), "検見川沖": (35.6108, 140.0233),
    "東扇島": (35.4965, 139.7523)
}
CANDIDATE_FACILITIES = ["本牧", "大黒", "磯子", "市原"]

MODELS_CONFIG = {
    "G1": {"model": "fish_catch_model_G1.pkl", "encoder": "label_encoders_G1.pkl", "color": "#3498db", "label": "G1(Bait)"},
    "G2": {"model": "fish_catch_model_G2.pkl", "encoder": "label_encoders_G2.pkl", "color": "#e74c3c", "label": "G2(Predator)"},
    "G3": {"model": "fish_catch_model_G3.pkl", "encoder": "label_encoders_G3.pkl", "color": "#f39c12", "label": "G3(Bottom)"},
    "G4": {"model": "fish_catch_model_G4.pkl", "encoder": "label_encoders_G4.pkl", "color": "#2ecc71", "label": "G4(Other)"},
    "water": "sub/water_temp_model.pkl", "turbidity": "sub/turbidity_model.pkl",
    "salt": "sub/salt_model.pkl", "do": "sub/do_model.pkl"
}

class FishingPredictor:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.configs = {}
        self.load_models()

    def load_models(self):
        for key, path in MODELS_CONFIG.items():
            if isinstance(path, dict):
                if os.path.exists(path["model"]):
                    self.models[key] = joblib.load(path["model"])
                    self.encoders[key] = joblib.load(path["encoder"])
                    self.configs[key] = path
            else:
                if os.path.exists(path):
                    self.models[key] = joblib.load(path)

    def evaluate_cpue_total_scaled(self, val):
        if val >= 20.0: return "S"
        if val >= 10.0: return "A"
        if val >= 4.0: return "B"
        if val >= 1.2: return "C"
        return "D"

    def get_model_features(self, model):
        try:
            if hasattr(model, 'feature_name_'): return model.feature_name_
            elif hasattr(model, 'feature_name'): return model.feature_name()
        except: pass
        return []

    def match_features(self, model, available_data):
        required_cols = self.get_model_features(model)
        if len(required_cols) == 0: return pd.DataFrame([available_data])
        input_data = {}
        for col in required_cols:
            val = available_data.get(col)
            if val is None:
                for k, v in available_data.items():
                    if k in col or col in k: val = v; break
            input_data[col] = [val if val is not None else 0]
        return pd.DataFrame(input_data)

    def get_coordinates(self, place_name):
        for key, val in KNOWN_LOCATIONS.items():
            if key in str(place_name): return val
        return None 

    def calculate_moon_age(self, dt):
        base = datetime.datetime(2000, 1, 6, 12, 0)
        diff = dt - base
        return round((diff.total_seconds() / 86400) % 29.53058867, 1)

    def get_weather_code_label(self, code):
        if code <= 1: return "晴れ"
        if code <= 48: return "曇り"
        return "雨"

    def safe_encode(self, encoder, val):
        try: return encoder.transform([val])[0]
        except: return 0 

    @lru_cache(maxsize=128)
    def _fetch_weather_api(self, lat, lon, start_date_str, end_date_str):
        # ログ出力: APIリクエスト開始
        print(f"[API Request] Fetching {start_date_str} - {end_date_str} for ({lat}, {lon})")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon, 
            "start_date": start_date_str,
            "end_date": end_date_str, 
            "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum,pressure_msl_mean,weather_code",
            "timezone": "Asia/Tokyo", "wind_speed_unit": "ms"
        }
        try:
            res = requests.get(url, params=params, timeout=10) # タイムアウトを10秒に緩和
            
            if res.status_code != 200:
                print(f"[API Error] Status: {res.status_code} | Reason: {res.text}")
                return None
            
            data = res.json()
            if "daily" in data: 
                df = pd.DataFrame(data["daily"])
                df['time'] = pd.to_datetime(df['time'])
                return df
            else:
                print(f"[API Warning] No 'daily' key in response for ({lat}, {lon})")
                
        except Exception as e:
            print(f"[API Exception] {e}")
            return None
        return None

    def fetch_weather_forecast_range(self, lat, lon, start_dt, end_dt):
        fetch_start = start_dt - datetime.timedelta(days=5)
        return self._fetch_weather_api(
            lat, lon, 
            fetch_start.strftime("%Y-%m-%d"), 
            end_dt.strftime("%Y-%m-%d")
        )

    def find_best_substitute(self, target_weather_row, date_str, candidates_weather_cache):
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

    def get_latest_marine_data(self, target_lat, target_lon):
        def calc_dist(lat1, lon1, lat2, lon2): 
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
        
        best_station = None
        min_dist = float('inf')
        
        for key, station in STATIONS.items():
            dist = calc_dist(target_lat, target_lon, station["lat"], station["lon"])
            if dist < min_dist:
                min_dist = dist
                best_station = station
        
        st = best_station
        
        if st is None or not os.path.exists(st['file']):
            return None, None
            
        try:
            df = pd.read_excel(st['file'])
            if df.empty:
                return None, None

            df['temp_datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            df = df.dropna(subset=['temp_datetime'])
            if df.empty:
                return None, None

            last_full_dt = df['temp_datetime'].max()
            last_date = last_full_dt.date()

            day_mask = df['temp_datetime'].dt.date == last_date
            df_latest = df[day_mask]

            cols = {
                "water_temp": '水温(上層)(℃)',
                "turbidity": '濁度(上層)(NTU)',
                "do": 'DO(上層)(mg/L)',
                "salt": '塩分(上層)(-)'
            }
            
            vals = {}
            for key, col_name in cols.items():
                if col_name in df_latest.columns:
                    numeric_vals = pd.to_numeric(df_latest[col_name], errors='coerce')
                    vals[key] = numeric_vals.mean()
                else:
                    vals[key] = None

            return vals, last_full_dt
            
        except Exception:
            return None, None

    def prepare_weather_data_parallel(self, points, start_dt, end_dt):
        unique_locs = set(points) | set(CANDIDATE_FACILITIES)
        weather_cache = {}
        
        print(f"\n--- Starting Weather Fetch (Max Workers=1) for {len(unique_locs)} locations ---")
        
        # ------------------------------------------------------------------
        # 修正: サーバー環境でのAPIレート制限回避のため、max_workers=1 に設定
        # ------------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_loc = {}
            for loc_name in unique_locs:
                coords = self.get_coordinates(loc_name)
                if coords:
                    future_to_loc[executor.submit(
                        self.fetch_weather_forecast_range, 
                        coords[0], coords[1], start_dt, end_dt
                    )] = loc_name
                else:
                    print(f"[Coords Error] Unknown location: {loc_name}")

            for future in as_completed(future_to_loc):
                loc_name = future_to_loc[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        weather_cache[loc_name] = data
                        print(f"✅ [Success] {loc_name}")
                    else:
                        print(f"⚠️ [Failure] {loc_name}: API returned None or Empty")
                except Exception as e:
                    print(f"❌ [Error] {loc_name}: {e}")
        
        print("--- Weather Fetch Completed ---\n")
        return weather_cache

    def run_prediction(self, target_date_str, target_points):
        analysis_data = []
        target_dt = pd.to_datetime(target_date_str)
        
        ref_coords = self.get_coordinates("川崎") 
        _, last_marine_date = self.get_latest_marine_data(ref_coords[0], ref_coords[1])
        if last_marine_date is None: 
            last_marine_date = datetime.datetime.now() - datetime.timedelta(days=1)
        
        sim_start_dt = last_marine_date + datetime.timedelta(days=1)
        if target_dt < sim_start_dt:
            sim_start_dt = target_dt - datetime.timedelta(days=5)

        global_weather_cache = self.prepare_weather_data_parallel(target_points, sim_start_dt, target_dt)
        
        for place_name in target_points:
            coords = self.get_coordinates(place_name)
            if not coords: continue

            current, _ = self.get_latest_marine_data(coords[0], coords[1])
            
            if current is None or current.get("water_temp") is None:
                current = {"water_temp": 12.0, "turbidity": 2.5, "salt": 31.5, "do": 9.5}
            if pd.isna(current["water_temp"]): current["water_temp"] = 12.0
            if pd.isna(current["turbidity"]): current["turbidity"] = 2.5
            if pd.isna(current["salt"]): current["salt"] = 31.5
            if pd.isna(current["do"]): current["do"] = 9.5

            df_w = global_weather_cache.get(place_name)
            
            # --- 修正: データがない場合のログ出力 ---
            if df_w is None: 
                print(f"❌ Skipping {place_name}: Weather data not found in cache.")
                continue
            
            c_cache = {k: v for k, v in global_weather_cache.items() if k in CANDIDATE_FACILITIES}

            final_result = None
            for i, row in df_w.iterrows():
                date = row['time']
                d_str = date.strftime('%Y-%m-%d')
                w_label = self.get_weather_code_label(row['weather_code'])
                
                pool = {
                    '気温': row['temperature_2m_mean'], '風速': row.get('wind_speed_10m_max', 0),
                    '降水量': row.get('precipitation_sum', 0), '気圧': row.get('pressure_msl_mean', 1013),
                    '日付': date.dayofyear, '日付(365)': date.dayofyear, '月齢': self.calculate_moon_age(date),
                    '前日の水温': current['water_temp'], '水温': current['water_temp'],
                    '前日の濁度': current['turbidity'], '濁度': current['turbidity'],
                    '前日の塩分': current['salt'], '塩分': current['salt'], '前日のDO': current['do'], 'DO': current['do'],
                    '平均気温': row['temperature_2m_mean'], '5日平均気温': row['temperature_2m_mean']
                }
                
                try:
                    m = self.models
                    pw = m['water'].predict(self.match_features(m['water'], pool))[0] if 'water' in m else current['water_temp']
                    pt = m['turbidity'].predict(self.match_features(m['turbidity'], pool))[0] if 'turbidity' in m else current['turbidity']
                    ps = m['salt'].predict(self.match_features(m['salt'], pool))[0] if 'salt' in m else current['salt']
                    pd_val = m['do'].predict(self.match_features(m['do'], pool))[0] if 'do' in m else current['do']
                    pt = max(0.1, pt)
                    pool.update({'予測水温': pw, '水温': pw, '前日との水温差': pw - current['water_temp'], '濁度': pt, '塩分': ps, 'DO': pd_val})
                except: pw, pt, ps, pd_val = current.values()

                if d_str == target_date_str:
                    sub_place = self.find_best_substitute(row, d_str, c_cache)
                    fish_breakdown = {}
                    fish_group_map = {}
                    total_cpue = 0
                    
                    g1_sum = 0
                    if "G1" in self.models:
                        m, e = self.models["G1"], self.encoders["G1"]
                        pool['施設名'] = self.safe_encode(e['施設名'], sub_place)
                        pool['天気'] = self.safe_encode(e['天気'], w_label)
                        for fish in e['魚種'].classes_:
                            pool['魚種'] = self.safe_encode(e['魚種'], fish)
                            pred = max(0, m.predict(self.match_features(m, pool))[0])
                            fish_breakdown[fish] = pred
                            fish_group_map[fish] = self.configs["G1"]["color"]
                            g1_sum += pred
                    
                    pool['G1CPUE'] = g1_sum
                    pool['Group1_Total_CPUE'] = g1_sum
                    total_cpue = g1_sum
                    
                    for g_name in ["G2", "G3", "G4"]:
                        if g_name in self.models:
                            m, e = self.models[g_name], self.encoders[g_name]
                            pool['施設名'] = self.safe_encode(e['施設名'], sub_place)
                            pool['天気'] = self.safe_encode(e['天気'], w_label)
                            g_group_sum = 0
                            for fish in e['魚種'].classes_:
                                pool['魚種'] = self.safe_encode(e['魚種'], fish)
                                pred = max(0, m.predict(self.match_features(m, pool))[0])
                                fish_breakdown[fish] = pred
                                fish_group_map[fish] = self.configs[g_name]["color"]
                                g_group_sum += pred
                            total_cpue += g_group_sum
                    
                    final_result = {
                        "name": place_name, "lat": coords[0], "lon": coords[1],
                        "total_cpue": total_cpue,
                        "fish_breakdown": fish_breakdown,
                        "fish_group_map": fish_group_map,
                        "weather": w_label, "temp": pw, "wind": row.get('wind_speed_10m_max', 0),
                        "rank": self.evaluate_cpue_total_scaled(total_cpue)
                    }
                current = {"water_temp": pw, "turbidity": pt, "salt": ps, "do": pd_val}
            
            if final_result: analysis_data.append(final_result)
            
        return analysis_data

    def run_period_analysis(self, place_name, start_date_str, days=7):
        results = []
        start_dt = pd.to_datetime(start_date_str)
        end_dt = start_dt + datetime.timedelta(days=days)
        ref_coords = self.get_coordinates(place_name)
        if not ref_coords: return []
        self.fetch_weather_forecast_range(ref_coords[0], ref_coords[1], start_dt, end_dt)
        for cand in CANDIDATE_FACILITIES:
            cc = self.get_coordinates(cand)
            if cc: self.fetch_weather_forecast_range(cc[0], cc[1], start_dt, end_dt)
        for i in range(days):
            target_dt = start_dt + datetime.timedelta(days=i)
            d_str = target_dt.strftime('%Y-%m-%d')
            data = self.run_prediction(d_str, [place_name])
            if data:
                res = data[0]
                res['date'] = d_str
                results.append(res)
        return results