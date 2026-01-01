import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
from matplotlib import font_manager as fm
import datetime
import platform
import os
import streamlit.components.v1 as components

# ロジックファイルからクラスをインポート
from logic import FishingPredictor, MAP_EXTENT, VISUAL_OFFSETS

# ==========================================
# 1. ページ設定 (※これを必ず最初に書く！)
# ==========================================
st.set_page_config(
    page_title="東京湾釣り予報AI",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. Google Analytics 設定 (デバッグ表示版)
# ==========================================
def inject_ga():
    GA_ID = "G-3L2NXKM7YT"
    
    # テスト用に背景を少し色付けし、読み込まれたら文字を表示する
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}');
        
        // 読み込み確認用メッセージ
        document.write('<div style="background:#e0f7fa; padding:10px; color:#006064; font-family:sans-serif; border-radius:5px;">✅ Google Analytics 起動完了 (ID: {GA_ID})</div>');
    </script>
    """
    
    # height=100 にして、画面にメッセージが見えるようにする
    components.html(ga_code, height=60)

# ここで実行
inject_ga()

# ==========================================
# 3. 日本語フォント設定
# ==========================================
def setup_japanese_font():
    font_path = "ipaexg.ttf"
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'IPAexGothic'
        except Exception as e:
            st.error(f"フォント読込エラー: {e}")
    else:
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic']
        elif system == 'Darwin':
            plt.rcParams['font.family'] = ['Hiragino Sans', 'AppleGothic']

setup_japanese_font()

# ==========================================
# 4. 以下、メインアプリの処理
# ==========================================

# カラーパレット定義
PRIMARY = "#0e4d92"
SECONDARY = "#f0f2f6"
ACCENT = "#ff6b6b"
WHITE = "#ffffff"

# カスタムCSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Noto Sans JP', sans-serif;
        background-color: {SECONDARY};
        color: #333;
    }}
    section[data-testid="stSidebar"] {{ display: none; }}
    
    .hero-container {{ text-align: center; padding: 2rem 0 1rem 0; }}
    .main-title {{ font-size: 2.2rem; font-weight: 900; color: {PRIMARY}; margin: 0; letter-spacing: 0.05em; }}
    .sub-title {{ font-size: 1.0rem; color: #666; margin-top: 0.5rem; }}
    
    .control-panel {{ background-color: {WHITE}; padding: 20px 25px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px; border: 1px solid #eef0f3; }}
    .streamlit-expanderHeader {{ font-weight: 700; color: {PRIMARY}; background-color: {WHITE}; border-radius: 8px; }}
    
    .result-card {{ background-color: {WHITE}; padding: 18px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); margin-bottom: 15px; border-left: 5px solid #ddd; transition: transform 0.2s ease; }}
    .result-card:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.08); }}
    
    .rank-S {{ border-left-color: #e74c3c !important; }}
    .rank-A {{ border-left-color: #e67e22 !important; }}
    .rank-B {{ border-left-color: #f1c40f !important; }}
    .rank-C {{ border-left-color: #2ecc71 !important; }}
    
    .rank-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.8rem; margin-left: 8px; vertical-align: middle; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    
    .fish-tag {{ display: inline-flex; align-items: center; background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-right: 5px; margin-top: 5px; }}
    .fish-score {{ background-color: #fff; color: #1565c0; padding: 0 5px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px; }}
    
    .weather-box {{ margin-top: 10px; background-color: #f8f9fa; border-radius: 8px; padding: 8px; font-size: 0.85rem; color: #555; display: flex; justify-content: space-around; align-items: center; }}
    .weather-item {{ display: flex; flex-direction: column; align-items: center; line-height: 1.2; }}
    .weather-label {{ font-size: 0.7rem; color: #888; margin-bottom: 2px; }}
    .weather-val {{ font-weight: bold; color: #333; }}
    
    div.stButton > button {{ width: 100%; border-radius: 10px; font-weight: 700; height: 3rem; background: linear-gradient(135deg, {PRIMARY} 0%, #003366 100%); color: white; border: none; box-shadow: 0 4px 10px rgba(0, 76, 153, 0.2); transition: all 0.3s; }}
    div.stButton > button:hover {{ box-shadow: 0 6px 15px rgba(0, 76, 153, 0.3); transform: scale(1.02); }}
    .stAlert {{ box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return FishingPredictor()

predictor = load_predictor()

def get_top_fish_html(fish_breakdown):
    if not fish_breakdown: return ""
    sorted_fish = sorted(fish_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
    html = '<div style="margin-top:8px;">'
    has_fish = False
    for name, score in sorted_fish:
        if score > 0.1:
            has_fish = True
            html += f'<span class="fish-tag">{name}<span class="fish-score">{score:.1f}</span></span>'
    html += '</div>'
    return html if has_fish else ""

def plot_map(data, date_str):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0) 
    try:
        img = mpimg.imread('tokyo_bay_map.png')
        ax.imshow(img, extent=MAP_EXTENT, aspect='auto', alpha=1.0, zorder=0)
    except:
        ax.set_facecolor('#d4e6f1')
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.axis('off')
    for item in data:
        x, y = item['lon'], item['lat']
        if item['name'] in VISUAL_OFFSETS:
            off = VISUAL_OFFSETS[item['name']]
            x += off['lon']; y += off['lat']
        cpue = item['total_cpue']
        size = 350 + (cpue * 45)
        colors = {'S':'#e74c3c', 'A':'#e67e22', 'B':'#f1c40f', 'C':'#2ecc71', 'D':'#95a5a6'}
        color = colors.get(item['rank'], 'gray')
        ax.scatter(x+0.003, y-0.003, s=size, c='black', alpha=0.15, zorder=9, edgecolors='none')
        ax.scatter(x, y, s=size, c=color, alpha=0.9, edgecolors='white', linewidth=2.5, zorder=10)
        label_txt = f"{item['name']}\n{cpue:.1f}"
        ax.text(x, y-0.015, label_txt, fontsize=12, fontweight='bold', ha='center', va='top', color='white', path_effects=[pe.withStroke(linewidth=3, foreground="#2c3e50")], zorder=11)
    return fig

def plot_trend_chart(df, threshold=10.0):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor(SECONDARY)
    ax.grid(True, linestyle=':', color='#ccc', alpha=0.7)
    ax.plot(df['date_dt'], df['total_cpue'], marker='o', markersize=8, linestyle='-', linewidth=3, color=PRIMARY, label='CPUE')
    ax.axhline(y=threshold, color=ACCENT, linestyle='--', linewidth=1.5, alpha=0.8, label='Aランク')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888')
    ax.spines['bottom'].set_color('#888')
    ax.tick_params(axis='x', colors='#555', rotation=0)
    ax.tick_params(axis='y', colors='#555')
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    return fig

# --- メインレイアウト ---
st.markdown("""
<div class="hero-container">
    <div class="main-title">TOKYO BAY FISHING AI 🐟</div>
    <div class="sub-title">AIによる気象・海況ビッグデータ分析</div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔎 検索条件を設定する", expanded=True):
    mode_mapping = { "🤔 日程は決まっている (どこに行く？)": "mode_date_fixed", "📅 行きたい場所がある (いつ行く？)": "mode_place_fixed" }
    selected_label = st.radio("目的を選んでください", list(mode_mapping.keys()), horizontal=True, label_visibility="collapsed")
    mode = mode_mapping[selected_label]
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    points = ["浦安", "若洲", "市原", "東扇島", "大黒", "磯子"]
    
    if mode == "mode_date_fixed":
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown("**📅 日程**")
            target_date = st.date_input("date_input", datetime.date.today() + datetime.timedelta(days=1), label_visibility="collapsed")
        with c2:
            st.markdown("**📍 候補エリア**")
            selected_points = st.multiselect("multi_select", points, default=points, label_visibility="collapsed")
        with c3:
            st.markdown("&nbsp;") 
            execute_btn = st.button("予測を実行 🚀", key="btn1")
    else:
        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c1:
            st.markdown("**📍 場所**")
            target_place = st.selectbox("place_select", points, label_visibility="collapsed")
        with c2:
            st.markdown("**📅 開始日 & 期間**")
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                start_date = st.date_input("start_date", datetime.date.today() + datetime.timedelta(days=1), label_visibility="collapsed")
            with col_in2:
                period = st.slider("days_slider", 3, 14, 7, label_visibility="collapsed")
        with c3:
            st.markdown("&nbsp;") 
            execute_btn = st.button("ベスト日程を探す 🔍", key="btn2")

if execute_btn:
    st.divider()
    today = datetime.date.today()
    limit_days = 14
    limit_date = today + datetime.timedelta(days=limit_days)
    is_date_error = False
    if mode == "mode_date_fixed":
        if target_date > limit_date: is_date_error = True
    else:
        if start_date > limit_date: is_date_error = True

    if is_date_error:
        st.error(f"⚠️ **予測可能な期間を超えています**\n\n気象データAPIの制約により、現在 **{limit_date.strftime('%Y-%m-%d')}** までの日程しか予測できません。\n日付を範囲内に変更して再度お試しください。")
        st.stop()
        
    if mode == "mode_date_fixed":
        if not selected_points:
            st.warning("場所を少なくとも1つ選んでください")
        else:
            with st.spinner('AIが気象・海況データを解析中...'):
                results = predictor.run_prediction(target_date.strftime("%Y-%m-%d"), selected_points)
            if results:
                c_map, c_list = st.columns([1.2, 1])
                with c_map:
                    st.subheader("🗺️ エリアポテンシャル")
                    fig_map = plot_map(results, target_date)
                    st.pyplot(fig_map)
                with c_list:
                    st.subheader("🏆 推奨ランキング")
                    df_res = pd.DataFrame(results).sort_values('total_cpue', ascending=False)
                    for i, row in df_res.iterrows():
                        r_color = {'S':'#e74c3c', 'A':'#e67e22', 'B':'#f1c40f', 'C':'#2ecc71', 'D':'#95a5a6'}.get(row['rank'], '#999')
                        fish_html = get_top_fish_html(row.get('fish_breakdown', {}))
                        st.markdown(f"""
<div class="result-card rank-{row['rank']}">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div><span style="font-size:1.1rem; font-weight:bold;">{row['name']}</span><span class="rank-badge" style="background-color:{r_color};">{row['rank']}</span></div>
        <div style="font-size:1.4rem; font-weight:900; color:{r_color};">{row['total_cpue']:.1f}</div>
    </div>
    <div class="weather-box" style="margin-top:10px;">
        <div class="weather-item"><span class="weather-label">天気</span><span class="weather-val">{row['weather']}</span></div>
        <div class="weather-item"><span class="weather-label">風速</span><span class="weather-val">{row['wind']:.1f}m</span></div>
        <div class="weather-item"><span class="weather-label">気温</span><span class="weather-val">{row['temp']:.1f}℃</span></div>
    </div>
    {fish_html}
</div>""", unsafe_allow_html=True)

    else:
        with st.spinner(f'{target_place} の向こう {period} 日間を解析中...'):
            period_results = predictor.run_period_analysis(target_place, start_date.strftime("%Y-%m-%d"), period)
        if period_results:
            df_period = pd.DataFrame(period_results)
            df_period['date_dt'] = pd.to_datetime(df_period['date'])
            df_period = df_period.sort_values('date_dt')
            st.subheader(f"📈 {target_place} の釣果予測推移")
            fig_chart = plot_trend_chart(df_period)
            st.pyplot(fig_chart)
            st.subheader("✨ おすすめ日程 Top 3")
            best_days = df_period.sort_values('total_cpue', ascending=False).head(3)
            cols = st.columns(3)
            for i, (idx, row) in enumerate(best_days.iterrows()):
                r_color = {'S':'#e74c3c', 'A':'#e67e22', 'B':'#f1c40f', 'C':'#2ecc71', 'D':'#95a5a6'}.get(row['rank'], '#999')
                fish_html = get_top_fish_html(row.get('fish_breakdown', {}))
                with cols[i]:
                    st.markdown(f"""
<div class="result-card rank-{row['rank']}" style="text-align:center;">
    <div style="font-size:1.5rem; font-weight:900; color:#333; margin-bottom:5px;">{row['date'][5:]}</div>
    <div style="font-size:2.2rem; font-weight:900; color:{r_color}; line-height:1;">{row['total_cpue']:.1f}</div>
    <span class="rank-badge" style="background-color:{r_color}; margin:5px 0 10px 0;">{row['rank']}</span>
    <div class="weather-box">
        <div class="weather-item"><span class="weather-label">天気</span><span class="weather-val">{row['weather']}</span></div>
        <div class="weather-item"><span class="weather-label">風速</span><span class="weather-val">{row['wind']:.1f}m</span></div>
        <div class="weather-item"><span class="weather-label">気温</span><span class="weather-val">{row['temp']:.1f}℃</span></div>
    </div>
    {fish_html}
</div>""", unsafe_allow_html=True)
            with st.expander("📋 データ一覧を表示"):
                st.dataframe(df_period[['date', 'rank', 'total_cpue', 'weather', 'wind', 'temp']], use_container_width=True)
