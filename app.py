import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
from matplotlib import font_manager as fm
import datetime
import platform
import os
import shutil
import base64
import streamlit.components.v1 as components

# --- 追加ライブラリ（HTML注入用） ---
import pathlib
from bs4 import BeautifulSoup
import logging

# ロジックファイルからクラスをインポート
from logic import FishingPredictor, MAP_EXTENT, VISUAL_OFFSETS

# -------------------------------------------
# 画像ファイルの準備 & Base64変換
# -------------------------------------------
def get_img_as_base64(filename):
    """画像をBase64文字列に変換する"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# 画像ファイル名 (PNG形式)
target_image_name = "sea_view.png"
img_b64 = get_img_as_base64(target_image_name)

# -------------------------------------------
# 0. Analytics & Clarity 設定 (index.html注入方式)
# -------------------------------------------
def inject_ga_and_clarity():
    # ID設定
    GA_ID = "G-3L2NXKM7YT"
    CLARITY_ID = "uvovjbyie6"

    # 1. Google Analytics Code
    ga_js = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}');
    </script>
    """

    # 2. Microsoft Clarity Code
    clarity_js = f"""
    <script type="text/javascript">
        (function(c,l,a,r,i,t,y){{
            c[a]=c[a]||function(){{(c[a].q=c[a].q||[]).push(arguments)}};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        }})(window, document, "clarity", "script", "{CLARITY_ID}");
    </script>
    """

    # index.htmlのパスを取得
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    
    try:
        # htmlを読み込む
        soup = BeautifulSoup(index_path.read_text(), features="html.parser")
        
        # すでに挿入済みかチェック (重複防止)
        # ClarityのIDが含まれていなければ挿入する
        if CLARITY_ID not in str(soup):
            # headタグの先頭に挿入
            if soup.head:
                soup.head.insert(0, BeautifulSoup(ga_js + clarity_js, "html.parser"))
                index_path.write_text(str(soup))
                logging.info("Analytics & Clarity tags injected successfully.")
    except Exception as e:
        # ローカル環境など権限がない場合はエラーをログに出すだけにする
        logging.error(f"Analytics injection failed: {e}")

# -------------------------------------------
# 1. 日本語フォント設定
# -------------------------------------------
def setup_japanese_font():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, "ipaexg.ttf")
    
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'IPAexGothic'
        except Exception as e:
            st.error(f"フォントエラー: {e}")
    else:
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic']
        elif system == 'Darwin':
            plt.rcParams['font.family'] = ['Hiragino Sans', 'AppleGothic']

setup_japanese_font()

# -------------------------------------------
# 2. ページ設定 & 計測タグ注入
# -------------------------------------------
st.set_page_config(
    page_title="東京湾釣り予報AI",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ここでタグ注入を実行
inject_ga_and_clarity()

# カラー定義
PRIMARY_BLUE = "#0e4d92"
LIGHT_BLUE = "#2980b9"
BG_COLOR = "#F7F7F7"
CARD_BG = "#FFFFFF"

# CSSスタイルの作成
if img_b64:
    hero_style = f"""
        background: linear-gradient(rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.35)), 
                    url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-position: top;
    """
else:
    hero_style = f"background: linear-gradient(135deg, {PRIMARY_BLUE}, {LIGHT_BLUE});"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Noto Sans JP', sans-serif;
        background-color: {BG_COLOR};
        color: #222;
    }}
    
    section[data-testid="stSidebar"] {{ display: none; }}
    .block-container {{ padding-top: 1rem; }}

    /* --- ヒーローヘッダー --- */
    .hero-container {{
        width: 100%;
        height: 350px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        border-radius: 16px;
        margin-bottom: 35px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        {hero_style}
    }}

    @media (max-width: 768px) {{
        .hero-container {{ height: 220px; }}
        .main-title {{ font-size: 2.2rem !important; }}
    }}

    .main-title {{
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.2rem;
        text-shadow: 0 3px 15px rgba(0,0,0,0.8);
        letter-spacing: 0.05em;
    }}
    .sub-title {{
        font-size: 1.2rem;
        color: #f0f0f0;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0,0,0,0.7);
    }}

    /* --- UIコンポーネント --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px; justify-content: center; margin-bottom: 30px; border-bottom: none !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; background-color: transparent; border-radius: 30px;
        color: #717171; font-weight: 700; font-size: 1rem; padding: 0 25px; border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #E8F0FE !important; color: {PRIMARY_BLUE} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {PRIMARY_BLUE} !important;
    }}

    span[data-baseweb="tag"] {{
        background-color: #E8F0FE !important;
        color: {PRIMARY_BLUE} !important;
        border: 1px solid {PRIMARY_BLUE} !important;
    }}

    .stSelectbox label, .stDateInput label, .stMultiSelect label, .stSlider label {{
        font-size: 0.9rem !important; font-weight: 700 !important; color: #333 !important;
    }}

    div.stButton > button {{
        width: 100%; border-radius: 12px; font-weight: 800; font-size: 1.1rem; height: 3.5rem;
        background: linear-gradient(90deg, {PRIMARY_BLUE} 0%, {LIGHT_BLUE} 100%);
        color: white; border: none; margin-top: 28px;
        box-shadow: 0 4px 10px rgba(14, 77, 146, 0.2); transition: all 0.2s;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(90deg, {LIGHT_BLUE} 0%, {PRIMARY_BLUE} 100%);
        transform: scale(1.02); box-shadow: 0 6px 15px rgba(14, 77, 146, 0.3); color: white;
    }}

    .result-card {{
        background-color: {CARD_BG}; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;
        border: 1px solid #EBEBEB; transition: transform 0.2s;
    }}
    .result-card:hover {{
        transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }}
    .rank-badge {{
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        color: white; font-weight: bold; font-size: 0.9rem; margin-left: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }}
    .fish-tag {{
        display: inline-flex; align-items: center; background-color: #E8F0FE;
        color: {PRIMARY_BLUE}; padding: 4px 10px; border-radius: 8px;
        font-size: 0.8rem; font-weight: 600; margin-right: 6px; margin-top: 8px;
    }}
    .weather-box {{
        margin-top: 15px; background-color: #F7F7F7; border-radius: 12px;
        padding: 12px; display: flex; justify-content: space-around; align-items: center;
    }}
    .weather-item {{ display: flex; flex-direction: column; align-items: center; }}
    .weather-label {{ font-size: 0.75rem; color: #717171; margin-bottom: 2px; }}
    .weather-val {{ font-weight: bold; color: #222; font-size: 1rem; }}
    .stAlert {{ border-radius: 12px; }}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# 3. ロジック初期化 & ヘルパー
# -------------------------------------------
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
            html += f'<span class="fish-tag">{name} {score:.1f}</span>'
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
        colors = {'S':'#FF385C', 'A':'#FF9F1C', 'B':'#FFD93D', 'C':'#6FCF97', 'D':'#AAB7B8'}
        color = colors.get(item['rank'], 'gray')
        
        ax.scatter(x+0.003, y-0.003, s=size, c='black', alpha=0.1, zorder=9, edgecolors='none')
        ax.scatter(x, y, s=size, c=color, alpha=0.9, edgecolors='white', linewidth=2.5, zorder=10)
        
        label_txt = f"{item['name']}\n{cpue:.1f}匹"
        
        ax.text(x, y-0.015, label_txt, fontsize=12, fontweight='bold', ha='center', va='top', 
                 color='white', path_effects=[pe.withStroke(linewidth=3, foreground="#484848")], zorder=11)
    return fig

def plot_trend_chart(df, threshold=10.0):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle=':', color='#ccc', alpha=0.7)
    
    ax.plot(df['date_dt'], df['total_cpue'], marker='o', markersize=8, 
            linestyle='-', linewidth=3, color=PRIMARY_BLUE, label='CPUE (匹/人)')
    
    ax.axhline(y=threshold, color='#FF385C', linestyle='--', linewidth=1.5, alpha=0.8, label='Aランク')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.tick_params(axis='x', colors='#555', rotation=0)
    ax.tick_params(axis='y', colors='#555')
    
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    return fig

# -------------------------------------------
# 4. メインレイアウト
# -------------------------------------------

# ヒーローヘッダー
st.markdown(f"""
<div class="hero-container">
    <div class="main-title">TOKYO BAY FISHING AI 🐟</div>
    <div class="sub-title">AIによる気象・海況ビッグデータ分析</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 🔎 検索パネル
# ==========================================
tab_date, tab_place = st.tabs(["🤔 日程から探す", "📍 場所から探す"])

mode = None
execute_btn = False

# --- タブ1: 日程が決まっている場合 ---
with tab_date:
    with st.container():
        st.markdown("##### 📅 いつ、どこに行きますか？")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            target_date = st.date_input(
                "日程",
                datetime.date.today() + datetime.timedelta(days=1),
                help="予測したい日付を選択してください",
                key="date_input_tab1"
            )
        with col2:
            points_list = ["浦安", "若洲", "市原", "東扇島", "大黒", "磯子"]
            selected_points = st.multiselect(
                "候補エリア",
                points_list,
                default=points_list,
                placeholder="エリアを選択...",
                key="points_input_tab1"
            )
        with col3:
            if st.button("検索する", key="btn_date_search"):
                mode = "mode_date_fixed"
                execute_btn = True
                # 【操作ログ記録】
                print(f"[{datetime.datetime.now()}] ACTION: DateSearch | Date: {target_date} | Areas: {selected_points}")

# --- タブ2: 場所が決まっている場合 ---
with tab_place:
    with st.container():
        st.markdown("##### 🎣 どこで、ベストな日を探しますか？")
        col1, col2, col3, col4 = st.columns([1.2, 1, 1.2, 1])
        
        with col1:
            points_list = ["浦安", "若洲", "市原", "東扇島", "大黒", "磯子"]
            target_place = st.selectbox("場所", points_list, key="place_input_tab2")
            
        with col2:
            start_date = st.date_input(
                "開始日",
                datetime.date.today() + datetime.timedelta(days=1),
                key="date_input_tab2"
            )
            
        with col3:
            period = st.slider("期間 (向こう何日間)", 3, 14, 7, key="period_input_tab2")

        with col4:
            if st.button("ベスト日程を探す", key="btn_place_search"):
                mode = "mode_place_fixed"
                execute_btn = True
                # 【操作ログ記録】
                print(f"[{datetime.datetime.now()}] ACTION: PlaceSearch | Place: {target_place} | Start: {start_date} | Period: {period}")

st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# -------------------------------------------
# 5. 結果表示エリア
# -------------------------------------------

if execute_btn:
    today = datetime.date.today()
    limit_days = 14
    limit_date = today + datetime.timedelta(days=limit_days)
    
    is_date_error = False
    if mode == "mode_date_fixed":
        if target_date > limit_date: is_date_error = True
    else:
        if start_date > limit_date: is_date_error = True

    if is_date_error:
        st.error(
            f"⚠️ **予測可能な期間を超えています**\n\n"
            f"気象データAPIの制約により、現在 **{limit_date.strftime('%Y-%m-%d')}** までの日程しか予測できません。\n"
            "日付を範囲内に変更して再度お試しください。"
        )
        st.stop()

    # --- 用語解説 ---
    with st.expander("ℹ️ 数値の見方について（CPUEとは？）"):
        st.markdown("""
        表示されている数値は **CPUE (Catch Per Unit Effort)** です。
        これは **「釣り人1人あたりが1日に釣れる予想匹数」** を表しています。
        *例: 5.6匹/人 → 1人あたり約5〜6匹の釣果が見込まれます。*
        """)
        
    if mode == "mode_date_fixed":
        if not selected_points:
            st.warning("場所を少なくとも1つ選んでください")
        else:
            with st.spinner('AIが気象・海況データを解析中...'):
                results = predictor.run_prediction(target_date.strftime("%Y-%m-%d"), selected_points)
            
            if results:
                st.markdown(f"### 📅 {target_date.strftime('%Y/%m/%d')} の予測結果")
                c_map, c_list = st.columns([1.2, 1])
                
                with c_map:
                    st.caption("エリアポテンシャルマップ")
                    fig_map = plot_map(results, target_date)
                    st.pyplot(fig_map)
                
                with c_list:
                    st.caption("推奨ランキング")
                    df_res = pd.DataFrame(results).sort_values('total_cpue', ascending=False)
                    
                    for i, row in df_res.iterrows():
                        r_color = {'S':'#FF385C', 'A':'#FF9F1C', 'B':'#FFD93D', 'C':'#6FCF97', 'D':'#AAB7B8'}.get(row['rank'], '#999')
                        fish_html_content = get_top_fish_html(row.get('fish_breakdown', {}))
                        
                        card_html = f"""
                        <div class="result-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div style="display:flex; align-items:center;">
                                    <span style="font-size:1.2rem; font-weight:bold;">{row['name']}</span>
                                    <span class="rank-badge" style="background-color:{r_color};">{row['rank']}</span>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:0.75rem; color:#888; margin-bottom:-5px;">予想釣果(CPUE)</div>
                                    <span style="font-size:1.8rem; font-weight:900; color:{r_color};">{row['total_cpue']:.1f}</span>
                                    <span style="font-size:1.0rem; font-weight:bold; color:#666;">匹/人</span>
                                </div>
                            </div>
                            <div class="weather-box">
                                <div class="weather-item">
                                    <span class="weather-label">天気</span>
                                    <span class="weather-val">{row['weather']}</span>
                                </div>
                                <div class="weather-item">
                                    <span class="weather-label">風速</span>
                                    <span class="weather-val">{row['wind']:.1f}m</span>
                                </div>
                                <div class="weather-item">
                                    <span class="weather-label">気温</span>
                                    <span class="weather-val">{row['temp']:.1f}℃</span>
                                </div>
                            </div>
                            {fish_html_content}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

    elif mode == "mode_place_fixed":
        with st.spinner(f'{target_place} の向こう {period} 日間を解析中...'):
            period_results = predictor.run_period_analysis(
                target_place, 
                start_date.strftime("%Y-%m-%d"), 
                period
            )
        
        if period_results:
            df_period = pd.DataFrame(period_results)
            df_period['date_dt'] = pd.to_datetime(df_period['date'])
            df_period = df_period.sort_values('date_dt')
            
            st.markdown(f"### 📈 {target_place} の釣果予測推移")
            fig_chart = plot_trend_chart(df_period)
            st.pyplot(fig_chart)
            
            st.markdown("#### ✨ おすすめ日程 Top 3")
            best_days = df_period.sort_values('total_cpue', ascending=False).head(3)
            
            cols = st.columns(3)
            for i, (idx, row) in enumerate(best_days.iterrows()):
                r_color = {'S':'#FF385C', 'A':'#FF9F1C', 'B':'#FFD93D', 'C':'#6FCF97', 'D':'#AAB7B8'}.get(row['rank'], '#999')
                fish_html_content = get_top_fish_html(row.get('fish_breakdown', {}))
                display_date = row['date'][5:].replace('-', '/')

                with cols[i]:
                    day_card_html = f"""
                    <div class="result-card" style="text-align:center;">
                        <div style="font-size:1.3rem; font-weight:800; color:#333; margin-bottom:5px;">
                            {display_date}
                        </div>
                        <div style="margin-bottom:5px;">
                            <span style="font-size:0.8rem; color:#888;">予想釣果</span>
                            <br>
                            <span style="font-size:2.5rem; font-weight:900; color:{r_color}; line-height:1;">
                                {row['total_cpue']:.1f}
                            </span>
                            <span style="font-size:1rem; color:#666; font-weight:bold;">匹/人</span>
                        </div>
                        <div style="margin: 10px 0;">
                            <span class="rank-badge" style="background-color:{r_color}; margin:0;">{row['rank']}</span>
                        </div>
                        <div class="weather-box">
                            <div class="weather-item">
                                <span class="weather-label">天気</span>
                                <span class="weather-val">{row['weather']}</span>
                            </div>
                            <div class="weather-item">
                                <span class="weather-label">風速</span>
                                <span class="weather-val">{row['wind']:.1f}m</span>
                            </div>
                        </div>
                        {fish_html_content}
                    </div>
                    """
                    st.markdown(day_card_html, unsafe_allow_html=True)
            
            with st.expander("📋 データ一覧を表示"):
                # Warning修正: use_container_width=True -> width='stretch'
                st.dataframe(df_period[['date', 'rank', 'total_cpue', 'weather', 'wind', 'temp']], width='stretch')