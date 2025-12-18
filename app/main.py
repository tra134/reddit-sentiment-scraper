# app/main.py - PHIÊN BẢN ĐÃ SỬA LOGIC HOÀN CHỈNH
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
import time
import sys
import os
import sqlite3
import hashlib
import feedparser
import json
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import tempfile

# --- THÊM IMPORT CHO NLTK ---
import nltk
import random
import string
from threading import Lock

# Cấu hình NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('punkt_tab')  # Gói mới cho NLTK bản gần đây

from textblob import TextBlob
from collections import Counter

# --- 1. SETUP ENVIRONMENT ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI Module
try:
    from visualizations import ui
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    # Tạo UI functions cơ bản nếu không tìm thấy module
    class SimpleUI:
        @staticmethod
        def load_css():
            st.markdown("""
            <style>
            .stButton>button { width: 100%; }
            .feature-card { padding: 1rem; border-radius: 10px; background: #f8f9fa; }
            .metric-card { padding: 1rem; border-radius: 10px; border: 1px solid #e0e0e0; }
            </style>
            """, unsafe_allow_html=True)
        
        @staticmethod
        def render_login_screen():
            st.markdown("# 🔐 Reddit Insider AI")
        
        @staticmethod
        def render_dashboard_header(username):
            st.markdown(f"# 👋 Chào {username}")
        
        @staticmethod
        def render_feature_card(icon, title, desc, btn_class, btn_text, callback):
            with st.container():
                st.markdown(f"### {icon} {title}")
                st.markdown(desc)
                if st.button(btn_text, key=btn_class):
                    callback()
        
        @staticmethod
        def render_history_list(history, delete_func):
            for item in history:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{item['title'][:50]}...**")
                    st.caption(f"📅 {item['timestamp']}")
                with col2:
                    if st.button("🗑️", key=f"del_{item['id']}"):
                        delete_func(item['id'])
                        st.rerun()
        
        @staticmethod
        def render_trending_card(post, callback=None):
            """Hiển thị card bài viết trending - FIXED VERSION"""
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 📝 {post.get('title', 'No Title')[:70]}...")
                    st.caption(f"r/{post.get('subreddit', 'unknown')} • 👤 {post.get('author', 'unknown')}")
                    st.caption(f"👍 {post.get('score', 0)} • 💬 {post.get('comments_count', 0)} • 🕐 {post.get('time_str', '')}")
                
                with col2:
                    if st.button("🔍 Phân tích", key=f"analyze_{post.get('id', '')}"):
                        if callback:
                            callback(post.get('url', ''))
                        else:
                            st.session_state.analyze_url = post.get('url', '')
                            st.session_state.auto_run = True
                            st.session_state.page = "Analysis"
                            st.rerun()
        
        @staticmethod
        def render_trend_analysis(analysis_result):
            """Hiển thị kết quả phân tích trend với đồ thị đầy đủ"""
            if not analysis_result:
                st.warning("Không có dữ liệu phân tích")
                return
            
            st.markdown("### 📈 Phân tích xu hướng chi tiết")
            
            # ========== PHẦN 1: THỐNG KÊ TỔNG QUAN ==========
            st.markdown("#### 📊 Thống kê tổng quan")
            summary = analysis_result.get('data_summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Bài viết", summary.get('total_posts_analyzed', 0))
            with col2:
                st.metric("⭐ Điểm TB", f"{summary.get('avg_score_per_post', 0):.1f}")
            with col3:
                st.metric("💬 Bình luận TB", f"{summary.get('avg_comments_per_post', 0):.1f}")
            with col4:
                st.metric("🚀 Engagement TB", f"{summary.get('avg_engagement_per_post', 0):.1f}")
            
            # ========== PHẦN 2: ĐỒ THỊ DỰ BÁO ==========
            forecast = analysis_result.get('forecast', {})
            if forecast and 'forecast' in forecast:
                forecast_data = forecast.get('forecast', [])
                if forecast_data:
                    st.markdown("#### 📈 Dự báo Engagement")
                    
                    # Tạo DataFrame cho đồ thị
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    # Tạo đồ thị với Plotly
                    fig = go.Figure()
                    
                    # Thêm đường dự báo chính
                    fig.add_trace(go.Scatter(
                        x=df_forecast['date'],
                        y=df_forecast['predicted_engagement'],
                        mode='lines+markers',
                        name='Dự báo',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Thêm vùng confidence interval
                    fig.add_trace(go.Scatter(
                        x=df_forecast['date'].tolist() + df_forecast['date'].tolist()[::-1],
                        y=df_forecast['predicted_upper'].tolist() + df_forecast['predicted_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Khoảng tin cậy',
                        showlegend=True
                    ))
                    
                    # Cấu hình layout
                    fig.update_layout(
                        title='Dự báo Engagement 7 ngày tới',
                        xaxis_title='Ngày',
                        yaxis_title='Engagement',
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Hiển thị xu hướng
                    if 'trend_direction' in forecast:
                        trend_emoji = {
                            'Tăng mạnh 🚀': '🚀',
                            'Tăng nhẹ ↗️': '↗️',
                            'Giảm mạnh 📉': '📉',
                            'Giảm nhẹ ↘️': '↘️',
                            'Ổn định ➡️': '➡️',
                            'Đang phân tích 📊': '📊'
                        }.get(forecast['trend_direction'], '📊')
                        
                        st.info(f"**Xu hướng:** {forecast['trend_direction']} {trend_emoji}")
            
            # ========== PHẦN 3: ĐỒ THỊ GIỜ CAO ĐIỂM ==========
            peak_hours = analysis_result.get('peak_hours', [])
            if peak_hours:
                st.markdown("#### 🕒 Giờ cao điểm đăng bài")
                
                peak_df = pd.DataFrame(peak_hours)
                peak_df['hour_str'] = peak_df['hour'].apply(lambda x: f"{x:02d}:00")
                
                # Sắp xếp theo giờ
                peak_df = peak_df.sort_values('hour')
                
                # Tạo đồ thị cột
                fig2 = px.bar(
                    peak_df,
                    x='hour_str',
                    y='total_engagement',
                    title='Engagement theo giờ trong ngày',
                    labels={'hour_str': 'Giờ', 'total_engagement': 'Engagement'},
                    color='total_engagement',
                    color_continuous_scale='Viridis'
                )
                
                fig2.update_layout(
                    xaxis_title='Giờ',
                    yaxis_title='Tổng Engagement',
                    height=350,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Hiển thị top 3 giờ cao điểm
                if len(peak_hours) >= 3:
                    top_hours = sorted(peak_hours, key=lambda x: x['total_engagement'], reverse=True)[:3]
                    st.markdown("**⏰ Top 3 giờ cao điểm:**")
                    for i, hour_data in enumerate(top_hours, 1):
                        hour = hour_data['hour']
                        engagement = hour_data['total_engagement']
                        posts = hour_data['post_count']
                        st.markdown(f"{i}. **{hour:02d}:00** - {engagement} engagement ({posts} bài)")
            
            # ========== PHẦN 4: TỪ KHÓA PHỔ BIẾN ==========
            keywords = analysis_result.get('top_keywords', [])
            if keywords:
                st.markdown("#### 🔑 Từ khóa phổ biến")
                
                # Tạo word cloud đơn giản
                keywords_df = pd.DataFrame(keywords[:10])  # Lấy top 10
                
                if not keywords_df.empty:
                    # Tạo đồ thị thanh cho từ khóa
                    fig3 = px.bar(
                        keywords_df,
                        x='keyword',
                        y='score',
                        title='Top từ khóa',
                        labels={'keyword': 'Từ khóa', 'score': 'Độ phổ biến'},
                        color='score',
                        color_continuous_scale='thermal'
                    )
                    
                    fig3.update_layout(
                        xaxis_title='Từ khóa',
                        yaxis_title='Độ phổ biến',
                        height=350,
                        template='plotly_white',
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Hiển thị danh sách từ khóa
                    keyword_list = " | ".join([f"**{k['keyword']}**" for k in keywords[:8]])
                    st.markdown(f"📌 *Các từ khóa hàng đầu:* {keyword_list}")
            
            # ========== PHẦN 5: THÔNG TIN BỔ SUNG ==========
            st.markdown("---")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if 'analysis_timestamp' in analysis_result:
                    ts = datetime.fromisoformat(analysis_result['analysis_timestamp'])
                    st.metric("🕐 Thời gian phân tích", ts.strftime('%H:%M %d/%m/%Y'))
            
            with col_info2:
                if 'subreddit' in analysis_result:
                    st.metric("👥 Cộng đồng", f"r/{analysis_result['subreddit']}")
            
            if 'note' in analysis_result:
                st.success(f"💡 {analysis_result['note']}")
        
        @staticmethod
        def render_sidebar_logged_in(username, groups, logout_callback, add_group_callback, delete_group_callback):
            with st.sidebar:
                st.markdown(f"### 👤 {username}")
                st.divider()
                
                # Navigation
                st.markdown("### 🧭 Điều hướng")
                pages = {
                    "📊 Dashboard": "Dashboard",
                    "📈 Xu hướng": "Trending",
                    "🔗 Phân tích bài viết": "Analysis"
                }
                
                for icon_name, page_name in pages.items():
                    if st.button(icon_name, use_container_width=True, key=f"nav_{page_name}"):
                        st.session_state.page = page_name
                        st.rerun()
                
                st.divider()
                
                # Groups management
                st.markdown("### 👥 Nhóm theo dõi")
                if not groups:
                    st.info("Chưa có nhóm nào")
                else:
                    for group in groups:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"r/{group['subreddit']}")
                        with col2:
                            if st.button("🗑️", key=f"del_group_{group['id']}"):
                                delete_group_callback(group['id'])
                                st.rerun()
                
                # Add group
                with st.form("add_group"):
                    new_group = st.text_input("Thêm subreddit", placeholder="python")
                    if st.form_submit_button("➕ Thêm", use_container_width=True):
                        if new_group:
                            add_group_callback(new_group)
                            st.rerun()
                
                st.divider()
                
                # Logout
                if st.button("🚪 Đăng xuất", type="secondary", use_container_width=True):
                    logout_callback()
    
    ui = SimpleUI()

# Config API
try:
    GOOGLE_GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_GEMINI_API_KEY = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- PAGE CONFIG (Bắt buộc gọi đầu tiên) ---
st.set_page_config(
    page_title="Reddit Insider AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DATABASE MANAGER (SQLITE - LOCAL) với thread safety
# ==========================================
db_lock = Lock()

class DBManager:
    def __init__(self, db_name="reddit_insider.db"):
        # Sử dụng temp directory cho Streamlit Cloud
        if os.environ.get('STREAMLIT_CLOUD'):
            temp_dir = tempfile.gettempdir()
            db_path = os.path.join(temp_dir, db_name)
        else:
            db_path = db_name
            
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with db_lock:
            c = self.conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS groups 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, subreddit TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS history 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, title TEXT, url TEXT, timestamp TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS trend_cache 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, subreddit TEXT, data TEXT, 
                          last_updated TEXT, UNIQUE(subreddit))''')
            self.conn.commit()

    def register(self, username, password):
        with db_lock:
            c = self.conn.cursor()
            hashed = hashlib.sha256(password.encode()).hexdigest()
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                self.conn.commit()
                return True
            except: 
                return False

    def login(self, username, password):
        with db_lock:
            c = self.conn.cursor()
            hashed = hashlib.sha256(password.encode()).hexdigest()
            c.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, hashed))
            return c.fetchone()

    def add_group(self, user_id, subreddit):
        with db_lock:
            c = self.conn.cursor()
            clean_sub = subreddit.replace('r/', '').replace('/', '').strip()
            if not clean_sub: 
                return False
            c.execute("SELECT id FROM groups WHERE user_id=? AND subreddit=?", (user_id, clean_sub))
            if not c.fetchone():
                c.execute("INSERT INTO groups (user_id, subreddit) VALUES (?, ?)", (user_id, clean_sub))
                self.conn.commit()
                return True
            return False

    def get_groups(self, user_id):
        with db_lock:
            c = self.conn.cursor()
            c.execute("SELECT id, subreddit FROM groups WHERE user_id=?", (user_id,))
            return [{'id': r[0], 'subreddit': r[1]} for r in c.fetchall()]

    def delete_group(self, group_id):
        with db_lock:
            c = self.conn.cursor()
            c.execute("DELETE FROM groups WHERE id=?", (group_id,))
            self.conn.commit()

    def add_history(self, user_id, title, url):
        with db_lock:
            c = self.conn.cursor()
            ts = datetime.now().strftime("%d/%m %H:%M")
            c.execute("SELECT id FROM history WHERE user_id=? AND url=? ORDER BY id DESC LIMIT 1", (user_id, url))
            if not c.fetchone():
                c.execute("INSERT INTO history (user_id, title, url, timestamp) VALUES (?, ?, ?, ?)", 
                          (user_id, title, url, ts))
                self.conn.commit()

    def get_history(self, user_id):
        with db_lock:
            c = self.conn.cursor()
            c.execute("SELECT id, title, url, timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20", (user_id,))
            return [{'id': r[0], 'title': r[1], 'url': r[2], 'timestamp': r[3]} for r in c.fetchall()]

    def delete_history(self, hist_id):
        with db_lock:
            c = self.conn.cursor()
            c.execute("DELETE FROM history WHERE id=?", (hist_id,))
            self.conn.commit()

    def cache_trend_data(self, subreddit, data):
        """Cache kết quả phân tích trend"""
        with db_lock:
            c = self.conn.cursor()
            ts = datetime.now().isoformat()
            
            try:
                def json_serializer(obj):
                    if isinstance(obj, (datetime, pd.Timestamp)):
                        return obj.isoformat()
                    elif isinstance(obj, (np.int64, np.float64)):
                        return int(obj) if isinstance(obj, np.int64) else float(obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    return str(obj)
                
                json_data = json.dumps(data, default=json_serializer, ensure_ascii=False)
                
                c.execute(
                    "INSERT OR REPLACE INTO trend_cache (subreddit, data, last_updated) VALUES (?, ?, ?)",
                    (subreddit, json_data, ts)
                )
                self.conn.commit()
                return True
                
            except Exception as e:
                print(f"❌ Cache error: {e}")
                return False

    def get_cached_trend_data(self, subreddit, max_age_minutes=30):
        """Lấy dữ liệu trend từ cache"""
        with db_lock:
            c = self.conn.cursor()
            cutoff_time = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()
            
            c.execute(
                "SELECT data FROM trend_cache WHERE subreddit=? AND last_updated > ?",
                (subreddit, cutoff_time)
            )
            
            result = c.fetchone()
            if result:
                try:
                    loaded_data = json.loads(result[0], object_hook=self._json_date_hook)
                    return loaded_data
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"❌ JSON decode error for cached data: {e}")
                    return None
            return None
    
    def _json_date_hook(self, dct):
        """Hàm helper để chuyển đổi chuỗi ISO thành datetime"""
        for key, value in dct.items():
            if isinstance(value, str):
                try:
                    dct[key] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass
        return dct

db = DBManager()

# ==========================================
# 3. CORE LOGIC VỚI FALLBACK TỰ ĐỘNG - ĐÃ SỬA
# ==========================================
class RedditLoader:
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        # TẠO USER-AGENT NGẪU NHIÊN ĐỂ TRÁNH BỊ CHẶN
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        self.user_agent_base = f'web:reddit_insider_ai_{random_id}:v1.0.0'
        
        self.headers = {
            'User-Agent': self.user_agent_base,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def fetch_data(self, url, retries=3, current_retry=0):
        """Fetch data với fallback tự động ẩn"""
        if current_retry >= retries:
            return None, f"Đã thử {retries} lần nhưng không thành công"
        
        try:
            # Thay đổi User-Agent cho mỗi lần thử
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
            self.session.headers['User-Agent'] = f'{self.user_agent_base}_{random_suffix}'
            
            if not url.startswith('http'):
                url = 'https://' + url
            
            if 'reddit.com' not in url:
                return None, "URL không phải là Reddit"
            
            # FALLBACK TỰ ĐỘNG: Strategy chain
            if current_retry == 0:
                # Thử JSON API đầu tiên
                url = self._normalize_url(url)
            elif current_retry == 1:
                # Thử old.reddit.com
                if 'www.reddit.com' in url:
                    url = url.replace('www.reddit.com', 'old.reddit.com')
                    url = self._normalize_url(url)
            elif current_retry == 2:
                # Thử RSS feed với format=xml
                url = self._convert_to_rss_url(url)
                if not url:
                    return None, "Không thể chuyển sang RSS"
            
            response = self.session.get(url, timeout=15, allow_redirects=True)
            
            if response.status_code == 200:
                if '.rss' in url or 'format=xml' in url:
                    # Parse RSS
                    return self._parse_rss_data(response.text, url)
                else:
                    try:
                        data = response.json()
                        return self._parse_reddit_data(data, url)
                    except json.JSONDecodeError:
                        return self._parse_html_fallback(response.text, url)
            
            elif response.status_code == 403:
                # Tự động thử phương thức khác
                time.sleep(1)
                return self.fetch_data(url, retries, current_retry + 1)
            
            elif response.status_code == 429:
                if current_retry < 2:
                    time.sleep(3)
                    return self.fetch_data(url, retries, current_retry + 1)
                else:
                    return None, "Reddit đang chặn yêu cầu. Vui lòng thử lại sau 1 phút."
            
            elif response.status_code == 404:
                if current_retry < retries - 1:
                    return self.fetch_data(url, retries, current_retry + 1)
                return None, "Không tìm thấy bài viết"
            
            else:
                return None, f"Lỗi HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return None, "Timeout khi kết nối đến Reddit"
        except Exception as e:
            return None, f"Lỗi: {str(e)[:100]}"
    
    def _normalize_url(self, url):
        """Chuẩn hóa URL Reddit"""
        if '?' in url:
            url = url.split('?')[0]
        
        url = url.rstrip('/')
        
        if '/comments/' in url and not url.endswith('.json') and not url.endswith('.rss'):
            url = f"{url}.json"
        elif '/r/' in url and not url.endswith('.json') and not url.endswith('.rss') and '/comments/' not in url:
            url = f"{url}.json"
        
        return url
    
    def _convert_to_rss_url(self, url):
        """Chuyển URL sang RSS format với ?format=xml"""
        try:
            if '/comments/' in url:
                match = re.search(r'/r/([^/]+)/comments/([^/]+)', url)
                if match:
                    subreddit, post_id = match.groups()
                    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.rss?format=xml"
            elif '/r/' in url:
                url = url.replace('.json', '').rstrip('/')
                return f"{url}.rss?format=xml"
        except:
            pass
        return None
    
    def _parse_rss_data(self, rss_content, url):
        """Parse RSS data"""
        try:
            import xml.etree.ElementTree as ET
            
            # Parse XML
            root = ET.fromstring(rss_content)
            
            # Tìm title
            title = ""
            for item in root.findall('.//item'):
                title_elem = item.find('title')
                if title_elem is not None:
                    title = title_elem.text
                    break
            
            # Tìm subreddit
            subreddit = "unknown"
            match = re.search(r'/r/([^/]+)', url)
            if match:
                subreddit = match.group(1)
            
            meta = {
                'title': title or 'No Title',
                'subreddit': subreddit,
                'score': 0,
                'author': 'Unknown',
                'content': 'Content from RSS feed',
                'upvote_ratio': 0,
                'created_utc': time.time(),
                'created_time': 'Không rõ',
                'num_comments': 0,
                'permalink': url,
                'url': url,
                'id': 'rss_' + str(hash(url) % 10000)
            }
            
            return {'meta': meta, 'comments': []}, None
            
        except Exception as e:
            print(f"RSS parse error: {e}")
            return self._parse_html_fallback(rss_content, url)
    
    def _parse_reddit_data(self, data, original_url):
        """Parse dữ liệu Reddit JSON"""
        try:
            meta = {}
            comments = []
            
            if isinstance(data, list) and len(data) >= 2:
                post_part = data[0]
                if ('data' in post_part and 
                    'children' in post_part['data'] and 
                    len(post_part['data']['children']) > 0):
                    
                    post_data = post_part['data']['children'][0]['data']
                    meta = self._extract_post_meta(post_data, original_url)
                    
                    comments_part = data[1]
                    if ('data' in comments_part and 
                        'children' in comments_part['data']):
                        
                        comments_data = comments_part['data']['children']
                        comments = self._extract_comments(comments_data)
                
            elif isinstance(data, dict):
                if 'data' in data and 'children' in data['data']:
                    children = data['data']['children']
                    
                    if children and len(children) > 0:
                        item = children[0]
                        if 'data' in item:
                            item_data = item['data']
                            
                            if item.get('kind') == 't3':
                                meta = self._extract_post_meta(item_data, original_url)
                            elif item.get('kind') == 't1':
                                comments = self._extract_comments(children)
            
            if not meta:
                return None, "Không thể phân tích dữ liệu bài viết"
            
            return {'meta': meta, 'comments': comments}, None
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            return None, f"Lỗi phân tích: {str(e)}"
    
    def _extract_post_meta(self, post_data, original_url):
        """Trích xuất metadata từ post data"""
        permalink = post_data.get('permalink', '')
        if permalink and not permalink.startswith('http'):
            permalink = f"https://www.reddit.com{permalink}"
        
        # Tính thời gian đăng
        created_time = ""
        if 'created_utc' in post_data:
            try:
                post_time = datetime.fromtimestamp(post_data['created_utc'])
                now = datetime.now()
                diff = now - post_time
                
                if diff.days > 0:
                    created_time = f"{diff.days} ngày trước"
                elif diff.seconds > 3600:
                    hours = diff.seconds // 3600
                    created_time = f"{hours} giờ trước"
                else:
                    minutes = diff.seconds // 60
                    created_time = f"{minutes} phút trước"
            except:
                created_time = "Không rõ"
        
        return {
            'title': post_data.get('title', 'No Title'),
            'subreddit': post_data.get('subreddit', 'unknown'),
            'score': post_data.get('score', 0),
            'author': post_data.get('author', '[deleted]'),
            'content': post_data.get('selftext', '')[:1500],
            'upvote_ratio': post_data.get('upvote_ratio', 0),
            'created_utc': post_data.get('created_utc', 0),
            'created_time': created_time,
            'num_comments': post_data.get('num_comments', 0),
            'permalink': permalink,
            'url': permalink or original_url,
            'id': post_data.get('id', ''),
        }
    
    def _extract_comments(self, comments_data):
        """Trích xuất comments"""
        comments = []
        
        for child in comments_data:
            try:
                if child.get('kind') == 't1':
                    comment = child['data']
                    
                    if comment.get('body') in ['[deleted]', '[removed]']:
                        continue
                    
                    comments.append({
                        'body': comment.get('body', ''),
                        'author': comment.get('author', '[deleted]'),
                        'score': comment.get('score', 0),
                        'created_utc': comment.get('created_utc', 0),
                        'id': comment.get('id', ''),
                    })
            except Exception as e:
                continue
        
        return comments

    def _parse_html_fallback(self, html, url):
        """Fallback parse từ HTML"""
        try:
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else 'No Title'
            title = title.replace(' : reddit', '').replace(' - Reddit', '').strip()
            
            subreddit_match = re.search(r'/r/([^/"\']+)', url)
            subreddit = subreddit_match.group(1) if subreddit_match else 'unknown'
            
            meta = {
                'title': title,
                'subreddit': subreddit,
                'score': 0,
                'author': 'Unknown',
                'content': 'Content available in HTML only',
                'url': url,
                'num_comments': 0,
                'permalink': url,
                'created_time': 'Không rõ'
            }
            
            return {'meta': meta, 'comments': []}, None
            
        except Exception as e:
            return None, f"HTML parse error: {e}"

class TrendingManager:
    def __init__(self):
        self.mirrors = ["https://www.reddit.com", "https://old.reddit.com"]
        # USER-AGENT CHO TRENDING - TẠO NGẪU NHIÊN
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.user_agent_base = f'web:reddit_trending_fetcher_{random_id}:v1.0.0'
        
    def fetch_feed(self, subreddits, limit=15):
        """Lấy dữ liệu bài viết từ subreddits với fallback tự động"""
        results = []
        
        for sub in subreddits:
            sub = sub.strip().replace('r/', '').replace('/', '')
            
            # Strategy 1: Thử các mirrors
            posts = self._try_mirrors(sub, limit)
            
            # Strategy 2: Thử RSS feed
            if not posts:
                posts = self._fetch_rss_feed(sub, limit)
            
            results.extend(posts)
        
        # Sắp xếp theo thời gian
        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return results
    
    def _try_mirrors(self, subreddit, limit):
        """Thử lấy data qua các mirrors"""
        for domain in self.mirrors:
            try:
                # Tạo User-Agent ngẫu nhiên cho mỗi request
                random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                headers = {
                    'User-Agent': f'{self.user_agent_base}_{random_suffix}',
                    'Accept': 'application/json'
                }
                
                url = f"{domain}/r/{subreddit}/hot.json?limit={limit}"
                resp = requests.get(url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_posts(data['data']['children'], subreddit)
                elif resp.status_code == 403:
                    # Thử RSS trên domain này
                    rss_url = f"{domain}/r/{subreddit}/hot.rss?format=xml"
                    rss_resp = requests.get(rss_url, headers=headers, timeout=10)
                    if rss_resp.status_code == 200:
                        return self._parse_rss_feed(rss_resp.text, subreddit)
                        
            except Exception as e:
                print(f"Error fetching from {domain}: {e}")
                continue
        
        return []
    
    def _parse_rss_feed(self, rss_content, subreddit):
        """Parse RSS feed content"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(rss_content)
            
            posts = []
            for item in root.findall('.//item'):
                try:
                    title = item.find('title').text if item.find('title') is not None else 'No Title'
                    link = item.find('link').text if item.find('link') is not None else ''
                    
                    post_id = ''
                    if '/comments/' in link:
                        post_id = link.split('/comments/')[1].split('/')[0]
                    
                    author = 'Unknown'
                    author_elem = item.find('{http://purl.org/dc/elements/1.1/}creator')
                    if author_elem is not None:
                        author = author_elem.text
                    
                    post = {
                        'id': post_id or f"rss_{len(posts)}",
                        'title': title,
                        'url': link,
                        'subreddit': subreddit,
                        'author': author,
                        'score': 0,
                        'comments_count': 0,
                        'created_utc': time.time(),
                        'timestamp': time.time(),
                        'thumbnail': None,
                        'selftext': '',
                        'upvote_ratio': 0,
                        'time_str': datetime.now().strftime('%H:%M %d/%m')
                    }
                    posts.append(post)
                except:
                    continue
            
            return posts
            
        except Exception as e:
            print(f"RSS parse error: {e}")
            return []
    
    def _fetch_rss_feed(self, subreddit, limit):
        """Fallback sử dụng RSS feed"""
        try:
            # Thêm format=xml vào RSS URL
            rss_url = f"https://www.reddit.com/r/{subreddit}/hot.rss?format=xml"
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            headers = {
                'User-Agent': f'{self.user_agent_base}_{random_suffix}'
            }
            
            response = requests.get(rss_url, headers=headers, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                posts = []
                for entry in feed.entries[:limit]:
                    post = {
                        'id': entry.id.split('/')[-1] if hasattr(entry, 'id') else f"rss_{len(posts)}",
                        'title': entry.title if hasattr(entry, 'title') else 'No Title',
                        'url': entry.link if hasattr(entry, 'link') else '',
                        'subreddit': subreddit,
                        'author': entry.author if hasattr(entry, 'author') else 'Unknown',
                        'score': 0,
                        'comments_count': 0,
                        'created_utc': time.mktime(entry.updated_parsed) if hasattr(entry, 'updated_parsed') else time.time(),
                        'timestamp': time.mktime(entry.updated_parsed) if hasattr(entry, 'updated_parsed') else time.time(),
                        'thumbnail': None,
                        'selftext': '',
                        'upvote_ratio': 0,
                        'time_str': datetime.now().strftime('%H:%M %d/%m')
                    }
                    posts.append(post)
                return posts
        except:
            pass
        return []
    
    def _parse_posts(self, posts_data, subreddit):
        """Parse dữ liệu bài viết từ JSON response"""
        posts = []
        for child in posts_data:
            p = child['data']
            try:
                thumb = None
                if p.get('thumbnail') and p['thumbnail'].startswith('http'):
                    thumb = p['thumbnail']
                elif p.get('preview'):
                    try:
                        thumb = p['preview']['images'][0]['source']['url'].replace('&amp;', '&')
                    except:
                        pass
                
                post = {
                    'id': p['id'],
                    'title': p.get('title', 'No Title'),
                    'url': f"https://www.reddit.com{p['permalink']}",
                    'subreddit': subreddit,
                    'author': p.get('author', '[deleted]'),
                    'score': p.get('score', 0),
                    'comments_count': p.get('num_comments', 0),
                    'created_utc': p.get('created_utc', time.time()),
                    'timestamp': p.get('created_utc', time.time()),
                    'thumbnail': thumb,
                    'selftext': p.get('selftext', ''),
                    'upvote_ratio': p.get('upvote_ratio', 0),
                    'time_str': datetime.fromtimestamp(p.get('created_utc', time.time())).strftime('%H:%M %d/%m')
                }
                posts.append(post)
            except Exception:
                continue
        return posts

# FORECAST ENGINE NÂNG CAO
class AdvancedForecastEngine:
    """Forecast engine với nhiều biểu đồ"""
    
    def forecast(self, posts_data, days=7):
        if not posts_data or len(posts_data) < 3:
            return self._get_empty_forecast()
        
        try:
            # Tính engagement và các chỉ số
            engagements = []
            scores = []
            comments = []
            timestamps = []
            
            for post in posts_data:
                engagement = post.get('score', 0) + post.get('comments_count', 0) * 2
                engagements.append(engagement)
                scores.append(post.get('score', 0))
                comments.append(post.get('comments_count', 0))
                timestamps.append(post.get('created_utc', time.time()))
            
            # Phân tích trend
            avg_engagement = np.mean(engagements)
            std_engagement = np.std(engagements)
            
            # Xác định xu hướng
            if len(engagements) >= 5:
                recent_avg = np.mean(engagements[-3:])
                older_avg = np.mean(engagements[:3])
                
                if recent_avg > older_avg * 1.3:
                    trend = "Tăng mạnh 🚀"
                    trend_slope = 0.03
                elif recent_avg > older_avg * 1.1:
                    trend = "Tăng nhẹ ↗️"
                    trend_slope = 0.015
                elif recent_avg < older_avg * 0.7:
                    trend = "Giảm mạnh 📉"
                    trend_slope = -0.03
                elif recent_avg < older_avg * 0.9:
                    trend = "Giảm nhẹ ↘️"
                    trend_slope = -0.015
                else:
                    trend = "Ổn định ➡️"
                    trend_slope = 0.0
            else:
                trend = "Đang phân tích 📊"
                trend_slope = 0.02
            
            # Tạo forecast data
            forecast_data = []
            today = datetime.now()
            
            for i in range(min(days, 7)):
                future_date = today + timedelta(days=i+1)
                
                # Dự báo với nhiễu ngẫu nhiên nhỏ
                base_prediction = avg_engagement * (1 + trend_slope) ** (i + 1)
                noise = np.random.normal(0, std_engagement * 0.1)
                predicted = max(10, base_prediction + noise)
                
                forecast_data.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'predicted_engagement': round(predicted, 1),
                    'predicted_lower': round(predicted * 0.8, 1),
                    'predicted_upper': round(predicted * 1.2, 1),
                    'confidence': 'medium'
                })
            
            # Tạo dữ liệu cho biểu đồ lịch sử
            history_data = []
            for i, (eng, ts) in enumerate(zip(engagements, timestamps)):
                try:
                    date_str = datetime.fromtimestamp(ts).strftime('%m/%d')
                    history_data.append({
                        'date': date_str,
                        'engagement': eng,
                        'score': scores[i],
                        'comments': comments[i]
                    })
                except:
                    continue
            
            return {
                'forecast': forecast_data,
                'history': history_data[:10],  # Lấy 10 điểm gần nhất
                'trend_direction': trend,
                'trend_slope': trend_slope,
                'current_stats': {
                    'avg_engagement': float(avg_engagement),
                    'std_engagement': float(std_engagement),
                    'max_engagement': float(np.max(engagements)),
                    'min_engagement': float(np.min(engagements)),
                    'total_posts': len(posts_data)
                },
                'confidence_interval': 'medium',
                'method_used': 'advanced_regression'
            }
            
        except Exception as e:
            print(f"Forecast error: {e}")
            return self._get_empty_forecast()
    
    def _get_empty_forecast(self):
        """Tạo forecast mặc định khi không có dữ liệu"""
        today = datetime.now()
        forecast_data = []
        
        for i in range(7):
            future_date = today + timedelta(days=i+1)
            forecast_data.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_engagement': 50 + i * 5,
                'predicted_lower': 30 + i * 3,
                'predicted_upper': 70 + i * 7,
                'confidence': 'low'
            })
        
        return {
            'forecast': forecast_data,
            'history': [],
            'trend_direction': 'Đang phân tích 📊',
            'trend_slope': 0.02,
            'current_stats': {
                'avg_engagement': 50,
                'std_engagement': 10,
                'max_engagement': 100,
                'min_engagement': 10,
                'total_posts': 0
            },
            'confidence_interval': 'low',
            'method_used': 'default'
        }

class TrendAnalysisManager:
    def __init__(self):
        self.forecast_engine = AdvancedForecastEngine()
    
    def analyze_subreddit_trends(self, subreddit, posts_data, days=7):
        """Phân tích xu hướng chi tiết"""
        
        # Kiểm tra cache
        cached_data = db.get_cached_trend_data(subreddit)
        if cached_data:
            return cached_data
        
        # Sử dụng AdvancedForecastEngine
        forecast_result = self.forecast_engine.forecast(posts_data, days)
        
        # Tạo kết quả hoàn chỉnh
        result = {
            'subreddit': subreddit,
            'analysis_period_days': days,
            'data_summary': self._calculate_basic_summary(posts_data),
            'peak_hours': self._calculate_peak_hours(posts_data),
            'top_keywords': self._extract_simple_keywords(posts_data),
            'top_topics': self._extract_topics(posts_data),
            'forecast': forecast_result,
            'analysis_timestamp': datetime.now().isoformat(),
            'note': 'Phân tích nâng cao với dự báo 7 ngày 📊'
        }
        
        # Cache kết quả
        db.cache_trend_data(subreddit, result)
        
        return result

    def _calculate_basic_summary(self, posts_data):
        """Tính summary cơ bản"""
        if not posts_data:
            return {}
            
        total_posts = len(posts_data)
        total_score = sum(p.get('score', 0) for p in posts_data)
        total_comments = sum(p.get('comments_count', 0) for p in posts_data)
        total_engagement = total_score + total_comments * 2
        
        # Tính độ biến động
        engagements = [p.get('score', 0) + p.get('comments_count', 0) * 2 for p in posts_data]
        volatility = np.std(engagements) if len(engagements) > 1 else 0
        
        return {
            'total_posts_analyzed': total_posts,
            'total_score': int(total_score),
            'total_engagement': int(total_engagement),
            'total_comments': int(total_comments),
            'avg_score_per_post': float(total_score / total_posts) if total_posts > 0 else 0,
            'avg_comments_per_post': float(total_comments / total_posts) if total_posts > 0 else 0,
            'avg_engagement_per_post': float(total_engagement / total_posts) if total_posts > 0 else 0,
            'volatility': float(volatility),
            'engagement_range': f"{min(engagements) if engagements else 0} - {max(engagements) if engagements else 0}"
        }

    def _calculate_peak_hours(self, posts_data):
        """Tính giờ cao điểm từ posts data"""
        if not posts_data:
            return []
            
        try:
            hour_engagement = {}
            hour_posts = {}
            
            for post in posts_data:
                try:
                    hour = datetime.fromtimestamp(post['created_utc']).hour
                    engagement = post.get('score', 0) + post.get('comments_count', 0) * 2
                    
                    if hour not in hour_engagement:
                        hour_engagement[hour] = 0
                        hour_posts[hour] = 0
                    
                    hour_engagement[hour] += engagement
                    hour_posts[hour] += 1
                except:
                    continue
            
            peak_hours = []
            for hour in sorted(hour_engagement.keys()):
                peak_hours.append({
                    'hour': int(hour),
                    'total_engagement': int(hour_engagement[hour]),
                    'post_count': int(hour_posts.get(hour, 0)),
                    'avg_engagement': int(hour_engagement[hour] / hour_posts[hour]) if hour_posts.get(hour, 0) > 0 else 0
                })
            
            return sorted(peak_hours, key=lambda x: x['total_engagement'], reverse=True)
            
        except Exception as e:
            print(f"Peak hours error: {e}")
            return []

    def _extract_simple_keywords(self, posts_data, top_n=10):
        """Trích xuất keywords"""
        if not posts_data:
            return []
            
        try:
            all_titles = " ".join([str(p.get('title', '')) for p in posts_data])
            words = re.findall(r'\b[a-zA-ZÀ-ỹ]{4,}\b', all_titles.lower())
            
            stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 'from', 'they', 'what', 
                        'about', 'when', 'where', 'which', 'would', 'could', 'should', 'their'}
            filtered_words = [w for w in words if w not in stopwords]
            
            word_counts = Counter(filtered_words).most_common(top_n)
            
            return [{'keyword': w.capitalize(), 'score': c/len(filtered_words) if filtered_words else 0} 
                   for w, c in word_counts]
                   
        except Exception as e:
            print(f"Keywords error: {e}")
            return []

    def _extract_topics(self, posts_data, top_n=5):
        """Trích xuất chủ đề từ nội dung"""
        if not posts_data:
            return []
        
        try:
            # Sử dụng keywords làm chủ đề đơn giản
            keywords = self._extract_simple_keywords(posts_data, top_n=top_n*2)
            
            topics = []
            for kw in keywords[:top_n]:
                topics.append({
                    'topic': kw['keyword'],
                    'relevance': kw['score'],
                    'posts_count': sum(1 for p in posts_data if kw['keyword'].lower() in p.get('title', '').lower())
                })
            
            return sorted(topics, key=lambda x: x['relevance'], reverse=True)
        except:
            return []

# ==========================================
# AI ANALYST NÂNG CAO VỚI GEMINI 2.0 FLASH
# ==========================================
class AdvancedAIAnalyst:
    def __init__(self):
        self.key = GOOGLE_GEMINI_API_KEY
        self.last_request_time = None
        self.request_count = 0
        self.cooldown_until = None
        
    def analyze(self, meta, comments):
        """Phân tích AI nâng cao với Gemini 2.0 Flash"""
        
        # 1. Kiểm tra điều kiện cơ bản
        if not self.key:
            return self._get_comprehensive_fallback(meta, comments, "Chưa cấu hình API key")
        
        if not GEMINI_AVAILABLE:
            return self._get_comprehensive_fallback(meta, comments, "Thư viện Gemini chưa được cài đặt")
        
        # 2. Kiểm tra rate limiting
        if self._should_use_fallback():
            return self._get_comprehensive_fallback(meta, comments, "AI đang bận, vui lòng thử lại sau")
        
        # 3. Thử gọi API
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.key)
            
            # Chuẩn bị dữ liệu
            analysis_data = self._prepare_analysis_data(meta, comments)
            
            # Tạo prompt chi tiết với yêu cầu tóm tắt
            prompt = self._create_detailed_prompt(analysis_data)
            
            # Gọi Gemini 2.0 Flash
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Update tracking
                self.last_request_time = datetime.now()
                self.request_count += 1
                
                # Gọi API
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 800,  # Tăng lên để có tóm tắt chi tiết
                        "top_p": 0.95,
                        "top_k": 40
                    },
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                )
                
                if response and response.text:
                    return self._format_ai_response(response.text, analysis_data)
                else:
                    return self._get_comprehensive_fallback(meta, comments, "Không nhận được phản hồi từ AI")
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    self.cooldown_until = datetime.now() + timedelta(minutes=5)
                    return self._get_comprehensive_fallback(meta, comments, "Đang sử dụng phân tích cơ bản do giới hạn API")
                else:
                    return self._get_comprehensive_fallback(meta, comments, f"Lỗi AI: {error_msg[:100]}")
                
        except Exception as e:
            return self._get_comprehensive_fallback(meta, comments, f"Lỗi hệ thống: {str(e)[:100]}")
    
    def _should_use_fallback(self):
        """Kiểm tra có nên dùng fallback không"""
        # Nếu đang trong cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return True
        
        # Nếu đã gọi quá nhiều request
        if self.request_count >= 10:
            if self.last_request_time:
                time_diff = (datetime.now() - self.last_request_time).total_seconds()
                if time_diff < 300:  # 5 phút
                    return True
        
        return False
    
    def _prepare_analysis_data(self, meta, comments):
        """Chuẩn bị dữ liệu phân tích"""
        # Phân tích sentiment từ comments
        sentiments = []
        emotion_counts = {}
        
        for c in comments[:20]:  # Chỉ phân tích 20 comments đầu
            try:
                blob = TextBlob(c['body'])
                pol = blob.sentiment.polarity
                
                if pol > 0.3:
                    sentiment = "Rất tích cực"
                    emotion = "😊 Hài lòng"
                elif pol > 0.1:
                    sentiment = "Tích cực"
                    emotion = "🙂 Vui vẻ"
                elif pol < -0.3:
                    sentiment = "Rất tiêu cực"
                    emotion = "😠 Tức giận"
                elif pol < -0.1:
                    sentiment = "Tiêu cực"
                    emotion = "😟 Lo lắng"
                else:
                    sentiment = "Trung lập"
                    emotion = "😐 Bình thường"
                
                sentiments.append({
                    'sentiment': sentiment,
                    'emotion': emotion,
                    'polarity': pol,
                    'text': c['body'][:100] + '...' if len(c['body']) > 100 else c['body'],
                    'score': c.get('score', 0)
                })
                
                # Đếm emotion
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
                    
            except:
                continue
        
        # Tính toán thống kê
        total_comments = len(comments)
        analyzed_comments = len(sentiments)
        
        # Top comments
        top_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)[:3]
        
        # Tạo tóm tắt nội dung
        content_summary = self._create_content_summary(meta, comments)
        
        return {
            'meta': meta,
            'total_comments': total_comments,
            'analyzed_comments': analyzed_comments,
            'sentiments': sentiments,
            'emotion_counts': emotion_counts,
            'top_comments': top_comments,
            'engagement_score': meta.get('score', 0) + total_comments * 2,
            'content_summary': content_summary
        }
    
    def _create_content_summary(self, meta, comments):
        """Tạo tóm tắt nội dung bài viết"""
        try:
            # Lấy nội dung chính
            title = meta.get('title', '')
            content = meta.get('content', '')
            
            # Nếu content quá ngắn, lấy thêm từ comments
            if len(content) < 100 and comments:
                # Lấy 3 comments hàng đầu
                top_comments_text = ' '.join([c['body'][:200] for c in comments[:3]])
                full_text = f"{title}. {content}. {top_comments_text}"
            else:
                full_text = f"{title}. {content}"
            
            # Giới hạn độ dài
            if len(full_text) > 1500:
                full_text = full_text[:1500] + "..."
            
            # Tạo tóm tắt đơn giản
            words = full_text.split()
            if len(words) > 100:
                # Lấy câu đầu và cuối
                sentences = re.split(r'[.!?]+', full_text)
                if len(sentences) > 2:
                    summary = sentences[0] + ". " + sentences[-2] + "."
                else:
                    summary = ' '.join(words[:50]) + "..."
            else:
                summary = full_text
            
            return summary[:500] + "..." if len(summary) > 500 else summary
            
        except Exception as e:
            print(f"Error creating summary: {e}")
            return meta.get('title', '')[:200]
    
    def _create_detailed_prompt(self, data):
        """Tạo prompt chi tiết cho AI với yêu cầu tóm tắt"""
        meta = data['meta']
        
        prompt = f"""Hãy phân tích bài đăng Reddit sau đây bằng tiếng Việt:

**THÔNG TIN BÀI ĐĂNG:**
- Tiêu đề: {meta.get('title', 'Không có tiêu đề')}
- Subreddit: r/{meta.get('subreddit', 'unknown')}
- Tác giả: {meta.get('author', 'Ẩn danh')}
- Điểm: {meta.get('score', 0)}
- Số bình luận: {data['total_comments']}
- Tỷ lệ upvote: {meta.get('upvote_ratio', 0):.1%}
- Engagement: {data['engagement_score']}

**NỘI DUNG CHÍNH (ĐÃ TÓM TẮT):**
{data['content_summary']}

**THỐNG KÊ CẢM XÚC ({data['analyzed_comments']}/{data['total_comments']} bình luận):**
{self._format_sentiment_stats(data['emotion_counts'])}

**YÊU CẦU PHÂN TÍCH:**
Hãy cung cấp phân tích với các phần sau:

1. **TÓM TẮT CHI TIẾT NỘI DUNG** (4-5 câu bằng tiếng Việt):
   - Tóm tắt ý chính của bài viết
   - Mục đích chính của tác giả
   - Thông tin quan trọng nhất

2. **PHÂN TÍCH CẢM XÚC CỘNG ĐỒNG**:
   - Xu hướng cảm xúc chung
   - Điểm đáng chú ý về phản ứng của cộng đồng
   - Mức độ tham gia thảo luận

3. **ĐÁNH GIÁ CHẤT LƯỢNG**:
   - Chất lượng nội dung bài đăng
   - Mức độ tương tác (engagement)
   - Tiềm năng viral (nếu có)

4. **KHUYẾN NGHỊ**:
   - Khuyến nghị cho tác giả (nếu cần)
   - Thời điểm tốt nhất để tham gia thảo luận

**LƯU Ý:** Hãy viết ngắn gọn, súc tích, tập trung vào insights có giá trị. Ưu tiên tóm tắt nội dung rõ ràng.

**PHÂN TÍCH:**"""
        
        return prompt
    
    def _format_sentiment_stats(self, emotion_counts):
        """Định dạng thống kê cảm xúc"""
        if not emotion_counts:
            return "Không có dữ liệu cảm xúc"
        
        stats = []
        total = sum(emotion_counts.values())
        
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            stats.append(f"- {emotion}: {count} ({percentage:.1f}%)")
        
        return "\n".join(stats)
    
    def _format_ai_response(self, response_text, data):
        """Định dạng phản hồi AI"""
        # Thêm header cho phân tích AI
        formatted = f"""
## 🤖 PHÂN TÍCH AI CHI TIẾT
        
**📊 Thông tin bài đăng:**
- **Tiêu đề:** {data['meta'].get('title', 'Không có tiêu đề')}
- **Subreddit:** r/{data['meta'].get('subreddit', 'unknown')}
- **Engagement:** {data['engagement_score']} điểm
- **Phân tích:** {data['analyzed_comments']}/{data['total_comments']} bình luận

---

{response_text}

---

*Phân tích được thực hiện bởi Gemini 2.0 Flash • {datetime.now().strftime("%H:%M %d/%m/%Y")}*
"""
        return formatted
    
    def _get_comprehensive_fallback(self, meta, comments, reason=""):
        """Fallback phân tích chi tiết với tóm tắt nội dung"""
        title = meta.get('title', 'Không có tiêu đề')
        score = meta.get('score', 0)
        upvote_ratio = meta.get('upvote_ratio', 0)
        num_comments = len(comments)
        engagement = score + num_comments * 2
        
        # Tạo tóm tắt nội dung
        content_summary = self._create_content_summary(meta, comments)
        
        # Phân tích sentiment cơ bản
        sentiment_stats = self._analyze_comments_sentiment_basic(comments)
        
        # Xác định chất lượng bài đăng
        quality = "Tốt" if engagement > 100 else "Trung bình" if engagement > 30 else "Thấp"
        viral_potential = "Cao" if engagement > 300 else "Trung bình" if engagement > 100 else "Thấp"
        
        # Tạo phân tích fallback chi tiết
        analysis = f"""
## 📊 PHÂN TÍCH CƠ BẢN

**🔍 THÔNG TIN BÀI ĐĂNG:**
- **Tiêu đề:** {title[:80]}...
- **Subreddit:** r/{meta.get('subreddit', 'unknown')}
- **Tác giả:** {meta.get('author', 'Ẩn danh')}
- **Thời gian:** {meta.get('created_time', 'Không rõ')}

**📈 CHỈ SỐ TƯƠNG TÁC:**
- **Điểm:** {score} ⭐
- **Tỷ lệ upvote:** {upvote_ratio:.1%} 📊
- **Bình luận:** {num_comments} 💬
- **Engagement:** {engagement} 📈
- **Chất lượng:** {quality} 
- **Tiềm năng viral:** {viral_potential}

### 📝 TÓM TẮT NỘI DUNG (Tiếng Việt)
{content_summary}

**🎭 PHÂN TÍCH CẢM XÚC:**
{sentiment_stats}

**💡 NHẬN XÉT CHUNG:**
- Bài đăng có mức độ tương tác **{quality.lower()}**
- Cộng đồng đang có phản ứng **{'tích cực' if '😊' in sentiment_stats else 'trung lập' if '😐' in sentiment_stats else 'tiêu cực'}**
- {'Có tiềm năng thu hút thêm tương tác' if viral_potential == 'Cao' else 'Cần cải thiện để tăng tương tác'}

"""
        return analysis
    
    def _analyze_comments_sentiment_basic(self, comments):
        """Phân tích sentiment cơ bản từ comments"""
        if not comments:
            return "Không có bình luận để phân tích"
        
        sentiments = []
        for c in comments[:10]:
            try:
                blob = TextBlob(c['body'])
                pol = blob.sentiment.polarity
                
                if pol > 0.1:
                    sentiments.append('positive')
                elif pol < -0.1:
                    sentiments.append('negative')
                else:
                    sentiments.append('neutral')
            except:
                continue
        
        if not sentiments:
            return "Không thể phân tích cảm xúc"
        
        pos = sentiments.count('positive')
        neg = sentiments.count('negative')
        neu = sentiments.count('neutral')
        total = len(sentiments)
        
        return f"""
- 😊 **Tích cực:** {pos} ({pos/total*100:.1f}%)
- 😐 **Trung lập:** {neu} ({neu/total*100:.1f}%)
- 😟 **Tiêu cực:** {neg} ({neg/total*100:.1f}%)
"""

# ==========================================
# DATA PROCESSING & EXPORT
# ==========================================
def process_nlp(comments):
    """Xử lý NLP chi tiết cho comments"""
    if not comments:
        return pd.DataFrame()
    
    data = []
    for idx, c in enumerate(comments):
        try:
            blob = TextBlob(c['body'])
            pol = blob.sentiment.polarity
            subj = blob.sentiment.subjectivity
            
            # Xác định sentiment
            if pol > 0.3:
                sent = 'Rất tích cực'
                emoji = '😊'
                color = '#4CAF50'
            elif pol > 0.1:
                sent = 'Tích cực'
                emoji = '🙂'
                color = '#8BC34A'
            elif pol < -0.3:
                sent = 'Rất tiêu cực'
                emoji = '😠'
                color = '#F44336'
            elif pol < -0.1:
                sent = 'Tiêu cực'
                emoji = '😟'
                color = '#FF9800'
            else:
                sent = 'Trung lập'
                emoji = '😐'
                color = '#9E9E9E'
            
            # Xác định emotion từ từ khóa
            txt = c['body'].lower()
            emotion = 'Bình thường'
            if any(x in txt for x in ['love', 'amazing', 'perfect', 'excellent', 'best']):
                emotion = 'Yêu thích ❤️'
            elif any(x in txt for x in ['hate', 'terrible', 'worst', 'awful', 'bad']):
                emotion = 'Ghét bỏ 💔'
            elif any(x in txt for x in ['lol', 'haha', 'funny', 'hilarious']):
                emotion = 'Vui vẻ 😂'
            elif any(x in txt for x in ['sad', 'sorry', 'unfortunately', 'bad news']):
                emotion = 'Buồn bã 😢'
            
            data.append({
                'id': idx + 1,
                'comment_id': c.get('id', f'c{idx}'),
                'author': c.get('author', '[deleted]'),
                'text': c['body'][:200] + '...' if len(c['body']) > 200 else c['body'],
                'score': c.get('score', 0),
                'polarity': round(pol, 3),
                'subjectivity': round(subj, 3),
                'sentiment': sent,
                'sentiment_emoji': emoji,
                'sentiment_color': color,
                'emotion': emotion,
                'word_count': len(c['body'].split()),
                'char_count': len(c['body'])
            })
        except Exception as e:
            print(f"Error processing comment {idx}: {e}")
            continue
    
    return pd.DataFrame(data)

def create_download_link(df, filename="sentiment_analysis.csv"):
    """Tạo link download CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Tải xuống CSV</a>'
    return href

def create_visualization(df):
    """Tạo visualization cho sentiment analysis"""
    if df.empty:
        return None
    
    # 1. Pie chart phân bố sentiment
    sentiment_counts = df['sentiment'].value_counts()
    
    fig1 = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.3,
        marker=dict(colors=df.drop_duplicates('sentiment').set_index('sentiment').loc[sentiment_counts.index, 'sentiment_color'].tolist())
    )])
    
    fig1.update_layout(
        title='Phân bố cảm xúc',
        height=400,
        showlegend=True
    )
    
    # 2. Bar chart sentiment theo điểm
    if 'score' in df.columns:
        fig2 = px.bar(
            df.nlargest(10, 'score'),
            x='author',
            y='score',
            color='sentiment',
            title='Top 10 bình luận điểm cao nhất',
            labels={'author': 'Tác giả', 'score': 'Điểm'},
            color_discrete_map=dict(zip(
                df['sentiment'].unique(),
                df.drop_duplicates('sentiment').set_index('sentiment')['sentiment_color'].tolist()
            ))
        )
        fig2.update_layout(height=400, xaxis_tickangle=-45)
    
    # 3. Scatter plot polarity vs subjectivity
    fig3 = px.scatter(
        df,
        x='polarity',
        y='subjectivity',
        color='sentiment',
        size='score',
        hover_data=['author', 'text'],
        title='Phân bố cảm xúc (Polarity vs Subjectivity)',
        labels={'polarity': 'Cực tính', 'subjectivity': 'Chủ quan'},
        color_discrete_map=dict(zip(
            df['sentiment'].unique(),
            df.drop_duplicates('sentiment').set_index('sentiment')['sentiment_color'].tolist()
        ))
    )
    fig3.update_layout(height=500)
    
    return fig1, fig2, fig3

# ==========================================
# 4. PAGE CONTROLLERS - GIAO DIỆN ĐƠN GIẢN
# ==========================================

def login_page():
    ui.render_login_screen()
    t1, t2 = st.tabs(["Đăng nhập", "Đăng ký"])
    with t1:
        with st.form("login"):
            u = st.text_input("Tên đăng nhập")
            p = st.text_input("Mật khẩu", type="password")
            if st.form_submit_button("Đăng nhập", use_container_width=True):
                user = db.login(u, p)
                if user:
                    st.session_state.user = {"id": user[0], "username": user[1]}
                    st.session_state.authenticated = True
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else: 
                    st.error("Sai tên đăng nhập hoặc mật khẩu")
    with t2:
        with st.form("reg"):
            u = st.text_input("Tên người dùng mới")
            p = st.text_input("Mật khẩu mới", type="password")
            if st.form_submit_button("Đăng ký", use_container_width=True):
                if len(u) < 3:
                    st.error("Tên người dùng phải có ít nhất 3 ký tự")
                elif len(p) < 6:
                    st.error("Mật khẩu phải có ít nhất 6 ký tự")
                elif db.register(u, p): 
                    st.success("Đăng ký thành công! Hãy đăng nhập.")
                else: 
                    st.error("Tên người dùng đã tồn tại")

def dashboard_page():
    user = st.session_state.user
    history = db.get_history(user['id'])
    ui.render_dashboard_header(user['username'])
    
    # Thống kê nhanh
    groups = db.get_groups(user['id'])
    st.markdown(f"### 📋 Tổng quan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Nhóm theo dõi", len(groups))
    with col2:
        st.metric("📝 Lịch sử phân tích", len(history))
    with col3:
        last_update = st.session_state.get('last_update')
        if isinstance(last_update, datetime):
            st.metric("🔄 Cập nhật cuối", last_update.strftime('%H:%M'))
        else:
            st.metric("🔄 Cập nhật cuối", "Chưa có")
    
    st.divider()
    
    # Feature cards
    st.markdown("### 🚀 Tính năng chính")
    c1, c2 = st.columns(2)
    with c1: 
        ui.render_feature_card(
            "📊", "Phân Tích Xu Hướng", 
            "Phân tích AI chuyên sâu các cộng đồng Reddit.", 
            "btn_tr", "Khám Phá Ngay", 
            lambda: (setattr(st.session_state, 'page', 'Trending'), st.rerun())
        )
    with c2: 
        ui.render_feature_card(
            "🔗", "Phân Tích Bài Viết", 
            "Phân tích chi tiết bài viết và bình luận.", 
            "btn_an", "Phân Tích", 
            lambda: (setattr(st.session_state, 'page', 'Analysis'), st.rerun())
        )
    
    st.divider()
    
    # Lịch sử
    if history:
        st.markdown("### 📜 Lịch sử phân tích gần đây")
        ui.render_history_list(history, db.delete_history)
    else:
        st.info("📝 Chưa có lịch sử phân tích nào. Hãy thử phân tích bài viết đầu tiên!")

def analyze_callback(url):
    """Callback khi click phân tích bài viết"""
    if not url or 'reddit.com' not in url:
        st.error("⚠️ URL không hợp lệ. Vui lòng kiểm tra lại.")
        return
    
    # Chuẩn hóa URL
    if not url.startswith('http'):
        url = 'https://' + url
    
    st.session_state.analyze_url = url
    st.session_state.auto_run = True
    st.session_state.page = "Analysis"
    st.rerun()

def trending_page():
    st.markdown("## 📊 Phân Tích Xu Hướng")
    user = st.session_state.user
    groups = db.get_groups(user['id'])
    
    if not groups:
        st.info("💡 Chưa có nhóm theo dõi. Hãy thêm nhóm ở thanh bên trái.")
        return
    
    # Khởi tạo managers
    trend_manager = TrendAnalysisManager()
    trending_manager = TrendingManager()
    
    # Control panel
    st.markdown("### ⚙️ Cài đặt phân tích")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analysis_days = st.selectbox(
            "Thời gian phân tích",
            options=[7, 14, 30],
            index=0,
            help="Số ngày dữ liệu để phân tích"
        )
    with col2:
        posts_limit = st.slider("Số bài viết", 10, 50, 20, help="Số bài viết lấy từ mỗi subreddit")
    
    with col3:
        if st.button("🔄 Cập nhật dữ liệu", type="primary", use_container_width=True):
            with st.spinner("Đang thu thập dữ liệu mới..."):
                subs = [g['subreddit'] for g in groups]
                st.session_state.trending_data = trending_manager.fetch_feed(subs, limit=posts_limit)
                st.session_state.last_update = datetime.now()
                st.success(f"✅ Đã cập nhật {len(st.session_state.trending_data)} bài viết")
                st.rerun()
    
    # Hiển thị thông tin cập nhật
    if st.session_state.get('last_update'):
        st.info(f"📅 Cập nhật lần cuối: {st.session_state.last_update.strftime('%H:%M %d/%m')} | " +
               f"📝 Tổng bài viết: {len(st.session_state.get('trending_data', []))}")

    if not st.session_state.get('trending_data'):
        st.info("👆 Nhấn 'Cập nhật dữ liệu' để bắt đầu phân tích xu hướng")
        return

    # Lọc subreddit
    all_subs = sorted(list(set([p['subreddit'] for p in st.session_state.trending_data])))
    
    if not all_subs:
        st.warning("Không lấy được dữ liệu từ subreddits. Vui lòng thử lại.")
        return
    
    st.markdown("### 🔍 Chọn cộng đồng phân tích")
    selected_subs = st.multiselect(
        "Chọn một hoặc nhiều cộng đồng:",
        options=all_subs,
        default=all_subs[:min(3, len(all_subs))],
        placeholder="Chọn cộng đồng..."
    )
    
    if not selected_subs:
        st.info("🎯 Hãy chọn ít nhất một cộng đồng để phân tích")
        return

    # Tab layout cho multiple subreddits
    st.markdown("### 📈 Kết quả phân tích")
    tabs = st.tabs([f"r/{sub}" for sub in selected_subs])
    
    for idx, sub in enumerate(selected_subs):
        with tabs[idx]:
            # Lọc dữ liệu cho subreddit hiện tại
            sub_posts = [p for p in st.session_state.trending_data if p['subreddit'] == sub]
            
            if not sub_posts:
                st.warning(f"Không có dữ liệu cho r/{sub}")
                continue
            
            # Phân tích xu hướng
            with st.spinner(f"🤖 Đang phân tích r/{sub} ({len(sub_posts)} bài viết)..."):
                analysis_result = trend_manager.analyze_subreddit_trends(
                    subreddit=sub,
                    posts_data=sub_posts,
                    days=analysis_days
                )
            
            # Hiển thị kết quả với đầy đủ đồ thị
            ui.render_trend_analysis(analysis_result)
            
            st.divider()
            
            # Hiển thị bài viết từ subreddit này
            st.markdown(f"### 📝 Bài viết gần đây từ r/{sub}")
            for post in sub_posts[:5]:
                ui.render_trending_card(post, analyze_callback)

def analysis_page():
    st.markdown("## 🔗 Phân Tích Bài Viết")
    
    # URL input đơn giản - không hiển thị fallback options
    url = st.text_input(
        "URL Reddit:",
        value=st.session_state.get('analyze_url', ""),
        placeholder="https://www.reddit.com/r/...",
        help="Dán link bài viết Reddit bất kỳ"
    )
      
    # Kiểm tra URL cơ bản
    url_valid = False
    if url:
        if 'reddit.com' not in url:
            st.warning("⚠️ URL không phải là Reddit. Vui lòng nhập URL Reddit hợp lệ.")
        elif not url.startswith('http'):
            st.warning("⚠️ URL phải bắt đầu với http:// hoặc https://")
        else:
            url_valid = True
    
    auto_run = st.session_state.get('auto_run', False)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True) or (auto_run and url_valid):
            st.session_state.auto_run = False
            run_analysis(url)
    
    with col2:
        if st.button("🔄 Xóa kết quả", type="secondary", use_container_width=True):
            if 'analysis_result' in st.session_state:
                del st.session_state.analysis_result
            st.rerun()

def run_analysis(url):
    """Chạy phân tích bài viết với fallback tự động"""
    with st.status("🔄 Đang phân tích...", expanded=True) as status:
        try:
            loader = RedditLoader()
            ai = AdvancedAIAnalyst()
            
            status.write("📥 **1. Tải dữ liệu từ Reddit...**")
            data, err = loader.fetch_data(url)
            
            if err:
                # THÔNG BÁO LỖI ĐƠN GIẢN
                if "403" in err or "chặn" in err:
                    st.error(f"""
                    🔒 **Không thể truy cập bài viết**
                    
                    Reddit đang chặn truy cập từ server này.
                    Hệ thống đã tự động thử các phương thức thay thế nhưng không thành công.
                    
                    **Đề xuất:**
                    1. Thử lại sau 1-2 phút
                    2. Thử bài viết khác
                    3. Kiểm tra xem bài viết có tồn tại không
                    """)
                else:
                    st.error(f"❌ Lỗi: {err}")
                
                status.update(state="error")
                return
            
            st.success(f"✅ Đã tải thành công: {data['meta']['title'][:80]}...")
            st.info(f"Subreddit: r/{data['meta']['subreddit']} • 👤 {data['meta']['author']} • 👍 {data['meta']['score']}")
            
            status.write("🤖 **2. Xử lý ngôn ngữ tự nhiên...**")
            df = process_nlp(data['comments']) if data['comments'] else pd.DataFrame()
            
            status.write("🧠 **3. Phân tích AI với Gemini 2.0 Flash...**")
            summary = ai.analyze(data['meta'], data['comments'])
            
            status.write("💾 **4. Lưu lịch sử...**")
            db.add_history(st.session_state.user['id'], data['meta']['title'], url)
            
            st.session_state.analysis_result = {
                'meta': data['meta'], 
                'df': df, 
                'summary': summary,
                'url': url,
                'analyzed_at': datetime.now()
            }
            
            status.update(state="complete", label="✅ Phân tích hoàn tất!")
            
        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình phân tích: {str(e)}")
            status.update(state="error")

def display_analysis_results():
    """Hiển thị kết quả phân tích"""
    if not st.session_state.get('analysis_result'):
        return
    
    result = st.session_state.analysis_result
    
    # Tabs cho các phần phân tích
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "📈 Biểu đồ", "💬 Bình luận", "📥 Xuất dữ liệu"])
    
    with tab1:
        display_overview_tab(result)
    
    with tab2:
        display_charts_tab(result)
    
    with tab3:
        display_comments_tab(result)
    
    with tab4:
        display_export_tab(result)

def display_overview_tab(result):
    """Tab tổng quan"""
    # Header với thông tin bài viết
    st.markdown(f"## 📄 {result['meta']['title']}")
    
    # Thông tin cơ bản
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Subreddit", f"r/{result['meta']['subreddit']}")
    with col2:
        st.metric("Điểm", result['meta']['score'])
    with col3:
        st.metric("Bình luận", len(result['df']) if not result['df'].empty else 0)
    with col4:
        ratio = result['meta'].get('upvote_ratio', 0)
        st.metric("Upvote Ratio", f"{ratio:.1%}")
    
    # Thời gian và tác giả
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("✍️ Tác giả", result['meta']['author'])
    with col_info2:
        st.metric("🕐 Đăng bài", result['meta'].get('created_time', 'Không rõ'))
    
    # AI Insight
    st.markdown("### 🤖 Phân Tích AI Chi Tiết")
    st.markdown("---")
    
    if result['summary']:
        st.markdown(result['summary'])
    else:
        st.warning("Không có phân tích AI")

def display_charts_tab(result):
    """Tab biểu đồ"""
    # Biểu đồ cảm xúc
    st.markdown("### 📊 Biểu Đồ Phân Tích Cảm Xúc")
    
    if not result['df'].empty:
        # Tạo visualizations
        try:
            fig1, fig2, fig3 = create_visualization(result['df'])
            
            # Hiển thị biểu đồ
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_chart2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Scatter plot
            st.plotly_chart(fig3, use_container_width=True)
            
        except Exception as e:
            st.error(f"Lỗi khi tạo biểu đồ: {e}")
            
            # Fallback: hiển thị bar chart đơn giản
            sentiment_counts = result['df']['sentiment'].value_counts()
            if not sentiment_counts.empty:
                st.bar_chart(sentiment_counts)
    else:
        st.info("Không có dữ liệu bình luận để hiển thị biểu đồ")

def display_comments_tab(result):
    """Tab bình luận"""
    # Bình luận chi tiết
    st.markdown("### 💬 Phân Tích Bình Luận Chi Tiết")
    
    if not result['df'].empty:
        # Filter và sắp xếp
        st.markdown("#### 🔍 Lọc bình luận")
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            sentiment_filter = st.multiselect(
                "Lọc theo cảm xúc",
                options=result['df']['sentiment'].unique(),
                default=result['df']['sentiment'].unique()[:3]
            )
        
        with col_filter2:
            sort_by = st.selectbox(
                "Sắp xếp theo",
                options=['score', 'polarity', 'word_count'],
                index=0
            )
        
        # Lọc dữ liệu
        filtered_df = result['df']
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
        
        # Sắp xếp
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
        
        # Hiển thị bình luận
        st.markdown(f"#### 📝 Bình luận ({len(filtered_df)}/{len(result['df'])})")
        
        for idx, row in filtered_df.head(20).iterrows():
            with st.container():
                col_comment1, col_comment2 = st.columns([4, 1])
                
                with col_comment1:
                    st.markdown(f"**{row['sentiment_emoji']} {row['sentiment']}** • 👤 {row['author']}")
                    st.markdown(f"> {row['text']}")
                
                with col_comment2:
                    st.metric("Điểm", row['score'])
                    st.caption(f"Polarity: {row['polarity']:.3f}")
                
                st.divider()
        
        # Thống kê
        st.markdown("#### 📈 Thống kê bình luận")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            avg_polarity = filtered_df['polarity'].mean()
            st.metric("🎭 Độ cực tính TB", f"{avg_polarity:.3f}")
        
        with col_stats2:
            avg_score = filtered_df['score'].mean()
            st.metric("⭐ Điểm TB", f"{avg_score:.1f}")
        
        with col_stats3:
            total_words = filtered_df['word_count'].sum()
            st.metric("📝 Tổng số từ", total_words)
        
    else:
        st.info("Không có bình luận để phân tích")

def display_export_tab(result):
    """Tab xuất dữ liệu"""
    # Xuất dữ liệu
    st.markdown("### 📥 Xuất Dữ Liệu Phân Tích")
    
    if not result['df'].empty:
        # Tạo DataFrame cho export
        export_df = result['df'].copy()
        
        # Thêm thông tin bài viết
        export_df['post_title'] = result['meta']['title']
        export_df['post_subreddit'] = result['meta']['subreddit']
        export_df['post_score'] = result['meta']['score']
        export_df['post_author'] = result['meta']['author']
        export_df['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Hiển thị preview
        st.markdown("#### 👁️ Preview dữ liệu")
        st.dataframe(export_df.head(10), use_container_width=True)
        
        # Download options
        st.markdown("#### 💾 Tải xuống")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            # CSV
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f"reddit_analysis_{result['meta']['subreddit']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            # Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Sentiment_Analysis')
            
            st.download_button(
                label="📊 Tải Excel",
                data=excel_buffer.getvalue(),
                file_name=f"reddit_analysis_{result['meta']['subreddit']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col_dl3:
            # JSON
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Tải JSON",
                data=json_data,
                file_name=f"reddit_analysis_{result['meta']['subreddit']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Thống kê export
        st.markdown("---")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("📊 Số dòng dữ liệu", len(export_df))
        with col_info2:
            st.metric("📈 Số cột dữ liệu", len(export_df.columns))
        
    else:
        st.info("Không có dữ liệu để xuất")
    
    # Link về bài viết gốc
    st.markdown("---")
    st.markdown(f"**🔗 Link bài viết gốc:** [{result['meta']['title'][:50]}...]({result.get('url', '#')})")

# --- MAIN ---
def main():
    # 1. Khởi tạo Session State
    default_state = {
        'authenticated': False,
        'user': None,
        'page': "Dashboard",
        'trending_data': [],
        'analyze_url': "",
        'auto_run': False,
        'analysis_result': None,
        'last_update': None
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 2. Load UI
    ui.load_css()
    
    # 3. Routing
    if not st.session_state.authenticated:
        login_page()
    else:
        user = st.session_state.user
        groups = db.get_groups(user['id'])
        
        # Render sidebar
        ui.render_sidebar_logged_in(
            user['username'], 
            groups,
            lambda: (setattr(st.session_state, 'authenticated', False), st.rerun()), 
            lambda sub: (db.add_group(user['id'], sub), st.rerun()), 
            lambda gid: (db.delete_group(gid), st.rerun())
        )
        
        # Page routing
        if st.session_state.page == "Dashboard":
            dashboard_page()
        elif st.session_state.page == "Trending":
            trending_page()
        elif st.session_state.page == "Analysis":
            analysis_page()
            display_analysis_results()

# Chạy ứng dụng
if __name__ == "__main__":
    main()