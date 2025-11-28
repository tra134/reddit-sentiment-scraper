# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import time
import sys
import os
import textwrap
import streamlit.components.v1 as components
import html
import feedparser  # QUAN TRỌNG: Dùng thư viện này để lách luật

# --- CẤU HÌNH API KEY ---
try:
    GOOGLE_GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_GEMINI_API_KEY = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- OPTIONAL IMPORTS ---
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import Auth
try:
    from core.user_database import user_db_manager
    from core.auth import authenticate_user, logout
    from services.user_service import UserService
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Reddit Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px; font-weight: bold;}
    .metric-card { background: #262730; padding: 15px; border-radius: 10px; border: 1px solid #4A5568; text-align: center; }
    .comment-card { background: #262730; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #FF416C; color: #E0E0E0; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    div.row-widget.stRadio > div { flex-direction: row; align-items: stretch; }
    div.row-widget.stRadio > div[role="radiogroup"] > label { background: #262730; border: 1px solid #4A5568; padding: 10px; flex: 1; text-align: center; border-radius: 8px; margin: 0 5px; }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] { background: #FF416C; border-color: #FF416C; color: white; }
</style>
""", unsafe_allow_html=True)

# --- HELPER ---
def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

def analyze_post_callback(url):
    st.session_state.trending_analysis_url = url
    st.session_state.trending_analysis_triggered = True
    st.session_state.active_tab = "🔗 Single Analysis"

# --- CORE 1: REDDIT LOADER (ANTI-BLOCK RSS) ---
class RedditLoader:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    def clean_html(self, raw_html):
        if not raw_html: return ""
        cleanr = re.compile('<.*?>')
        return re.sub(cleanr, '', raw_html).strip()

    def fetch(self, url):
        clean_url = url.split('?')[0].rstrip('/')
        if not clean_url.endswith('.rss'): clean_url += '.rss'

        try:
            with st.spinner('📡 Đang kết nối RSS...'):
                # Dùng requests tải trước để vượt qua tường lửa cơ bản
                response = requests.get(clean_url, headers=self.headers, timeout=10)
                if response.status_code != 200: return {'success': False, 'error': f'Lỗi HTTP {response.status_code}'}
                
                feed = feedparser.parse(response.content)
                if not feed.entries: return {'success': False, 'error': 'Không tìm thấy nội dung.'}

                post = feed.entries[0]
                comments = []
                for entry in feed.entries[1:]:
                    ts = time.mktime(entry.updated_parsed) if entry.updated_parsed else time.time()
                    comments.append({
                        'body': self.clean_html(entry.content[0].value if 'content' in entry else entry.summary),
                        'author': entry.author if 'author' in entry else "Unknown",
                        'score': 0,
                        'created_utc': ts,
                        'timestamp': datetime.fromtimestamp(ts),
                        'permalink': entry.link
                    })

                return {
                    'success': True,
                    'meta': {
                        'title': post.title,
                        'subreddit': feed.feed.get('subtitle', 'Reddit').replace('r/', ''),
                        'score': 0, 'num_comments': len(comments),
                        'author': post.author if 'author' in post else "Unknown",
                        'created': datetime.fromtimestamp(time.mktime(post.updated_parsed)) if post.updated_parsed else datetime.now(),
                        'url': post.link, 'permalink': post.link,
                        'selftext': self.clean_html(post.content[0].value if 'content' in post else post.summary)
                    },
                    'comments': comments
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# --- CORE 2: TRENDING MANAGER (PURE RSS FIX) ---
class TrendingPostsManager:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    def fetch_trending_posts(self, subreddit, limit=5):
        # Sử dụng RSS Feed chuẩn của Reddit (Không dùng JSON/Mirror nữa vì hay chết)
        # Thêm ?limit= để báo cho server Reddit biết
        url = f"https://www.reddit.com/r/{subreddit}/hot.rss?limit={limit}"
        try:
            # Requests tải XML về
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200: return []
            
            feed = feedparser.parse(resp.content)
            posts = []
            
            # RSS trả về danh sách bài viết trực tiếp
            for entry in feed.entries[:limit]:
                # Lấy ảnh từ HTML content
                thumb = ''
                content = entry.content[0].value if 'content' in entry else entry.summary
                img = re.search(r'src="([^"]+jpg|[^"]+png)"', content)
                if img: thumb = img.group(1)

                # RSS không có score, ta gán mặc định để không lỗi code
                posts.append({
                    'id': entry.id if 'id' in entry else str(hash(entry.link)),
                    'title': entry.title,
                    'author': entry.author if 'author' in entry else "Unknown",
                    'score': 0, # RSS không có score
                    'comments_count': 0, # RSS không có count
                    'created_utc': time.mktime(entry.updated_parsed) if entry.updated_parsed else time.time(),
                    'url': entry.link, 
                    'subreddit': subreddit, 
                    'thumbnail': thumb
                })
            return posts
        except: return []
    
    def fetch_multiple_subreddits(self, subreddits, limit_per_sub=5):
        all_posts = []
        bar = st.progress(0, text="Đang tải tin tức RSS...")
        for i, sub in enumerate(subreddits):
            posts = self.fetch_trending_posts(sub.strip(), limit=limit_per_sub)
            all_posts.extend(posts)
            bar.progress((i + 1) / len(subreddits))
        bar.empty()
        return all_posts
    
    def analyze_trends(self, posts):
        if not posts: return {}
        stats = {}
        for p in posts:
            sub = p['subreddit']
            if sub not in stats: 
                stats[sub] = {'count': 0, 'posts': [], 'authors': set()}
            stats[sub]['count'] += 1
            stats[sub]['posts'].append(p)
            stats[sub]['authors'].add(p['author'])
        return stats

# --- CORE 3: AI SUMMARIZER (MULTI-MODEL) ---
class AISummarizer:
    def __init__(self):
        self.api_key = GOOGLE_GEMINI_API_KEY
        if GEMINI_AVAILABLE and self.api_key:
            try: genai.configure(api_key=self.api_key)
            except: pass

    def generate_summary(self, title, body, comments):
        if not GEMINI_AVAILABLE or not self.api_key: return "⚠️ Chưa cấu hình AI."
        
        cmts = "\n".join([f"- {c['body'][:200]}..." for c in comments[:15]])
        prompt = f"Tóm tắt bài Reddit (Tiếng Việt):\nTitle: {title}\nBody: {body}\nComments: {cmts}\nOutput: Tóm tắt, Phản ứng, Cảm xúc."
        
        models = ['gemini-1.5-flash','gemini-2.0-flash', 'gemini-pro']
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                return f"**⚡ {m}:**\n\n{model.generate_content(prompt).text}"
            except: continue
        return "⚠️ Lỗi kết nối AI."

# --- CORE 4: NLP ---
class EnhancedNLPEngine:
    def __init__(self):
        self.emotions = {'Vui': {'love', 'good'}, 'Giận': {'hate', 'bad'}, 'Sợ': {'scary'}, 'Buồn': {'sad'}}
    def process_batch(self, comments):
        results = []
        for c in comments:
            pol = TextBlob(c['body']).sentiment.polarity if TEXTBLOB_AVAILABLE else 0
            if pol > 0.1: sent = 'Positive'; em = '😊'
            elif pol < -0.1: sent = 'Negative'; em = '😠'
            else: sent = 'Neutral'; em = '😐'
            
            ems = []
            words = set(re.findall(r'\w+', c['body'].lower()))
            for e, k in self.emotions.items():
                if words.intersection(k): ems.append(e)
            if not ems: ems = ['Neutral']
            
            results.append(c | {'sentiment': sent, 'polarity': pol, 'emoji': em, 'emotions': ems, 'word_count': len(c['body'].split())})
        return results

# --- CORE 5: VIZ (SAFE) ---
class EnhancedVizEngine:
    @staticmethod
    def plot_sentiment_distribution(df):
        if df.empty: return None
        return px.pie(df, names='sentiment', title="Sentiment", hole=0.5, color='sentiment',
                     color_discrete_map={'Positive':'#00CC96','Negative':'#FF416C','Neutral':'#FECB52'})

    @staticmethod
    def plot_emotion_radar(df):
        ems = [e for sub in df['emotions'] for e in sub if e != 'Neutral']
        if not ems: return None
        counts = Counter(ems)
        return px.line_polar(r=list(counts.values()), theta=list(counts.keys()), line_close=True, title="Emotion")

    @staticmethod
    def plot_sentiment_timeline(df):
        if len(df) < 2 or 'timestamp' not in df.columns: return None
        df = df.sort_values('timestamp')
        return px.scatter(df, x='timestamp', y='polarity', color='sentiment', title="Timeline")

# --- UI COMPONENTS ---
def show_auth_section():
    if not st.session_state.authenticated:
        st.markdown("<div class='auth-section'>", unsafe_allow_html=True)
        st.info("Chế độ Demo (Cloud/Local)")
        if st.button("🚀 Đăng nhập Demo", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {"username": "DemoUser", "email": "demo@mail.com", "id": 1}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success(f"Chào {st.session_state.user['username']}!")
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

def show_trending_posts():
    st.markdown("### 🔥 Xu Hướng (RSS)")
    c1, c2 = st.columns([3, 1])
    with c1:
        subs = st.text_input("Subreddits", "technology, python, artificial, vietnam")
    with c2:
        limit = st.slider("Số lượng", 3, 10, 5)

    if st.button("Tải dữ liệu", use_container_width=True):
        tm = TrendingPostsManager()
        st.session_state.trending_data = tm.fetch_multiple_subreddits([s.strip() for s in subs.split(',')], limit_per_sub=limit)
        st.rerun()

    if 'trending_data' in st.session_state and st.session_state.trending_data:
        posts = st.session_state.trending_data
        st.success(f"Đã tải {len(posts)} bài viết.")
        
        for p in posts:
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    if p.get('thumbnail'): st.image(p['thumbnail'], use_container_width=True)
                    else: st.markdown("### 📄")
                with c2:
                    st.markdown(f"**{p['title']}**")
                    # RSS không có số liệu score/comment, chỉ hiện tên
                    st.caption(f"r/{p['subreddit']} • {p['author']}")
                    
                    if st.button("⚡ Phân tích", key=f"btn_{p['id']}"):
                        analyze_post_callback(p['url'])
                        st.rerun()
            st.divider()
    else:
        st.info("Nhấn nút tải để xem tin tức.")

def show_trend_analysis():
    st.markdown("### 📈 Thống Kê Xu Hướng")
    if 'trending_data' not in st.session_state:
        st.warning("Hãy tải dữ liệu bên tab Trending trước.")
        return
    posts = st.session_state.trending_data
    tm = TrendingPostsManager()
    stats = tm.analyze_trends(posts)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng bài viết", len(posts))
    c2.metric("Chủ đề", len(stats))
    c3.metric("Tác giả", len(set(p['author'] for p in posts)))
    
    if stats:
        data = [{'Sub': k, 'Count': v['count']} for k, v in stats.items()]
        st.plotly_chart(px.bar(data, x='Sub', y='Count', title="Số lượng bài viết theo chủ đề"), use_container_width=True)

def show_single_analysis():
    st.markdown("### 🔗 Phân Tích Chi Tiết")
    if st.session_state.get('trending_analysis_triggered'):
        url = st.session_state.trending_analysis_url
        st.session_state.trending_analysis_triggered = False
        perform_analysis(url)
    else:
        url = st.text_input("Link Reddit:", key="u_in")
        if st.button("Chạy"): perform_analysis(url)
        elif st.session_state.get('current_analysis'):
            render_results(st.session_state.current_analysis)

def perform_analysis(url):
    loader = RedditLoader()
    nlp = EnhancedNLPEngine()
    ai = AISummarizer()
    
    with st.status("Đang xử lý...", expanded=True) as s:
        s.update(label="Tải RSS...")
        raw = loader.fetch(url)
        if not raw['success']: 
            s.update(label="Lỗi!", state="error")
            st.error(raw['error'])
            return

        s.update(label="NLP...")
        df = pd.DataFrame(nlp.process_batch(raw['comments']))
        if not df.empty: df = df[df['word_count'] >= 2]

        s.update(label="Gemini AI...")
        summary = ai.generate_summary(raw['meta']['title'], raw['meta']['selftext'], raw['comments'])
        
        st.session_state.current_analysis = {'df': df, 'meta': raw['meta'], 'summary': summary}
        st.session_state.history.append({'title': raw['meta']['title'], 'timestamp': datetime.now(), 'url': url})
        s.update(label="Xong!", state="complete")
    
    render_results(st.session_state.current_analysis)

def render_results(data):
    st.markdown(f"### {data['meta']['title']}")
    viz = EnhancedVizEngine()
    
    t1, t2, t3, t4 = st.tabs(["AI Tóm tắt", "Cảm xúc", "Radar", "Dữ liệu"])
    with t1:
        st.markdown("### 🤖 Báo cáo AI")
        # Fix lỗi div thừa
        st.markdown(f"<div class='comment-card'>\n\n{data['summary']}\n\n</div>", unsafe_allow_html=True)
    with t2:
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.plot_sentiment_distribution(data['df']), use_container_width=True)
        c2.plotly_chart(viz.plot_sentiment_timeline(data['df']), use_container_width=True)
    with t3:
        fig = viz.plot_emotion_radar(data['df'])
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Không đủ dữ liệu cảm xúc.")
    with t4: st.dataframe(data['df'])

def show_dashboard():
    st.markdown("### 📊 Dashboard")
    c1, c2 = st.columns(2)
    if c1.button("🔥 Trending"): switch_tab("Trending"); st.rerun()
    if c2.button("🔗 Single Analysis"): switch_tab("Single"); st.rerun()
    st.subheader("Lịch sử")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-3:]):
            st.text(f"🕒 {h['timestamp'].strftime('%H:%M')} - {h['title']}")

# --- MAIN ---
def main():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if "history" not in st.session_state: st.session_state.history = []
    if "active_tab" not in st.session_state: st.session_state.active_tab = "Dashboard"
    if "trending_analysis_triggered" not in st.session_state: st.session_state.trending_analysis_triggered = False

    with st.sidebar:
        st.title("Reddit AI ⚡")
        show_auth_section()

    if not st.session_state.authenticated:
        st.markdown("<div class='welcome-container'><h1>🧠 Analytics Pro</h1><p>RSS • Gemini • Secure</p></div>", unsafe_allow_html=True)
        st.info("Vui lòng đăng nhập Demo.")
        return

    st.markdown('<div class="main-header"><h1>🧠 Analytics Pro</h1><p>Cloud Ready</p></div>', unsafe_allow_html=True)
    
    opts = ["Dashboard", "Trending", "Analysis", "Single"]
    sel = st.radio("Nav", opts, index=opts.index(st.session_state.active_tab) if st.session_state.active_tab in opts else 0, horizontal=True, label_visibility="collapsed")
    if sel != st.session_state.active_tab: 
        st.session_state.active_tab = sel
        st.rerun()

    st.markdown("---")
    if st.session_state.active_tab == "Dashboard": show_dashboard()
    elif st.session_state.active_tab == "Trending": show_trending_posts()
    elif st.session_state.active_tab == "Analysis": show_trend_analysis()
    elif st.session_state.active_tab == "Single": show_single_analysis()

if __name__ == "__main__":
    main()