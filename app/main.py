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
import feedparser 

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

# Import Auth (Safe Mode)
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
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px; font-weight: bold;}
    .metric-card { background: #262730; padding: 15px; border-radius: 10px; border: 1px solid #4A5568; text-align: center; }
    .comment-card { background: #262730; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #FF512F; color: #E0E0E0; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    div.row-widget.stRadio > div { flex-direction: row; align-items: stretch; }
    div.row-widget.stRadio > div[role="radiogroup"] > label { background: #262730; border: 1px solid #4A5568; padding: 10px; flex: 1; text-align: center; border-radius: 8px; margin: 0 5px; }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] { background: #FF512F; border-color: #FF512F; color: white; }
</style>
""", unsafe_allow_html=True)

# --- HELPER ---
def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

def analyze_post_callback(url):
    st.session_state.trending_analysis_url = url
    st.session_state.trending_analysis_triggered = True
    st.session_state.active_tab = "🔗 Single Analysis"

# --- CORE 1: REDDIT LOADER (LOCAL JSON - DIRECT REDDIT PRIORITY) ---
class RedditLoader:
    def __init__(self):
        # DANH SÁCH SERVER MỚI NHẤT (Ưu tiên reddit.com cho Local)
        self.mirrors = [
            "https://www.reddit.com",       # Chính chủ (Ngon nhất cho Local)
            "https://redlib.vling.moe",     # Mirror 1 (Ổn định)
            "https://r.fxy.net",            # Mirror 2
            "https://snoo.habedieeh.re"     # Mirror 3
        ]
        # Header giả lập Chrome mới nhất
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
        }

    def fetch(self, url):
        # 1. Xử lý URL: Lấy path
        try:
            if "reddit.com" in url:
                path = url.split("reddit.com")[1].split('?')[0]
            elif "redd.it" in url:
                return {'success': False, 'error': 'Vui lòng dùng link đầy đủ.'}
            else:
                # Xử lý link mirror
                path = "/" + "/".join(url.split("/")[3:]) if url.startswith("http") else url
            
            path = path.rstrip('/') + ".json"
        except:
            return {'success': False, 'error': 'Link không hợp lệ.'}

        # 2. Thử lần lượt (Retry Logic)
        last_error = ""
        for domain in self.mirrors:
            target_url = domain + path
            try:
                # Timeout ngắn để chuyển server nhanh nếu chết
                response = requests.get(target_url, headers=self.headers, timeout=8)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) >= 2:
                        post_data = data[0]['data']['children'][0]['data']
                        comments_data = data[1]['data']['children']
                        
                        comments = []
                        for c in comments_data:
                            if 'data' in c and 'body' in c['data']:
                                d = c['data']
                                ts = d.get('created_utc', time.time())
                                comments.append({
                                    'body': d.get('body', ''),
                                    'author': d.get('author', 'Unknown'),
                                    'score': d.get('score', 0),
                                    'created_utc': ts,
                                    'timestamp': datetime.fromtimestamp(ts),
                                    'permalink': f"https://www.reddit.com{d.get('permalink','')}"
                                })
                        
                        return {
                            'success': True,
                            'meta': {
                                'title': post_data.get('title'),
                                'subreddit': post_data.get('subreddit'),
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'author': post_data.get('author'),
                                'created': datetime.fromtimestamp(post_data.get('created_utc')),
                                'url': f"https://www.reddit.com{post_data.get('permalink')}",
                                'selftext': post_data.get('selftext', '')
                            },
                            'comments': comments
                        }
                else:
                    last_error = f"HTTP {response.status_code} tại {domain}"
                    continue
            except Exception as e:
                last_error = str(e)
                continue

        return {'success': False, 'error': f'Không thể tải dữ liệu. Lỗi cuối: {last_error}'}

# --- CORE 2: TRENDING MANAGER (LOCAL JSON) ---
class TrendingPostsManager:
    def __init__(self):
        # Danh sách mirror cho trending
        self.mirrors = [
            "https://www.reddit.com", 
            "https://redlib.vling.moe",
            "https://r.fxy.net"
        ]
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}

    def fetch_trending_posts(self, subreddit, limit=5):
        for domain in self.mirrors:
            url = f"{domain}/r/{subreddit}/hot.json?limit={limit}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = []
                    
                    # Check structure
                    if 'data' not in data or 'children' not in data['data']:
                        continue

                    for child in data['data']['children']:
                        p = child['data']
                        thumb = ''
                        if 'thumbnail' in p and p['thumbnail'].startswith('http'):
                            thumb = p['thumbnail']
                        elif 'preview' in p and 'images' in p['preview']:
                            thumb = p['preview']['images'][0]['source']['url'].replace('&amp;', '&')

                        posts.append({
                            'id': p['id'],
                            'title': p['title'],
                            'author': p['author'],
                            'score': p['score'],             # INT
                            'comments_count': p['num_comments'], # INT
                            'created_utc': p['created_utc'],
                            'url': f"https://www.reddit.com{p['permalink']}",
                            'subreddit': subreddit,
                            'thumbnail': thumb
                        })
                    return posts
            except:
                continue
        return []

    def fetch_multiple_subreddits(self, subreddits, limit_per_sub=5):
        all_posts = []
        bar = st.progress(0, text="Đang tải...")
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
            if sub not in stats: stats[sub] = {'count': 0, 'total_score': 0, 'total_comments': 0}
            stats[sub]['count'] += 1
            stats[sub]['total_score'] += p['score']
            stats[sub]['total_comments'] += p['comments_count']
        return stats

# --- CORE 3: AI SUMMARIZER (1.5 FLASH PRIORITY) ---
class AISummarizer:
    def __init__(self):
        self.api_key = GOOGLE_GEMINI_API_KEY
        if GEMINI_AVAILABLE and self.api_key:
            try: genai.configure(api_key=self.api_key)
            except: pass

    def generate_summary(self, title, body, comments):
        if not GEMINI_AVAILABLE: return "⚠️ Chưa cài thư viện Google AI."
        if not self.api_key: return "⚠️ Chưa cấu hình API Key."

        cmts = "\n".join([f"- {c['body'][:200]}..." for c in comments[:15]])
        prompt = f"""
        Tóm tắt bài Reddit sau (Tiếng Việt):
        Title: {title}
        Body: {body[:1000]}...
        Comments: {cmts}
        
        Output Markdown:
        1. **Vấn đề chính:** (Ngắn gọn).
        2. **Phản ứng:** Đồng tình/Phản đối.
        3. **Góc nhìn:** 3 điểm chính.
        """
        
        models = ['gemini-1.5-flash','gemini-2.0-flash', 'gemini-pro']
        err = ""
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                return f"**⚡ Phân tích bởi {m}:**\n\n{model.generate_content(prompt).text}"
            except Exception as e:
                err = str(e)
                continue
        return f"⚠️ Lỗi AI: {err}"

# --- CORE 4: NLP ENGINE ---
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

# --- CORE 5: VIZ ENGINE (SAFE) ---
class EnhancedVizEngine:
    @staticmethod
    def plot_sentiment_distribution(df):
        if df.empty: return None
        return px.pie(df, names='sentiment', title="Sentiment", hole=0.5, color='sentiment',
                     color_discrete_map={'Positive':'#00CC96','Negative':'#FF512F','Neutral':'#FECB52'})

    @staticmethod
    def plot_emotion_radar(df):
        ems = [e for sub in df['emotions'] for e in sub if e != 'Neutral']
        if not ems: return None
        counts = Counter(ems)
        return px.line_polar(r=list(counts.values()), theta=list(counts.keys()), line_close=True, title="Emotion Radar")

    @staticmethod
    def plot_sentiment_timeline(df):
        if len(df) < 2 or 'timestamp' not in df.columns: return None
        df = df.sort_values('timestamp')
        return px.scatter(df, x='timestamp', y='polarity', color='sentiment', title="Timeline")

# --- UI COMPONENTS ---
def show_auth_section():
    if not st.session_state.authenticated:
        st.markdown("<div class='auth-section'>", unsafe_allow_html=True)
        st.info("Chế độ Local (Không cần DB)")
        if st.button("🚀 Vào App Ngay", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {"username": "LocalUser", "email": "admin@local", "id": 1}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success(f"Chào {st.session_state.user['username']}!")
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

def show_trending_posts():
    st.markdown("### 🔥 Xu Hướng (Local JSON)")
    c1, c2 = st.columns([3, 1])
    with c1:
        subs = st.text_input("Subreddits", "technology, python, artificial")
    with c2:
        limit = st.slider("Số lượng", 3, 10, 5)

    if st.button("Tải dữ liệu", use_container_width=True):
        tm = TrendingPostsManager()
        with st.spinner("Đang tải..."):
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
                    st.caption(f"r/{p['subreddit']} • ⬆️ {p['score']} • 💬 {p['comments_count']}")
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
    
    total_score = sum(p['score'] for p in posts)
    total_cmt = sum(p['comments_count'] for p in posts)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng bài viết", len(posts))
    c2.metric("Tổng Upvote", f"{total_score:,}")
    c3.metric("Tổng Bình luận", f"{total_cmt:,}")
    
    if stats:
        data = [{'Sub': k, 'Upvotes': v['total_score']} for k, v in stats.items()]
        st.plotly_chart(px.bar(data, x='Sub', y='Upvotes', title="So sánh tương tác"), use_container_width=True)

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
        s.update(label="Tải JSON (Local)...")
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
    st.markdown(f"### 📄 {data['meta']['title']}")
    viz = EnhancedVizEngine()
    
    t1, t2, t3, t4 = st.tabs(["AI Tóm tắt", "Cảm xúc", "Radar", "Dữ liệu"])
    
    with t1:
        st.markdown("### 🤖 Báo cáo AI")
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
        st.info("Vui lòng đăng nhập (Click nút 'Vào App Ngay').")
        return

    st.markdown('<div class="main-header"><h1>🧠 Analytics Pro</h1><p>Local JSON • Gemini 1.5 • Secure</p></div>', unsafe_allow_html=True)
    
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