# app/main.py
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
from textblob import TextBlob
from collections import Counter

# --- 1. SETUP ENVIRONMENT ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI Module
try:
    from visualizations import ui
except ImportError:
    st.error("⚠️ Lỗi: Không tìm thấy file 'app/visualizations/ui.py'. Hãy tạo file UI trước.")
    st.stop()

# Import TrendAnalysisService
try:
    from services.trend_service import TrendAnalysisService
    TREND_SERVICE_AVAILABLE = True
except ImportError:
    TREND_SERVICE_AVAILABLE = False
    st.warning("⚠️ TrendAnalysisService không khả dụng. Chạy với tính năng cơ bản.")

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
# 2. DATABASE MANAGER (SQLITE - LOCAL)
# ==========================================
class DBManager:
    def __init__(self, db_name="reddit_insider.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
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
        c = self.conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            self.conn.commit()
            return True
        except: return False

    def login(self, username, password):
        c = self.conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        c.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, hashed))
        return c.fetchone()

    def add_group(self, user_id, subreddit):
        c = self.conn.cursor()
        clean_sub = subreddit.replace('r/', '').replace('/', '').strip()
        if not clean_sub: return False
        c.execute("SELECT id FROM groups WHERE user_id=? AND subreddit=?", (user_id, clean_sub))
        if not c.fetchone():
            c.execute("INSERT INTO groups (user_id, subreddit) VALUES (?, ?)", (user_id, clean_sub))
            self.conn.commit()
            return True
        return False

    def get_groups(self, user_id):
        c = self.conn.cursor()
        c.execute("SELECT id, subreddit FROM groups WHERE user_id=?", (user_id,))
        return [{'id': r[0], 'subreddit': r[1]} for r in c.fetchall()]

    def delete_group(self, group_id):
        c = self.conn.cursor()
        c.execute("DELETE FROM groups WHERE id=?", (group_id,))
        self.conn.commit()

    def add_history(self, user_id, title, url):
        c = self.conn.cursor()
        ts = datetime.now().strftime("%d/%m %H:%M")
        c.execute("SELECT id FROM history WHERE user_id=? AND url=? ORDER BY id DESC LIMIT 1", (user_id, url))
        if not c.fetchone():
            c.execute("INSERT INTO history (user_id, title, url, timestamp) VALUES (?, ?, ?, ?)", 
                      (user_id, title, url, ts))
            self.conn.commit()

    def get_history(self, user_id):
        c = self.conn.cursor()
        c.execute("SELECT id, title, url, timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20", (user_id,))
        return [{'id': r[0], 'title': r[1], 'url': r[2], 'timestamp': r[3]} for r in c.fetchall()]

    def delete_history(self, hist_id):
        c = self.conn.cursor()
        c.execute("DELETE FROM history WHERE id=?", (hist_id,))
        self.conn.commit()

    def cache_trend_data(self, subreddit, data):
        """Cache kết quả phân tích trend để tăng tốc độ"""
        c = self.conn.cursor()
        ts = datetime.now().isoformat()
        try:
            c.execute(
                "INSERT OR REPLACE INTO trend_cache (subreddit, data, last_updated) VALUES (?, ?, ?)",
                (subreddit, data, ts)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Cache error: {e}")

    def get_cached_trend_data(self, subreddit, max_age_minutes=30):
        """Lấy dữ liệu trend từ cache nếu còn mới"""
        c = self.conn.cursor()
        cutoff_time = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()
        c.execute(
            "SELECT data FROM trend_cache WHERE subreddit=? AND last_updated > ?",
            (subreddit, cutoff_time)
        )
        result = c.fetchone()
        return result[0] if result else None

db = DBManager()

# ==========================================
# 3. CORE LOGIC
# ==========================================

class RedditLoader:
    def __init__(self):
        self.mirrors = ["https://r.fxy.net", "https://l.opnxng.com", "https://www.reddit.com"]
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36'}

    def clean_html(self, raw):
        return re.sub(re.compile('<.*?>'), '', raw).strip() if raw else ""

    def fetch_data(self, url):
        try:
            if "reddit.com" in url: path = url.split("reddit.com")[1].split('?')[0]
            elif url.startswith("http"): path = "/" + "/".join(url.split("/")[3:])
            else: path = url
            path = path.rstrip('/')
        except: return None, "Link không hợp lệ."

        for domain in self.mirrors:
            target = f"{domain}{path}.json"
            try:
                resp = requests.get(target, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    return self._parse_json(resp.json(), path)
            except: continue
        
        return self._fetch_rss(path)

    def _parse_json(self, data, path):
        try:
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
                        'timestamp': datetime.fromtimestamp(ts),
                        'link': f"https://www.reddit.com{d.get('permalink','')}"
                    })
            
            meta = {
                'title': post_data.get('title'),
                'subreddit': post_data.get('subreddit'),
                'score': post_data.get('score', 0),
                'count': post_data.get('num_comments', 0),
                'author': post_data.get('author'),
                'content': post_data.get('selftext', ''),
                'url': f"https://www.reddit.com{path}",
                'source': 'JSON'
            }
            return {'meta': meta, 'comments': comments}, None
        except: return None, "Lỗi cấu trúc JSON."

    def _fetch_rss(self, path):
        rss_url = f"https://www.reddit.com{path}.rss"
        try:
            resp = requests.get(rss_url, headers=self.headers, timeout=10)
            feed = feedparser.parse(resp.content)
            if not feed.entries: return None, "Không tìm thấy dữ liệu."
            
            post = feed.entries[0]
            comments = []
            for entry in feed.entries[1:]:
                ts = time.mktime(entry.updated_parsed)
                comments.append({
                    'body': self.clean_html(entry.content[0].value),
                    'author': entry.author,
                    'score': 0,
                    'timestamp': datetime.fromtimestamp(ts),
                    'link': entry.link
                })
            
            meta = {
                'title': post.title,
                'subreddit': feed.feed.get('subtitle', 'Reddit').replace('r/', ''),
                'score': 0, 'count': len(comments), 'author': post.author,
                'content': self.clean_html(post.content[0].value),
                'url': post.link, 'source': 'RSS'
            }
            return {'meta': meta, 'comments': comments}, None
        except Exception as e: return None, str(e)

class TrendingManager:
    def __init__(self):
        self.loader = RedditLoader()
        
    def fetch_feed(self, subreddits, limit=15):
        """Lấy dữ liệu bài viết từ subreddits"""
        results = []
        progress_bar = st.progress(0)
        
        for i, sub in enumerate(subreddits):
            success = False
            # Thử các mirror
            for domain in self.loader.mirrors:
                try:
                    url = f"{domain}/r/{sub.strip()}/hot.json?limit={limit}"
                    resp = requests.get(url, headers=self.loader.headers, timeout=8)
                    if resp.status_code == 200:
                        data = resp.json()
                        posts = self._parse_posts(data['data']['children'], sub.strip())
                        results.extend(posts)
                        success = True
                        break
                except Exception as e:
                    continue
            
            # Fallback RSS
            if not success:
                try:
                    posts = self._fetch_rss_feed(sub.strip(), limit)
                    results.extend(posts)
                except:
                    pass
            
            progress_bar.progress((i + 1) / len(subreddits))
        
        progress_bar.empty()
        
        # Sắp xếp theo thời gian
        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return results

    def _parse_posts(self, posts_data, subreddit):
        """Parse dữ liệu bài viết từ JSON response"""
        posts = []
        for child in posts_data:
            p = child['data']
            try:
                # Xử lý thumbnail
                thumb = None
                if p.get('thumbnail') and p['thumbnail'].startswith('http'):
                    thumb = p['thumbnail']
                elif p.get('preview'):
                    try:
                        thumb = p['preview']['images'][0]['source']['url'].replace('&amp;', '&')
                    except:
                        pass
                
                # Xử lý media
                media_url = None
                if p.get('url_overridden_by_dest'):
                    media_url = p['url_overridden_by_dest']
                
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
                    'media_url': media_url,
                    'selftext': p.get('selftext', ''),
                    'upvote_ratio': p.get('upvote_ratio', 0),
                    'time_str': datetime.fromtimestamp(p.get('created_utc', time.time())).strftime('%H:%M %d/%m')
                }
                posts.append(post)
            except Exception as e:
                continue
        return posts

    def _fetch_rss_feed(self, subreddit, limit):
        """Fallback sử dụng RSS feed"""
        try:
            feed = feedparser.parse(f"https://www.reddit.com/r/{subreddit}/hot.rss")
            posts = []
            for entry in feed.entries[:limit]:
                post = {
                    'id': entry.id.split('/')[-1],
                    'title': entry.title,
                    'url': entry.link,
                    'subreddit': subreddit,
                    'author': entry.author,
                    'score': 0,
                    'comments_count': 0,
                    'created_utc': time.mktime(entry.updated_parsed),
                    'timestamp': time.mktime(entry.updated_parsed),
                    'thumbnail': None,
                    'media_url': None,
                    'selftext': '',
                    'upvote_ratio': 0,
                    'time_str': datetime.fromtimestamp(time.mktime(entry.updated_parsed)).strftime('%H:%M %d/%m')
                }
                posts.append(post)
            return posts
        except:
            return []

class TrendAnalysisManager:
    def __init__(self):
        self.trend_service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Khởi tạo TrendAnalysisService với error handling"""
        if not TREND_SERVICE_AVAILABLE:
            return
            
        try:
            self.trend_service = TrendAnalysisService(
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                min_posts_for_analysis=5
            )
        except Exception as e:
            st.error(f"❌ Không thể khởi tạo TrendAnalysisService: {e}")
            self.trend_service = None

    def analyze_subreddit_trends(self, subreddit, posts_data, days=7):
        """Phân tích xu hướng cho một subreddit"""
        if not self.trend_service:
            return {'error': 'service_unavailable', 'message': 'Trend analysis service không khả dụng'}
        
        try:
            # Kiểm tra cache trước
            cached_data = db.get_cached_trend_data(subreddit)
            if cached_data:
                return eval(cached_data)  # Chuyển string thành dict
            
            # Phân tích với service
            with st.spinner(f"🤖 Đang phân tích xu hướng cho r/{subreddit}..."):
                result = self.trend_service.analyze_subreddit(
                    subreddit=subreddit,
                    posts_data=posts_data,
                    days=days,
                    topic_top_n=5,
                    kw_top_n=10
                )
            
            # Cache kết quả
            if 'error' not in result:
                db.cache_trend_data(subreddit, str(result))
            
            return result
            
        except Exception as e:
            return {'error': 'analysis_failed', 'reason': str(e)}

class LegacyTrendAnalyzer:
    """Fallback analyzer khi TrendAnalysisService không khả dụng"""
    
    def __init__(self):
        self.stopwords = set([
            'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'is', 'are', 'and', 'of', 'with', 
            'i', 'it', 'this', 'that', 'my', 'your', 'have', 'has', 'do', 'can', 'be', 'but',
            'là', 'và', 'của', 'những', 'các', 'trong', 'khi', 'cho', 'để', 'với', 'có', 'người'
        ])

    def basic_analysis(self, posts_data, subreddit):
        """Phân tích cơ bản khi không có service chính"""
        if not posts_data:
            return {'error': 'no_data', 'message': 'Không có dữ liệu để phân tích'}
        
        df = pd.DataFrame(posts_data)
        
        # Tính metrics cơ bản
        total_posts = len(df)
        total_score = df['score'].sum()
        total_comments = df['comments_count'].sum()
        avg_score = df['score'].mean()
        
        # Phân tích giờ cao điểm
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        hourly = df.groupby('hour').agg({
            'score': 'sum',
            'comments_count': 'sum'
        }).reset_index()
        hourly['total_engagement'] = hourly['score'] + hourly['comments_count'] * 2
        
        peak_hours = []
        for _, row in hourly.iterrows():
            peak_hours.append({
                'hour': int(row['hour']),
                'total_engagement': int(row['total_engagement']),
                'post_count': len(df[df['hour'] == row['hour']])
            })
        
        # Từ khóa đơn giản
        all_titles = " ".join(df['title'].astype(str).tolist()).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_titles)
        filtered_words = [w for w in words if w not in self.stopwords]
        word_counts = Counter(filtered_words).most_common(10)
        
        keywords = [{'keyword': w, 'score': c/len(filtered_words)} for w, c in word_counts]
        
        return {
            'subreddit': subreddit,
            'analysis_period_days': 7,
            'data_summary': {
                'total_posts_analyzed': total_posts,
                'total_score': int(total_score),
                'total_engagement': int(total_score + total_comments * 2),
                'total_comments': int(total_comments),
                'avg_score_per_post': float(avg_score),
                'avg_comments_per_post': float(df['comments_count'].mean()),
                'avg_engagement_per_post': float((total_score + total_comments * 2) / total_posts)
            },
            'peak_hours': peak_hours,
            'top_keywords': keywords[:8],
            'top_topics': [],
            'forecast': {'error': 'advanced_analysis_unavailable'},
            'analysis_timestamp': datetime.now().isoformat()
        }

class AIAnalyst:
    def __init__(self):
        self.key = GOOGLE_GEMINI_API_KEY
        
    def analyze(self, meta, comments):
        if not self.key or not GEMINI_AVAILABLE: 
            return "⚠️ AI chưa sẵn sàng. Vui lòng cấu hình API key."
        
        try:
            genai.configure(api_key=self.key)
            cmts = "\n".join([f"- {c['body'][:150]}" for c in comments[:20]])
            prompt = f"""
            Phân tích bài Reddit: {meta['title']}
            Nội dung: {meta['content'][:800]}
            Comment: {cmts}
            
            Output Markdown:
            ### 🎯 Tóm tắt
            ### 🌊 Phản ứng
            ### 💡 Insight
            """
            models = ['gemini-1.5-flash','gemini-2.0-flash', 'gemini-pro']
            for m in models:
                try: 
                    return genai.GenerativeModel(m).generate_content(prompt).text
                except: continue
            return "⚠️ AI quá tải."
        except Exception as e:
            return f"⚠️ Lỗi AI: {str(e)}"

def process_nlp(comments):
    """Xử lý NLP cơ bản cho comments"""
    data = []
    for c in comments:
        blob = TextBlob(c['body'])
        pol = blob.sentiment.polarity
        sent = 'Tích cực' if pol > 0.1 else 'Tiêu cực' if pol < -0.1 else 'Trung lập'
        txt = c['body'].lower()
        emo = 'Bình thường'
        if any(x in txt for x in ['love','good', 'great', 'awesome']): emo = 'Vui vẻ'
        elif any(x in txt for x in ['hate','bad', 'terrible', 'awful']): emo = 'Giận dữ'
        data.append({**c, 'polarity': pol, 'sentiment': sent, 'emotion': emo})
    return pd.DataFrame(data)

# ==========================================
# 4. PAGE CONTROLLERS
# ==========================================

def login_page():
    ui.render_login_screen()
    t1, t2 = st.tabs(["Đăng nhập", "Đăng ký"])
    with t1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                user = db.login(u, p)
                if user:
                    st.session_state.user = {"id": user[0], "username": user[1]}
                    st.session_state.authenticated = True
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else: st.error("Sai thông tin")
    with t2:
        with st.form("reg"):
            u = st.text_input("New User")
            p = st.text_input("New Pass", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                if db.register(u, p): st.success("Đăng ký thành công!")
                else: st.error("Tên đã tồn tại")

def dashboard_page():
    user = st.session_state.user
    history = db.get_history(user['id'])
    ui.render_dashboard_header(user['username'])
    
    # Feature cards
    c1, c2 = st.columns(2)
    with c1: 
        ui.render_feature_card(
            "📊", "Phân Tích Xu Hướng", 
            "Phân tích AI chuyên sâu các cộng đồng.", 
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
    ui.render_history_list(history, db.delete_history)

def trending_page():
    st.markdown("## 📊 Phân Tích Xu Hướng")
    user = st.session_state.user
    groups = db.get_groups(user['id'])
    
    if not groups:
        st.info("💡 Chưa có nhóm theo dõi. Hãy thêm nhóm ở thanh bên trái.")
        return
    
    # Khởi tạo managers
    trend_manager = TrendAnalysisManager()
    legacy_analyzer = LegacyTrendAnalyzer()
    trending_manager = TrendingManager()
    
    # Control panel
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analysis_days = st.selectbox(
            "Thời gian phân tích",
            options=[7, 14, 30],
            index=0,
            help="Số ngày dữ liệu để phân tích"
        )
    with col2:
        if st.button("🔄 Cập nhật dữ liệu", type="primary", use_container_width=True):
            with st.spinner("Đang thu thập dữ liệu mới..."):
                subs = [g['subreddit'] for g in groups]
                st.session_state.trending_data = trending_manager.fetch_feed(subs, limit=20)
                st.session_state.last_update = datetime.now()
                st.rerun()
    
    with col3:
        if st.session_state.get('last_update'):
            st.caption(f"📅 Cập nhật lần cuối: {st.session_state.last_update.strftime('%H:%M %d/%m')}")

    # Hiển thị trạng thái service
    if not TREND_SERVICE_AVAILABLE:
        st.warning("""
        ⚠️ **TrendAnalysisService không khả dụng** 
        - Đang chạy ở chế độ cơ bản
        - Cài đặt: `pip install prophet bertopic keybert sentence-transformers`
        """)
    
    if not st.session_state.get('trending_data'):
        st.info("👆 Nhấn 'Cập nhật dữ liệu' để bắt đầu phân tích xu hướng")
        return

    # Lọc subreddit
    all_subs = sorted(list(set([p['subreddit'] for p in st.session_state.trending_data])))
    selected_subs = st.multiselect(
        "🔍 Chọn cộng đồng để phân tích:",
        options=all_subs,
        default=all_subs[:min(3, len(all_subs))],  # Mặc định 3 sub đầu
        placeholder="Chọn cộng đồng..."
    )
    
    if not selected_subs:
        st.info("🎯 Hãy chọn ít nhất một cộng đồng để phân tích")
        return

    # Tab layout cho multiple subreddits
    tabs = st.tabs([f"r/{sub}" for sub in selected_subs])
    
    for idx, sub in enumerate(selected_subs):
        with tabs[idx]:
            # Lọc dữ liệu cho subreddit hiện tại
            sub_posts = [p for p in st.session_state.trending_data if p['subreddit'] == sub]
            
            if not sub_posts:
                st.warning(f"Không có dữ liệu cho r/{sub}")
                continue
            
            # Phân tích xu hướng
            if trend_manager.trend_service:
                # Sử dụng service chính
                analysis_result = trend_manager.analyze_subreddit_trends(
                    subreddit=sub,
                    posts_data=sub_posts,
                    days=analysis_days
                )
            else:
                # Fallback legacy analysis
                analysis_result = legacy_analyzer.basic_analysis(sub_posts, sub)
            
            # Hiển thị kết quả
            if 'error' in analysis_result:
                st.error(f"Lỗi phân tích r/{sub}: {analysis_result.get('message', analysis_result['error'])}")
                if analysis_result.get('reason'):
                    st.code(analysis_result['reason'])
            else:
                ui.render_trend_analysis(analysis_result)
            
            st.divider()
            
            # Hiển thị bài viết từ subreddit này
            st.markdown(f"### 📝 Bài viết gần đây từ r/{sub}")
            for post in sub_posts[:5]:  # Hiển thị 5 bài mới nhất
                ui.render_trending_card(post, analyze_callback)

def analyze_callback(url):
    """Callback khi click phân tích bài viết"""
    st.session_state.analyze_url = url
    st.session_state.auto_run = True
    st.session_state.page = "Analysis"
    st.rerun()

def analysis_page():
    st.markdown("## 🔗 Phân Tích Bài Viết")
    
    url = st.text_input(
        "URL Reddit:",
        value=st.session_state.get('analyze_url', ''),
        placeholder="https://www.reddit.com/r/...",
        help="Dán link bài viết Reddit để phân tích"
    )
    
    auto_run = st.session_state.get('auto_run', False)
    
    if st.button("🚀 Phân tích", type="primary", use_container_width=True) or (auto_run and url):
        st.session_state.auto_run = False
        
        with st.status("🔄 Đang phân tích...", expanded=True) as status:
            try:
                loader = RedditLoader()
                ai = AIAnalyst()
                
                status.write("📥 **1. Tải dữ liệu từ Reddit...**")
                data, err = loader.fetch_data(url)
                if err:
                    st.error(f"❌ Lỗi: {err}")
                    status.update(state="error")
                    return
                
                status.write("🤖 **2. Xử lý ngôn ngữ tự nhiên...**")
                df = process_nlp(data['comments'])
                
                status.write("🧠 **3. Phân tích AI...**")
                summary = ai.analyze(data['meta'], data['comments'])
                
                status.write("💾 **4. Lưu lịch sử...**")
                db.add_history(st.session_state.user['id'], data['meta']['title'], url)
                
                st.session_state.analysis_result = {
                    'meta': data['meta'], 
                    'df': df, 
                    'summary': summary
                }
                
                status.update(state="complete", label="✅ Phân tích hoàn tất!")
                
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình phân tích: {str(e)}")
                status.update(state="error")

    # Hiển thị kết quả
    if st.session_state.get('analysis_result'):
        ui.render_analysis_result_full(st.session_state.analysis_result)

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

if __name__ == "__main__":
    main()