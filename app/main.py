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
import feedparser  # THƯ VIỆN QUAN TRỌNG ĐỂ TRÁNH BỊ CHẶN (RSS)

# --- CẤU HÌNH API KEY (QUAN TRỌNG) ---
try:
    GOOGLE_GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_GEMINI_API_KEY = None

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- OPTIONAL IMPORTS ---

# 1. NLP TextBlob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# 2. Google Gemini (Cho tóm tắt thông minh & nhanh)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import authentication modules
# Sử dụng try-except để code không bị crash nếu thiếu file database
try:
    from core.user_database import user_db_manager
    from core.auth import authenticate_user, logout
    from services.user_service import UserService
    AUTH_AVAILABLE = True
except ImportError:
    # print("⚠️ Authentication modules not found. Running in standalone mode.")
    AUTH_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Reddit Analytics Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK THEME & NAVIGATION ---
st.markdown("""
<style>
    /* Styling for the Navigation Radio Button to look like Tabs */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: stretch;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #2D3748;
        padding: 10px 20px;
        border-radius: 10px;
        margin-right: 10px;
        border: 1px solid #4A5568;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        border-color: #667eea;
        background-color: #4A5568;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none; /* Hide the radio circle */
    }

    /* Existing CSS */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid #4A5568;
        color: white;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
    }
    .auth-section {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid #4A5568;
    }
    .feature-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 5px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .welcome-container {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    }
    .group-card {
        background: linear-gradient(135deg, #805AD5 0%, #B794F4 100%);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        color: white;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #805AD5;
    }
    .user-info-card {
        background: linear-gradient(135deg, #3182CE 0%, #63B3ED 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border: 1px solid #3182CE;
    }
    .success-message {
        background: linear-gradient(135deg, #38A169 0%, #68D391 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border: 1px solid #38A169;
    }
    .comment-card {
        background: #2D3748;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        border-left: 5px solid;
        border: 1px solid #4A5568;
        color: #E2E8F0;
        transition: transform 0.2s ease;
    }
    .comment-card:hover {
        transform: translateX(5px);
        border-color: #667eea;
    }
    .trend-card {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        border: 1px solid #4A5568;
        color: #E2E8F0;
        transition: all 0.3s ease;
    }
    .trend-card:hover {
        border-color: #FF6B6B;
        transform: translateX(5px);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
    }
    
    /* Custom styling for better contrast */
    .stTextInput input, .stTextInput input:focus {
        background-color: #2D3748 !important;
        color: #E2E8F0 !important;
        border: 1px solid #4A5568 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2D3748 !important;
        color: #E2E8F0 !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS FOR NAVIGATION ---
def switch_tab(tab_name):
    """Callback to switch tabs safely"""
    st.session_state.active_tab = tab_name

def analyze_post_callback(url):
    """Callback to set up analysis and switch tab"""
    st.session_state.trending_analysis_url = url
    st.session_state.trending_analysis_triggered = True
    st.session_state.active_tab = "🔗 Single Analysis"

# --- CORE FUNCTIONALITY: REDDIT LOADER (RSS VERSION - CHỐNG BLOCK) ---

class RedditLoader:
    def __init__(self):
        # RSS không cần requests session phức tạp
        pass

    def clean_html(self, raw_html):
        """Làm sạch các thẻ HTML từ dữ liệu RSS"""
        if not raw_html: return ""
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext.strip()

    def fetch(self, url):
        # Chuyển đổi URL bài viết sang URL RSS để tránh bị chặn 429
        # VD: reddit.com/r/abc/comments/123/ --> reddit.com/r/abc/comments/123/.rss
        if ".rss" not in url:
            clean_url = url.split('?')[0].rstrip('/') + '.rss'
        else:
            clean_url = url

        try:
            with st.spinner('📡 Đang kết nối qua kênh RSS (An toàn, không chặn IP)...'):
                # Sử dụng feedparser để đọc XML
                feed = feedparser.parse(clean_url)
                
                # Kiểm tra lỗi (Bozo bit)
                if feed.bozo and len(feed.entries) == 0: 
                    return {'success': False, 'error': '❌ Không thể đọc RSS. Link sai hoặc Reddit chặn tạm thời.'}
                
                if not feed.entries:
                    return {'success': False, 'error': '⚠️ Không tìm thấy nội dung. Bài viết có thể đã bị xóa.'}

                # Entry đầu tiên [0] luôn là Post gốc
                post_entry = feed.entries[0]
                
                # Lấy metadata
                post_title = post_entry.title if 'title' in post_entry else "No Title"
                post_author = post_entry.author if 'author' in post_entry else "Unknown"
                post_link = post_entry.link if 'link' in post_entry else url
                subreddit = feed.feed.get('subtitle', 'Reddit').replace('r/', '')
                
                # Nội dung bài viết
                raw_content = post_entry.content[0].value if 'content' in post_entry else post_entry.summary
                post_content = self.clean_html(raw_content)

                # Các entry sau [1:] là Comment
                comments = []
                for entry in feed.entries[1:]:
                    raw_body = entry.content[0].value if 'content' in entry else entry.summary
                    clean_body = self.clean_html(raw_body)
                    
                    # RSS không cung cấp score chi tiết cho comment, gán mặc định
                    comments.append({
                        'body': clean_body,
                        'author': entry.author if 'author' in entry else "Unknown",
                        'score': 0, 
                        'created_utc': time.mktime(entry.updated_parsed) if entry.updated_parsed else time.time(),
                        'permalink': entry.link,
                        'timestamp': datetime.fromtimestamp(time.mktime(entry.updated_parsed)) if entry.updated_parsed else datetime.now()
                    })

                return {
                    'success': True,
                    'meta': {
                        'title': post_title,
                        'subreddit': subreddit,
                        'score': 0, # RSS không có score realtime cho post
                        'num_comments': len(comments),
                        'author': post_author,
                        'created': datetime.fromtimestamp(time.mktime(post_entry.updated_parsed)) if post_entry.updated_parsed else datetime.now(),
                        'url': post_link,
                        'permalink': post_link,
                        'selftext': post_content
                    },
                    'comments': comments
                }

        except Exception as e:
            return {'success': False, 'error': f'Lỗi hệ thống: {str(e)}'}

# --- TRENDING POSTS MANAGER (RSS VERSION) ---

class TrendingPostsManager:
    def __init__(self):
        pass
    
    def fetch_trending_posts(self, subreddit, limit=10, time_filter='day'):
        """Fetch trending posts using RSS to avoid blocking"""
        # RSS URL cho top/hot: https://www.reddit.com/r/{subreddit}/hot.rss
        url = f"https://www.reddit.com/r/{subreddit}/hot.rss?limit={limit}"
        
        try:
            feed = feedparser.parse(url)
            posts = []
            
            for entry in feed.entries[:limit]:
                # Extract image if any
                thumbnail = ''
                if 'content' in entry:
                    content_str = entry.content[0].value
                    img_match = re.search(r'src="(https://i.redd.it/[^"]+)"', content_str)
                    if img_match:
                        thumbnail = img_match.group(1)

                posts.append({
                    'id': entry.id if 'id' in entry else str(hash(entry.link)),
                    'title': entry.title,
                    'author': entry.author if 'author' in entry else "Unknown",
                    'score': 0, # RSS hidden score
                    'comments_count': 0, # RSS hidden count
                    'created_utc': time.mktime(entry.updated_parsed) if entry.updated_parsed else time.time(),
                    'url': entry.link,
                    'subreddit': subreddit,
                    'thumbnail': thumbnail
                })
            
            return posts
        except Exception as e:
            print(f"Error fetching trending RSS for {subreddit}: {e}")
            return []
    
    def fetch_multiple_subreddits(self, subreddits, limit_per_sub=5):
        all_posts = []
        for subreddit in subreddits:
            posts = self.fetch_trending_posts(subreddit, limit=limit_per_sub)
            all_posts.extend(posts)
            # RSS requests are safer, no sleep needed usually
        return all_posts
    
    def analyze_trends(self, posts):
        """Analyze trends from posts"""
        if not posts: return {}
        
        subreddit_stats = {}
        for post in posts:
            sub = post['subreddit']
            if sub not in subreddit_stats:
                subreddit_stats[sub] = {
                    'count': 0, 
                    'total_score': 0, 
                    'total_comments': 0, 
                    'posts': [],
                    'avg_score': 0, # Placeholder
                    'avg_comments': 0 # Placeholder
                }
            
            subreddit_stats[sub]['count'] += 1
            subreddit_stats[sub]['posts'].append(post)
        
        return subreddit_stats

# --- AI SUMMARIZER CLASS (GEMINI 1.5 FLASH) ---

class AISummarizer:
    def __init__(self):
        self.model = None
        self.api_key = GOOGLE_GEMINI_API_KEY
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Lỗi cấu hình Gemini: {e}")

    def generate_summary(self, title, body, top_comments):
        if not GEMINI_AVAILABLE:
            return "⚠️ Chưa cài thư viện `google-generativeai`. Chạy `pip install google-generativeai`."
        
        if not self.model:
            return "⚠️ Chưa cấu hình API Key. Vui lòng kiểm tra `.streamlit/secrets.toml`."

        comments_text = ""
        if top_comments:
            comments_text = "\n".join([f"- {c['body']}" for c in top_comments[:10]])

        prompt = f"""
        Bạn là một trợ lý AI thông minh. Hãy tóm tắt bài thảo luận Reddit sau bằng TIẾNG VIỆT (Markdown).
        
        THÔNG TIN:
        - Tiêu đề: {title}
        - Nội dung chính: {body}
        - Bình luận nổi bật:
        {comments_text}
        
        YÊU CẦU:
        1. **Vấn đề chính:** Tóm tắt ngắn gọn vấn đề đang được thảo luận.
        2. **Phản ứng cộng đồng:** Người dùng đồng tình, phản đối hay đưa ra lời khuyên gì?
        3. **Cảm xúc chủ đạo:** Tích cực / Tiêu cực / Trung lập.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {str(e)}"

# --- NLP ENGINE ---

class EnhancedNLPEngine:
    def __init__(self):
        self.emotions = {
            'Anger': {'hate', 'stupid', 'angry', 'mad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst'},
            'Joy': {'love', 'great', 'awesome', 'happy', 'perfect', 'amazing', 'excellent', 'wonderful', 'fantastic', 'best'},
            'Trust': {'secure', 'safe', 'trust', 'reliable', 'quality', 'confident', 'dependable', 'honest', 'authentic'},
            'Fear': {'scary', 'worry', 'risk', 'afraid', 'problem', 'dangerous', 'concerned', 'nervous', 'anxious'},
            'Surprise': {'wow', 'unexpected', 'shocked', 'surprised', 'amazed', 'astonished', 'incredible', 'unbelievable'},
            'Sadness': {'sad', 'disappointed', 'sorry', 'bad', 'poor', 'unfortunate', 'regret', 'upset', 'depressed'}
        }
        
    def analyze_text(self, text):
        # Enhanced sentiment analysis
        polarity = 0
        subjectivity = 0.5
        
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Fallback basic sentiment
            positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic', 'perfect', 'wonderful'}
            negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'ridiculous'}
            words = set(re.findall(r'\w+', text.lower()))
            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))
            total_words = len(words)
            if total_words > 0:
                polarity = (pos_count - neg_count) / total_words

        # Enhanced sentiment categorization
        if polarity > 0.2: 
            sentiment = 'Positive'
            sentiment_emoji = '😊'
        elif polarity > 0.05:
            sentiment = 'Slightly Positive'
            sentiment_emoji = '🙂'
        elif polarity < -0.2: 
            sentiment = 'Negative'
            sentiment_emoji = '😠'
        elif polarity < -0.05:
            sentiment = 'Slightly Negative' 
            sentiment_emoji = '😕'
        else: 
            sentiment = 'Neutral'
            sentiment_emoji = '😐'

        # Enhanced emotion detection
        detected_emotions = []
        words = set(re.findall(r'\w+', text.lower()))
        emotion_scores = {}
        
        for emotion, keywords in self.emotions.items():
            matches = len(words.intersection(keywords))
            if matches > 0:
                emotion_scores[emotion] = matches
                detected_emotions.append(emotion)
        
        # Sort by score and get top 2 emotions
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [emotion for emotion, score in sorted_emotions[:2]]
        
        if not detected_emotions: 
            top_emotions = ['Neutral']

        return {
            'sentiment': sentiment,
            'sentiment_emoji': sentiment_emoji,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotions': top_emotions,
            'word_count': len(text.split())
        }

    def process_batch(self, comments):
        results = []
        total_comments = len(comments)
        
        # Use a simpler progress indicator
        progress_bar = st.progress(0)
        
        for i, comment in enumerate(comments):
            analysis = self.analyze_text(comment['body'])
            results.append(comment | analysis)
            
            # Update progress less frequently
            if i % 5 == 0 or i == total_comments - 1:
                progress_bar.progress((i + 1) / total_comments)
        
        progress_bar.empty()
        return results

# --- VIZ ENGINE ---

class EnhancedVizEngine:
    @staticmethod
    def plot_sentiment_distribution(df):
        sentiment_order = ['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']
        counts = df['sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
        
        color_map = {
            'Positive': '#00D4AA',
            'Slightly Positive': '#4AE8C5',
            'Neutral': '#FFD166', 
            'Slightly Negative': '#FF9E64',
            'Negative': '#FF6B6B'
        }
        
        fig = px.pie(
            values=counts.values, 
            names=counts.index, 
            hole=0.5,
            color=counts.index,
            color_discrete_map=color_map,
            title="🧠 Sentiment Distribution Analysis"
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        return fig

    @staticmethod
    def plot_emotion_radar(df):
        all_emotions = [e for sublist in df['emotions'] for e in sublist if e != 'Neutral']
        if not all_emotions: 
            return None
        
        counts = Counter(all_emotions)
        categories = list(counts.keys())
        values = [counts[cat] for cat in categories]
        max_val = max(values) if values else 1
        normalized_values = [v/max_val for v in values]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_values, 
            theta=categories, 
            fill='toself',
            line=dict(color='#FF6B6B', width=3),
            fillcolor='rgba(255, 107, 107, 0.3)',
            name="Emotion Intensity"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#4A5568', color='white'),
                angularaxis=dict(gridcolor='#4A5568', color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            title="😊 Emotional Footprint Analysis",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        return fig

    @staticmethod
    def plot_sentiment_timeline(df):
        if len(df) < 2: return None
        
        # Ensure timestamp is datetime
        try:
            df['dt'] = pd.to_datetime(df['created_utc'], unit='s')
        except:
            return None
            
        df = df.sort_values('dt')
        
        color_map = {
            'Positive': '#00D4AA', 'Slightly Positive': '#4AE8C5',
            'Neutral': '#FFD166', 'Slightly Negative': '#FF9E64', 'Negative': '#FF6B6B'
        }
        
        fig = px.scatter(df, x='dt', y='polarity', color='sentiment',
                         title="📈 Sentiment Evolution Over Time",
                         color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                          xaxis=dict(gridcolor='#4A5568'), yaxis=dict(gridcolor='#4A5568'))
        return fig

# --- AUTHENTICATION COMPONENTS ---

def show_login_form():
    """Login form - Modern design"""
    st.markdown("### 🔐 Login to Your Account")
    
    with st.form("login_form_enhanced"):
        username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_clicked = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
        with col2:
            st.form_submit_button("🔄 Clear", use_container_width=True)
    
    if login_clicked:
        if not AUTH_AVAILABLE:
            # Demo Login Mode
            st.session_state.authenticated = True
            st.session_state.user = {"id": 1, "username": username or "DemoUser", "email": "demo@example.com"}
            st.success("🎉 Logged in (Demo Mode)!")
            st.rerun()
            return

        if not username or not password:
            st.error("❌ Please enter both username and password")
            return
        
        try:
            db = user_db_manager.get_session()
            user = authenticate_user(db, username, password)
            
            if user:
                st.session_state.authenticated = True
                st.session_state.user = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name
                }
                st.success(f"🎉 Welcome back, {user.username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
                
        except Exception as e:
            st.error(f"❌ Login failed: {str(e)}")

def show_register_form():
    """Register form - Modern design"""
    st.markdown("### 📝 Create New Account")
    
    with st.form("register_form_enhanced"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("👤 Username *", placeholder="Choose username")
            email = st.text_input("📧 Email *", placeholder="your.email@example.com")
        with col2:
            password = st.text_input("🔒 Password *", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input("✅ Confirm Password *", type="password")
        
        full_name = st.text_input("👨‍💼 Full Name (optional)", placeholder="Your full name")
        
        register_clicked = st.form_submit_button("🚀 Create Account", use_container_width=True, type="primary")
    
    if register_clicked:
        if not AUTH_AVAILABLE:
            st.warning("⚠️ Registration is disabled in Demo Mode.")
            return

        if not all([username, email, password, confirm_password]):
            st.error("❌ Please fill in all required fields")
            return
            
        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return
        
        try:
            db = user_db_manager.get_session()
            user_service = UserService(db)
            user = user_service.create_user(
                username=username.strip(),
                email=email.lower().strip(),
                password=password,
                full_name=full_name.strip() if full_name else None
            )
            st.success("🎉 Registration Successful! You can now login.")
            
        except Exception as e:
            st.error(f"❌ Registration failed: {str(e)}")

def show_user_groups():
    """Display user's groups in sidebar with enhanced design"""
    if not st.session_state.authenticated:
        return
        
    try:
        # Check if auth system is real or demo
        if not AUTH_AVAILABLE:
            st.markdown("### 📈 Your Tracked Groups (Demo)")
            st.info("Groups are static in demo mode.")
            return

        db = user_db_manager.get_session()
        user_service = UserService(db)
        current_user = st.session_state.user
        
        with st.sidebar.expander("📊 Your Groups", expanded=True):
            st.markdown("### 🎯 Manage Groups")
            
            # Add new group
            with st.form("add_group_form"):
                group_name = st.text_input("Group Name", placeholder="Tech News")
                subreddit = st.text_input("Subreddit", placeholder="technology")
                
                if st.form_submit_button("➕ Add Group", use_container_width=True):
                    if group_name and subreddit:
                        try:
                            user_service.add_user_group(
                                user_id=current_user["id"],
                                group_name=group_name,
                                subreddit=subreddit.replace('r/', '').strip()
                            )
                            st.success(f"✅ Added '{group_name}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            # Show current groups
            st.markdown("---")
            user_groups = user_service.get_user_groups(current_user["id"])
            if user_groups:
                for group in user_groups:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{group.group_name}**\nr/{group.subreddit}")
                    with col2:
                        if st.button("🗑️", key=f"del_{group.id}"):
                            user_service.remove_user_group(current_user["id"], group.id)
                            st.rerun()
            else:
                st.info("🎯 No groups yet.")

    except Exception as e:
        st.error(f"Error loading groups: {e}")

def show_auth_section():
    """Enhanced authentication section in sidebar"""
    if not st.session_state.authenticated:
        st.markdown("<div class='auth-section'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Authentication")
        st.write("Login or register to unlock advanced features")
        
        tab1, tab2 = st.tabs(["🚪 Login", "📝 Register"])
        with tab1:
            show_login_form()
        with tab2:
            show_register_form()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Premium Features")
        cols = st.columns(2)
        with cols[0]:
            st.markdown('<span class="feature-badge">🧠 NLP Analysis</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-badge">📊 Advanced Viz</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<span class="feature-badge">😊 Emotion AI</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-badge">⏳ Time Trends</span>', unsafe_allow_html=True)
        
    else:
        # User is logged in
        user_info = st.session_state.user
        st.success(f"👋 Welcome back, **{user_info['username']}**!")
        
        st.markdown(f"""
        <div class='user-info-card'>
            <strong>👨‍💼 Account Information</strong><br>
            📧 {user_info['email']}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            if AUTH_AVAILABLE: logout()
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🛠️ Your Dashboard")
        show_user_groups()

# --- TRENDING CONTENT DISPLAY ---

def show_trending_posts():
    """Display trending posts from user's followed groups using RSS"""
    if not st.session_state.authenticated:
        return

    # Demo list if no DB connected
    subreddits = ["technology", "programming", "python", "artificial"]
    
    if AUTH_AVAILABLE:
        try:
            db = user_db_manager.get_session()
            user_service = UserService(db)
            user_groups = user_service.get_user_groups(st.session_state.user["id"])
            if user_groups:
                subreddits = [g.subreddit for g in user_groups]
        except:
            pass

    st.markdown("### 🔥 Trending from Your Groups (RSS)")

    # Time filter (Simulated for RSS)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        st.selectbox("Time Range", ["day", "week"], index=0, key="trending_time_filter")
    with col3:
        limit = st.slider("Posts per group", 3, 10, 5, key="trending_limit")

    # Fetch trending posts
    with st.spinner("🔄 Fetching trending posts..."):
        trend_manager = TrendingPostsManager()
        all_posts = trend_manager.fetch_multiple_subreddits(subreddits, limit_per_sub=limit)

    if not all_posts:
        st.warning("❌ Could not fetch trending posts. Reddit might be slow.")
        return

    # Display posts by subreddit
    for subreddit in subreddits:
        sub_posts = [p for p in all_posts if p.get('subreddit') == subreddit]
        if not sub_posts: continue

        with st.expander(f"📁 r/{subreddit}", expanded=True):
            for post in sub_posts:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{post['title']}**")
                    st.markdown(f"👤 {post['author']} • [Link]({post['url']})")
                
                with col2:
                    st.button(
                        "🧠 Analyze", 
                        key=f"analyze_{post['id']}", 
                        use_container_width=True, 
                        type="primary",
                        on_click=analyze_post_callback,
                        args=(post['url'],)
                    )
                st.markdown("---")

def show_trend_analysis():
    """Show trend analysis across all followed groups"""
    if not st.session_state.authenticated:
        return
    
    st.markdown("### 📈 Community Trends Analysis")
    st.info("Analysis based on RSS feeds from your groups.")
    
    # Simple stats for RSS data
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Communities", "4")
    with col2:
        st.metric("Active Threads Analyzed", "20")

def show_welcome_page():
    """Enhanced welcome page for unauthenticated users"""
    st.markdown("""
    <div class="welcome-container">
        <h1 style="font-size: 3em; margin-bottom: 20px;">🧠 Reddit Analytics Pro</h1>
        <p style="font-size: 1.3em; opacity: 0.95;">Advanced NLP • Emotional Intelligence • Temporal Trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👋 Please Login in the sidebar to access the dashboard.")

# --- SINGLE ANALYSIS FUNCTIONALITY ---

def perform_analysis(url):
    """Perform analysis on a given URL and display results"""
    loader = RedditLoader()
    nlp = EnhancedNLPEngine()
    viz = EnhancedVizEngine()
    ai_summarizer = AISummarizer()
    
    with st.status("🔍 Analyzing...", expanded=True) as status:
        # Fetch data
        status.update(label="🔄 Fetching data from Reddit (RSS Mode)...")
        raw_data = loader.fetch(url)
        if not raw_data['success']:
            status.update(label="❌ Failed", state="error")
            st.error(f"Error: {raw_data['error']}")
            return None
        
        # Process comments
        status.update(label=f"🧠 Analyzing {len(raw_data['comments'])} comments...")
        processed_comments = nlp.process_batch(raw_data['comments'])
        df = pd.DataFrame(processed_comments)
        
        # Lọc comment quá ngắn
        if not df.empty:
            df = df[df['word_count'] >= 2]
        
        # Generate Summary
        status.update(label="🤖 Generating AI Summary...")
        summary_text = ai_summarizer.generate_summary(
            title=raw_data['meta']['title'],
            body=raw_data['meta']['selftext'],
            top_comments=raw_data['comments']
        )
        
        # Save to session
        st.session_state.current_analysis = {
            'df': df, 
            'meta': raw_data['meta'],
            'summary': summary_text,
            'processed_at': datetime.now()
        }
        
        # Save to history
        hist_entry = {
            'id': str(time.time()), 
            'url': url, 
            'title': raw_data['meta']['title'][:50],
            'sub': raw_data['meta']['subreddit'],
            'comments': len(df),
            'timestamp': datetime.now()
        }
        st.session_state.history.append(hist_entry)
        
        status.update(label=f"✅ Analyzed {len(df)} comments", state="complete")

    # Display Results
    data = st.session_state.current_analysis
    meta = data['meta']
    df = data['df']
    summary = data['summary']
    
    # Header
    st.markdown(f"### 📄 {meta['title']}")
    st.caption(f"Subreddit: r/{meta['subreddit']} | Author: {meta['author']}")

    # Tabs
    tab_summary, tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Overview", "🧠 Emotions", "🔬 Comments"])
    
    with tab_summary:
        st.markdown("### 🤖 AI Executive Summary")
        st.markdown(f"<div class='comment-card'>{summary}</div>", unsafe_allow_html=True)
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.plot_sentiment_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(viz.plot_sentiment_timeline(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(viz.plot_emotion_radar(df), use_container_width=True)
    
    with tab3:
        st.dataframe(df)

def show_single_analysis():
    """Single URL analysis functionality"""
    st.markdown("### 🔗 Analyze Reddit Thread")
    
    # Check if we have a URL from trending posts analysis
    if st.session_state.get('trending_analysis_triggered'):
        url = st.session_state.trending_analysis_url
        st.session_state.trending_analysis_triggered = False
        st.session_state.url = url
        perform_analysis(url)
        return
    
    # Normal URL input
    url = st.text_input("🔗 Reddit Thread URL", key="single_analysis_url", placeholder="https://www.reddit.com/r/...")
    
    if st.button("🚀 ANALYZE", type="primary"):
        if url:
            st.session_state.url = url
            perform_analysis(url)
    
    elif st.session_state.get('current_analysis'):
        # Re-render results if already exists
        # We need to pass the URL stored in history or session to redraw
        # For simplicity, we assume the user wants to see the last analysis result
        # Just call display logic directly
        data = st.session_state.current_analysis
        
        # Re-draw Tabs (Copy paste of logic inside perform_analysis for re-rendering)
        # Or better: Just re-run perform_analysis with the same data? 
        # Since streamlit re-runs the whole script, we just need to render the UI parts.
        
        # --- RENDER UI FROM STATE ---
        meta = data['meta']
        df = data['df']
        summary = data['summary']
        viz = EnhancedVizEngine()
        
        st.markdown(f"### 📄 {meta['title']}")
        st.caption(f"Subreddit: r/{meta['subreddit']} | Author: {meta['author']}")

        tab_summary, tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Overview", "🧠 Emotions", "🔬 Comments"])
        
        with tab_summary:
            st.markdown("### 🤖 AI Executive Summary")
            st.markdown(f"<div class='comment-card'>{summary}</div>", unsafe_allow_html=True)
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(viz.plot_sentiment_distribution(df), use_container_width=True)
            with col2: st.plotly_chart(viz.plot_sentiment_timeline(df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(viz.plot_emotion_radar(df), use_container_width=True)
        
        with tab3:
            st.dataframe(df)

def show_dashboard():
    """Main dashboard view"""
    st.markdown("### 📊 Quick Access")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🔥 Trending Posts", use_container_width=True, on_click=switch_tab, args=("🔥 Trending Posts",))
    with col2:
        st.button("📈 Analyze Trends", use_container_width=True, on_click=switch_tab, args=("📈 Trend Analysis",))
    with col3:
        st.button("🔗 Single Analysis", use_container_width=True, on_click=switch_tab, args=("🔗 Single Analysis",))
    
    st.markdown("### 📈 Recent Activity")
    if st.session_state.history:
        for hist in reversed(st.session_state.history[-3:]):
            st.info(f"{hist['timestamp'].strftime('%H:%M')} - {hist['title']}")
    else:
        st.info("No analysis history yet.")

# --- MAIN APP ---

def main():
    # Initialize session state
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if "user" not in st.session_state: st.session_state.user = None
    if "history" not in st.session_state: st.session_state.history = []
    if "current_analysis" not in st.session_state: st.session_state.current_analysis = None
    if "trending_analysis_triggered" not in st.session_state: st.session_state.trending_analysis_triggered = False
    if "active_tab" not in st.session_state: st.session_state.active_tab = "🏠 Dashboard"

    # NAVIGATION OPTIONS
    NAV_DASHBOARD = "🏠 Dashboard"
    NAV_TRENDING = "🔥 Trending Posts"
    NAV_ANALYSIS = "📈 Trend Analysis"
    NAV_SINGLE = "🔗 Single Analysis"
    NAV_OPTIONS = [NAV_DASHBOARD, NAV_TRENDING, NAV_ANALYSIS, NAV_SINGLE]

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Reddit AI Tool")
        show_auth_section()

    # --- MAIN CONTENT ---
    if not st.session_state.authenticated:
        show_welcome_page()
        return

    # Authenticated UI
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Reddit Analytics Pro</h1>
        <p>RSS-Powered • AI Enhanced • Secure</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tabs
    selected_nav = st.radio(
        "Navigation", 
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state.active_tab) if st.session_state.active_tab in NAV_OPTIONS else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    # Sync radio with state if changed manually
    if selected_nav != st.session_state.active_tab:
        st.session_state.active_tab = selected_nav
        st.rerun()

    st.markdown("---")

    if st.session_state.active_tab == NAV_DASHBOARD:
        show_dashboard()
    elif st.session_state.active_tab == NAV_TRENDING:
        show_trending_posts()
    elif st.session_state.active_tab == NAV_ANALYSIS:
        show_trend_analysis()
    elif st.session_state.active_tab == NAV_SINGLE:
        show_single_analysis()

if __name__ == "__main__":
    main()