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

# --- CẤU HÌNH API KEY (QUAN TRỌNG) ---
# Đảm bảo bạn đã thêm GOOGLE_API_KEY vào secrets trên Streamlit Cloud
try:
    GOOGLE_GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_GEMINI_API_KEY = None # Xử lý lỗi nếu chưa cấu hình

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

# Import authentication modules (Xử lý lỗi nếu chưa có database)
try:
    from core.user_database import user_db_manager
    from core.auth import authenticate_user, logout
    from services.user_service import UserService
    AUTH_AVAILABLE = True
except ImportError as e:
    # print(f"Authentication modules not available: {e}")
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

# --- [UPDATE 1/2] TRENDING POSTS MANAGER (ANTI-BLOCK) ---

class TrendingPostsManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Danh sách server trung gian (Mirrors) để tránh bị chặn IP
        self.instances = [
            "https://l.opnxng.com",
            "https://r.nf",
            "https://snoo.habedieeh.re",
            "https://reddit.smnz.de",
            "https://libreddit.bus-hit.me"
        ]
    
    def fetch_trending_posts(self, subreddit, limit=10, time_filter='day'):
        """Fetch trending posts using Mirrors"""
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Thử lần lượt các server mirror
        for instance in self.instances:
            try:
                # Tạo URL mirror
                url = f"{instance}/r/{subreddit}/top.json?limit={limit}&t={time_filter}"
                
                # Gọi request
                response = self.session.get(url, headers=headers, timeout=8)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = []
                    
                    # Cấu trúc JSON mirror thường giống Reddit gốc
                    children = data.get('data', {}).get('children', [])
                    
                    for post in children:
                        post_data = post.get('data', {})
                        posts.append({
                            'id': post_data.get('id'),
                            'title': post_data.get('title'),
                            'author': post_data.get('author'),
                            'score': post_data.get('score', 0),
                            'comments_count': post_data.get('num_comments', 0),
                            'created_utc': post_data.get('created_utc'),
                            # Tạo link gốc về reddit để user click vào xem
                            'url': f"https://www.reddit.com{post_data.get('permalink')}",
                            'subreddit': subreddit,
                            'upvote_ratio': post_data.get('upvote_ratio', 0),
                            'thumbnail': post_data.get('thumbnail', ''),
                            'is_video': post_data.get('is_video', False),
                            'over_18': post_data.get('over_18', False)
                        })
                    
                    # Nếu lấy thành công thì trả về luôn, thoát vòng lặp
                    return posts
                
            except Exception:
                continue # Thử server tiếp theo nếu lỗi

        # Nếu thử hết mà vẫn không được
        return []
    
    def fetch_multiple_subreddits(self, subreddits, limit_per_sub=5):
        """Fetch trending posts from multiple subreddits"""
        all_posts = []
        
        # Tạo thanh tiến trình
        progress_bar = st.progress(0)
        total = len(subreddits)

        for i, subreddit in enumerate(subreddits):
            posts = self.fetch_trending_posts(subreddit, limit=limit_per_sub)
            all_posts.extend(posts)
            progress_bar.progress((i + 1) / total)
            time.sleep(0.1)  # Nghỉ nhẹ
        
        progress_bar.empty()

        # Sort by score and return
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        return all_posts
    
    def analyze_trends(self, posts):
        """Analyze trends from posts"""
        if not posts:
            return {}
        
        # Analyze by subreddit
        subreddit_stats = {}
        for post in posts:
            sub = post['subreddit']
            if sub not in subreddit_stats:
                subreddit_stats[sub] = {
                    'count': 0,
                    'total_score': 0,
                    'total_comments': 0,
                    'posts': []
                }
            
            subreddit_stats[sub]['count'] += 1
            subreddit_stats[sub]['total_score'] += post['score']
            subreddit_stats[sub]['total_comments'] += post['comments_count']
            subreddit_stats[sub]['posts'].append(post)
        
        # Calculate averages
        for sub in subreddit_stats:
            stats = subreddit_stats[sub]
            stats['avg_score'] = stats['total_score'] / stats['count']
            stats['avg_comments'] = stats['total_comments'] / stats['count']
        
        return subreddit_stats

# --- [UPDATE 2/2] REDDIT LOADER (ANTI-BLOCK FOR SINGLE POST) ---

class RedditLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.instances = [
            "https://l.opnxng.com",
            "https://r.nf",
            "https://snoo.habedieeh.re",
            "https://reddit.smnz.de",
            "https://libreddit.bus-hit.me"
        ]

    def fetch(self, url):
        """Fetch single post data using Mirrors to bypass blocking"""
        
        # 1. Trích xuất đường dẫn (path) từ URL gốc
        # Ví dụ: https://www.reddit.com/r/funny/comments/xyz/ -> /r/funny/comments/xyz/
        path = ""
        if "reddit.com" in url:
            try:
                path = url.split("reddit.com")[-1]
            except:
                return {'success': False, 'error': 'Invalid Reddit URL'}
        elif url.startswith("/r/"):
             path = url
        else:
             # URL không hợp lệ
             return {'success': False, 'error': 'Invalid URL format'}
        
        # Xóa query params nếu có
        if "?" in path:
            path = path.split("?")[0]
        
        # Đảm bảo có đuôi .json
        if not path.endswith(".json"):
            if path.endswith("/"):
                path = path[:-1] + ".json"
            else:
                path = path + ".json"

        # 2. Xoay vòng qua các Mirrors để lấy dữ liệu
        for instance in self.instances:
            try:
                mirror_url = f"{instance}{path}"
                
                with st.spinner(f'🔄 Fetching data (trying {instance.split("//")[1]})...'):
                    resp = self.session.get(mirror_url, timeout=15)
                    
                    if resp.status_code == 200:
                        # --- XỬ LÝ DỮ LIỆU ---
                        data = resp.json()
                        
                        # Cấu trúc JSON của mirror giống Reddit: List 2 phần tử [Post, Comments]
                        post_data = data[0]['data']['children'][0]['data']
                        comments_data = data[1]['data']['children']
                        
                        comments = []
                        
                        # Hàm đệ quy lấy comment
                        def process_comments_recursive(children_list, depth=0):
                            for item in children_list:
                                if item['kind'] == 't1':  # Comment
                                    d = item['data']
                                    body = d.get('body', '')
                                    if body and body not in ['[deleted]', '[removed]']:
                                        comments.append({
                                            'id': d.get('id'),
                                            'author': d.get('author', 'Unknown'),
                                            'body': body,
                                            'score': d.get('score', 0),
                                            'created_utc': d.get('created_utc'),
                                            'timestamp': datetime.fromtimestamp(d.get('created_utc', 0)),
                                            'depth': d.get('depth', 0),
                                            'permalink': f"https://reddit.com{d.get('permalink', '')}"
                                        })
                                    
                                    # Lấy replies (comment con)
                                    if 'replies' in d and d['replies'] and isinstance(d['replies'], dict):
                                        process_comments_recursive(d['replies']['data']['children'], depth + 1)
                        
                        process_comments_recursive(comments_data)
                        
                        return {
                            'success': True,
                            'meta': {
                                'title': post_data.get('title'),
                                'subreddit': post_data.get('subreddit'),
                                'score': post_data.get('score'),
                                'upvote_ratio': post_data.get('upvote_ratio'),
                                'comments_count': post_data.get('num_comments'),
                                'author': post_data.get('author'),
                                'created': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                'url': post_data.get('url'),
                                'permalink': post_data.get('permalink'),
                                'selftext': post_data.get('selftext', '')
                            },
                            'comments': comments
                        }
            except Exception:
                continue # Thử mirror tiếp theo

        return {'success': False, 'error': 'All mirrors failed to fetch data. Reddit blocked access.'}

# --- AI SUMMARIZER CLASS (GEMINI 2.0 FLASH) ---

class AISummarizer:
    def __init__(self):
        self.model = None
        self.api_key = GOOGLE_GEMINI_API_KEY
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Dùng model Flash cho nhanh và miễn phí
                self.model = genai.GenerativeModel('gemini-1.5-flash') 
            except Exception as e:
                print(f"Lỗi cấu hình Gemini: {e}")

    def generate_summary(self, title, body, top_comments):
        if not GEMINI_AVAILABLE:
            return "⚠️ Chưa cài thư viện Google AI. Hãy chạy: pip install google-generativeai"
        
        if not self.model:
            return "⚠️ Chưa cấu hình API Key hoặc lỗi Key."

        # Chuẩn bị nội dung
        comments_text = ""
        if top_comments:
            # Lấy 10 comment để AI hiểu phản ứng cộng đồng
            comments_text = "\n".join([f"- {c['body']}" for c in top_comments[:10]])

        # Prompt tối ưu cho Tiếng Việt và Reddit
        prompt = f"""
        Bạn là một trợ lý AI thông minh. Hãy tóm tắt bài thảo luận Reddit sau bằng Tiếng Việt.
        
        THÔNG TIN:
        - Tiêu đề: {title}
        - Nội dung chính: {body}
        - Bình luận nổi bật:
        {comments_text}
        
        YÊU CẦU:
        1. Tóm tắt nội dung chính: Vấn đề là gì? (Ngắn gọn).
        2. Phản ứng cộng đồng: Mọi người khuyên gì/nghĩ gì?
        3. Văn phong: Tự nhiên, dễ hiểu, KHÔNG LỖI FONT.
        4. Trình bày: Dùng gạch đầu dòng.
        """

        try:
            # Gọi API (Gemini Flash rất nhanh)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {str(e)}"

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
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Enhanced fallback sentiment
            positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic', 'perfect', 'wonderful'}
            negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'ridiculous'}
            
            words = set(re.findall(r'\w+', text.lower()))
            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))
            total_words = len(words)
            
            if total_words > 0:
                polarity = (pos_count - neg_count) / total_words
            else:
                polarity = 0
            subjectivity = 0.5 + (abs(polarity) * 0.3)

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
            'emotion_scores': emotion_scores,
            'word_count': len(text.split()),
            'char_count': len(text),
            'readability_score': max(0, min(100, 100 - (len(text.split()) / 3)))  # Simple readability estimate
        }

    def process_batch(self, comments):
        results = []
        total_comments = len(comments)
        
        # Use a simpler progress indicator for better performance
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, comment in enumerate(comments):
            analysis = self.analyze_text(comment['body'])
            results.append(comment | analysis)
            
            # Update progress less frequently for better performance
            if i % 10 == 0 or i == total_comments - 1:
                progress_bar.progress((i + 1) / total_comments)
                progress_text.text(f"🔄 Processing comments... {i+1}/{total_comments}")
        
        progress_bar.empty()
        progress_text.empty()
        return results

class EnhancedVizEngine:
    @staticmethod
    def plot_sentiment_distribution(df):
        sentiment_order = ['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']
        counts = df['sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
        
        # Bright colors for dark theme
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
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_color='white')
        fig.update_layout(
            font=dict(size=12, color='white'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
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
            showlegend=False, 
            height=400,
            title="😊 Emotional Footprint Analysis",
            font=dict(size=12, color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def plot_sentiment_timeline(df):
        if len(df) < 2: 
            return None
            
        df_copy = df.copy().sort_values('timestamp')
        window_size = max(1, min(20, int(len(df_copy)/3)))
        df_copy['MA'] = df_copy['polarity'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Add scatter points with sentiment colors
        color_map = {
            'Positive': '#00D4AA',
            'Slightly Positive': '#4AE8C5',
            'Neutral': '#FFD166',
            'Slightly Negative': '#FF9E64',
            'Negative': '#FF6B6B'
        }
        
        for sentiment in df_copy['sentiment'].unique():
            sentiment_df = df_copy[df_copy['sentiment'] == sentiment]
            fig.add_trace(go.Scatter(
                x=sentiment_df['timestamp'], 
                y=sentiment_df['polarity'], 
                mode='markers',
                marker=dict(
                    color=color_map.get(sentiment, '#888'),
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                name=sentiment,
                hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Polarity: %{y:.3f}<extra></extra>',
                text=[f"Sentiment: {s}" for s in sentiment_df['sentiment']]
            ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['MA'], 
            mode='lines',
            line=dict(color='#667eea', width=4),
            name='Trend Line',
            hovertemplate='Trend: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="📈 Sentiment Evolution Over Time",
            yaxis_title="Sentiment Polarity",
            xaxis_title="Time",
            height=400,
            showlegend=True,
            font=dict(size=12, color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white')),
            xaxis=dict(color='white', gridcolor='#4A5568'),
            yaxis=dict(color='white', gridcolor='#4A5568')
        )
        return fig

    @staticmethod
    def plot_engagement_metrics(df, meta):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Comment Scores', '⏰ Activity Hours', '📝 Word Count Distribution', '😊 Emotion Frequency'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Comment scores
        score_bins = pd.cut(df['score'], bins=5)
        score_counts = score_bins.value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=[str(x) for x in score_counts.index], y=score_counts.values, name="Scores", marker_color='#667eea'),
            row=1, col=1
        )
        
        # Activity hours
        df['hour'] = df['timestamp'].dt.hour
        hour_counts = df['hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hour_counts.index, y=hour_counts.values, name="Activity", marker_color='#4ECDC4'),
            row=1, col=2
        )
        
        # Word count distribution
        fig.add_trace(
            go.Histogram(x=df['word_count'], nbinsx=20, name="Word Count", marker_color='#FF6B6B'),
            row=2, col=1
        )
        
        # Emotion frequency
        all_emotions = [e for sublist in df['emotions'] for e in sublist]
        emotion_counts = Counter(all_emotions)
        fig.add_trace(
            go.Bar(x=list(emotion_counts.keys()), y=list(emotion_counts.values()), name="Emotions", marker_color='#FFD166'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600, 
            showlegend=False, 
            title_text="📊 Comprehensive Engagement Analysis",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Update subplot titles color
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(color='white')
            
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
        if not username or not password:
            st.error("❌ Please enter both username and password")
            return False
        
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
                st.balloons()
                time.sleep(1)
                st.rerun()
                return True
            else:
                st.error("❌ Invalid username or password")
                return False
                
        except Exception as e:
            st.error(f"❌ Login failed: {str(e)}")
            return False
    
    return False

def show_register_form():
    """Register form - Modern design"""
    st.markdown("### 📝 Create New Account")
    
    with st.form("register_form_enhanced"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("👤 Username *", placeholder="Choose username (min 3 chars)")
            email = st.text_input("📧 Email *", placeholder="your.email@example.com")
        with col2:
            password = st.text_input("🔒 Password *", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input("✅ Confirm Password *", type="password")
        
        full_name = st.text_input("👨‍💼 Full Name (optional)", placeholder="Your full name")
        
        register_clicked = st.form_submit_button("🚀 Create Account", use_container_width=True, type="primary")
    
    if register_clicked:
        if not all([username, email, password, confirm_password]):
            st.error("❌ Please fill in all required fields")
            return False
            
        if len(username) < 3:
            st.error("❌ Username must be at least 3 characters")
            return False
            
        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return False
            
        if len(password) < 6:
            st.error("❌ Password must be at least 6 characters")
            return False
            
        if "@" not in email:
            st.error("❌ Please enter a valid email address")
            return False
        
        try:
            db = user_db_manager.get_session()
            user_service = UserService(db)
            user = user_service.create_user(
                username=username.strip(),
                email=email.lower().strip(),
                password=password,
                full_name=full_name.strip() if full_name else None
            )
            
            st.markdown(f"""
            <div class="success-message">
                <h3>🎉 Registration Successful!</h3>
                <p><strong>Username:</strong> {user.username}</p>
                <p><strong>Email:</strong> {user.email}</p>
                <p>💡 You can now login with your credentials</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            return True
            
        except ValueError as e:
            error_msg = str(e)
            if "username" in error_msg.lower():
                st.error(f"❌ Username '{username}' is already taken")
            elif "email" in error_msg.lower():
                st.error(f"❌ Email '{email}' is already registered")
            else:
                st.error(f"❌ {error_msg}")
            return False
        except Exception as e:
            st.error(f"❌ Registration failed: {str(e)}")
            return False
    
    return False

def show_user_groups():
    """Display user's groups in sidebar with enhanced design"""
    if not st.session_state.authenticated:
        return
        
    try:
        db = user_db_manager.get_session()
        user_service = UserService(db)
        current_user = st.session_state.user
        
        if not hasattr(user_service, 'get_user_groups'):
            st.info("👥 Group features coming soon!")
            return
            
        with st.sidebar.expander("📊 Your Groups", expanded=True):
            st.markdown("### 🎯 Manage Groups")
            
            # Add new group
            if hasattr(user_service, 'add_user_group'):
                with st.form("add_group_form"):
                    group_name = st.text_input("Group Name", placeholder="Tech News", key="new_group_name")
                    subreddit = st.text_input("Subreddit", placeholder="technology", key="new_subreddit")
                    
                    if st.form_submit_button("➕ Add Group", use_container_width=True):
                        if group_name and subreddit:
                            try:
                                subreddit_clean = subreddit.replace('r/', '').strip()
                                user_service.add_user_group(
                                    user_id=current_user["id"],
                                    group_name=group_name,
                                    subreddit=subreddit_clean
                                )
                                st.success(f"✅ Added '{group_name}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                        else:
                            st.error("⚠️ Please fill in all fields")
            
            # Show current groups
            if hasattr(user_service, 'get_user_groups'):
                st.markdown("---")
                st.markdown("### 📈 Your Tracked Groups")
                user_groups = user_service.get_user_groups(current_user["id"])
                if user_groups:
                    for group in user_groups:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(
                                f"<div class='group-card'>"
                                f"<strong>📊 {group.group_name}</strong><br>"
                                f"<small>📍 r/{group.subreddit}</small>"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                        with col2:
                            if hasattr(user_service, 'remove_user_group'):
                                if st.button("🗑️", key=f"del_{group.id}"):
                                    user_service.remove_user_group(current_user["id"], group.id)
                                    st.rerun()
                else:
                    st.info("🎯 No groups yet. Add some to start tracking!")
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
        # User is logged in - Enhanced display
        user_info = st.session_state.user
        st.success(f"👋 Welcome back, **{user_info['username']}**!")
        
        # Enhanced user info card
        st.markdown(f"""
        <div class='user-info-card'>
            <strong>👨‍💼 Account Information</strong><br>
            📧 {user_info['email']}<br>
            🆔 ID: {user_info['id']}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_btn"):
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                logout()
                st.session_state.authenticated = False
                st.session_state.user = None
                st.rerun()
        
        # User features
        st.markdown("---")
        st.markdown("### 🛠️ Your Dashboard")
        show_user_groups()

# --- TRENDING CONTENT DISPLAY ---

def show_trending_posts():
    """Display trending posts from user's followed groups"""
    if not st.session_state.authenticated:
        return

    try:
        db = user_db_manager.get_session()
        user_service = UserService(db)
        current_user = st.session_state.user

        # Get user's groups
        user_groups = user_service.get_user_groups(current_user["id"])
        if not user_groups:
            st.info("🌟 Add some groups in the sidebar to see trending posts!")
            return

        subreddits = [group.subreddit for group in user_groups]

        st.markdown("### 🔥 Trending from Your Groups")

        # Time filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write("")  # Spacer
        with col2:
            time_filter = st.selectbox(
                "Time Range",
                ["day", "week", "month"],
                index=0,
                key="trending_time_filter"
            )
        with col3:
            limit = st.slider("Posts per group", 3, 10, 5, key="trending_limit")

        # Fetch trending posts
        with st.spinner("🔄 Fetching trending posts..."):
            trend_manager = TrendingPostsManager()
            all_posts = trend_manager.fetch_multiple_subreddits(subreddits, limit_per_sub=limit)

        if not all_posts:
            st.warning("❌ Could not fetch trending posts. Please check your internet connection.")
            return

        # Display posts by subreddit
        for subreddit in subreddits:
            sub_posts = [p for p in all_posts if p.get('subreddit') == subreddit]
            if not sub_posts:
                continue

            with st.expander(f"📁 r/{subreddit} - {len(sub_posts)} trending posts", expanded=True):
                for post in sub_posts:
                    # Validate required keys
                    required_keys = ['title', 'score', 'author', 'comments_count', 'created_utc', 'url']
                    if not all(k in post for k in required_keys):
                        st.warning(f"⚠️ Post missing data: {post}")
                        continue

                    # Compute time
                    try:
                        post_time = datetime.fromtimestamp(post['created_utc'])
                        time_ago = datetime.now() - post_time
                        days_ago = time_ago.days
                        hours_ago = time_ago.seconds // 3600
                    except Exception:
                        days_ago, hours_ago = 0, 0

                    # Create columns for layout
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{post['title']}**")
                        st.markdown(f"👤 {post['author']} • 💬 {post['comments_count']} comments • ⬆️ {post['score']} • 🕒 {days_ago}d {hours_ago}h ago")
                    
                    with col2:
                        # Create buttons for actions
                        btn_col1, btn_col2 = st.columns(2)
                        
                        with btn_col1:
                            if st.button("📖 Read", key=f"read_{post['id']}", use_container_width=True):
                                # Open in new tab using JavaScript
                                js = f"window.open('{post['url']}', '_blank');"
                                components.html(f"<script>{js}</script>", height=0)
                        
                        with btn_col2:
                            # --- MODIFIED: Switch tab logic with CALLBACK ---
                            st.button(
                                "🧠 Analyze", 
                                key=f"analyze_{post['id']}", 
                                use_container_width=True, 
                                type="primary",
                                on_click=analyze_post_callback,
                                args=(post['url'],)
                            )

                    st.markdown("---")

    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_trend_analysis():
    """Show trend analysis across all followed groups"""
    if not st.session_state.authenticated:
        return
    
    try:
        db = user_db_manager.get_session()
        user_service = UserService(db)
        current_user = st.session_state.user
        
        user_groups = user_service.get_user_groups(current_user["id"])
        if not user_groups:
            return
        
        subreddits = [group.subreddit for group in user_groups]
        
        st.markdown("### 📈 Community Trends Analysis")
        
        with st.spinner("🔍 Analyzing trends across your groups..."):
            trend_manager = TrendingPostsManager()
            all_posts = trend_manager.fetch_multiple_subreddits(subreddits, limit_per_sub=5)
            trends = trend_manager.analyze_trends(all_posts)
        
        if not trends:
            return
        
        # Display trend metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_posts = len(all_posts)
        avg_score = sum(post['score'] for post in all_posts) / total_posts if total_posts > 0 else 0
        avg_comments = sum(post['comments_count'] for post in all_posts) / total_posts if total_posts > 0 else 0
        most_active_sub = max(trends.items(), key=lambda x: x[1]['count'])[0] if trends else "N/A"
        
        with col1:
            st.metric("📊 Total Posts", total_posts)
        with col2:
            st.metric("⭐ Avg Score", f"{avg_score:.0f}")
        with col3:
            st.metric("💬 Avg Comments", f"{avg_comments:.0f}")
        with col4:
            st.metric("🏆 Most Active", f"r/{most_active_sub}")
        
        # Subreddit comparison
        st.markdown("#### 📊 Subreddit Performance")
        
        subreddit_data = []
        for sub, stats in trends.items():
            subreddit_data.append({
                'Subreddit': f"r/{sub}",
                'Posts': stats['count'],
                'Avg Score': stats['avg_score'],
                'Avg Comments': stats['avg_comments'],
                'Total Engagement': stats['total_score'] + stats['total_comments']
            })
        
        if subreddit_data:
            df_subs = pd.DataFrame(subreddit_data)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_score = px.bar(
                    df_subs, 
                    x='Subreddit', 
                    y='Avg Score',
                    title='📈 Average Score by Subreddit',
                    color='Avg Score',
                    color_continuous_scale='viridis'
                )
                fig_score.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_score, use_container_width=True)
            
            with col2:
                fig_engagement = px.pie(
                    df_subs,
                    values='Total Engagement',
                    names='Subreddit',
                    title='🎯 Engagement Distribution'
                )
                fig_engagement.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_engagement, use_container_width=True)
        
        # Top performing posts
        st.markdown("#### 🏆 Top Performing Posts")
        
        top_posts = sorted(all_posts, key=lambda x: x['score'], reverse=True)[:5]
        
        for i, post in enumerate(top_posts, 1):
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="trend-card">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.2em; margin-right: 10px;">{emoji}</span>
                        <strong style="color: #E2E8F0;">{post['title'][:80]}...</strong>
                    </div>
                    <div style="color: #CBD5E0; font-size: 0.9em;">
                        <span>r/{post['subreddit']}</span> • 
                        <span>👤 {post['author']}</span> • 
                        <span>⭐ {post['score']} points</span> • 
                        <span>💬 {post['comments_count']} comments</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📖", key=f"top_read_{post['id']}", use_container_width=True):
                        js = f"window.open('{post['url']}', '_blank');"
                        components.html(f"<script>{js}</script>", height=0)
                with btn_col2:
                    st.button(
                        "🧠", 
                        key=f"top_analyze_{post['id']}", 
                        use_container_width=True, 
                        type="primary",
                        on_click=analyze_post_callback,
                        args=(post['url'],)
                    )
            
    except Exception as e:
        st.error(f"Error analyzing trends: {e}")

def show_welcome_page():
    """Enhanced welcome page for unauthenticated users"""
    st.markdown("""
    <div class="welcome-container">
        <h1 style="font-size: 3em; margin-bottom: 20px;">🧠 Reddit Analytics Pro</h1>
        <p style="font-size: 1.3em; opacity: 0.95;">Advanced NLP • Emotional Intelligence • Temporal Trends • Professional Insights</p>
        <div style="margin-top: 30px;">
            <span class="feature-badge">🤖 AI-Powered</span>
            <span class="feature-badge">📈 Real-Time</span>
            <span class="feature-badge">🔒 Secure</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Next-Generation Reddit Analytics
        
        **Unlock powerful insights from Reddit discussions with our advanced AI platform:**
        
        ### 🧠 Intelligent Analysis
        - **Sentiment Analysis**: Deep understanding of discussion moods and tones
        - **Emotion Detection**: AI-powered emotion classification across 6 core emotions
        - **Behavioral Patterns**: Identify user engagement patterns and trends
        
        ### 📊 Advanced Visualization
        - **Interactive Dashboards**: Real-time, responsive charts and graphs
        - **Temporal Analysis**: Track sentiment evolution with precision
        - **Comparative Analytics**: Compare multiple threads and time periods
        
        ### 🎯 Professional Applications
        - **Market Research**: Understand customer opinions and feedback
        - **Brand Monitoring**: Track brand sentiment across communities
        - **Community Management**: Identify key influencers and pain points
        """)
    
    with col2:
        st.markdown("""
        ### 🔐 Get Started
        
        **Unlock all features:**
        
        1. **Create Account** - Quick registration
        2. **Login** - Access your dashboard
        3. **Analyze** - Start gaining insights
        
        ### 💡 Pro Features
        
        🎯 Smart Filtering  
        💾 Data Export  
        📱 Mobile Optimized  
        🔄 Real-Time Updates  
        📈 Professional Reports
        """)
        
        st.info("""
        💫 **Premium Analytics**
        - Advanced NLP processing
        - Emotional intelligence
        - Professional reporting
        - Export capabilities
        """)

    with col3:
        st.markdown("""
        ### 🏆 Why Choose Us?
        
        ⭐ **Accuracy** Advanced algorithms for precise sentiment analysis
        
        ⭐ **Speed** Real-time processing of large datasets
        
        ⭐ **Depth** Multi-dimensional analysis beyond basic sentiment
        
        ⭐ **Usability** Intuitive interface for all skill levels
        """)

# --- SINGLE ANALYSIS FUNCTIONALITY ---

def perform_analysis(url):
    """Perform analysis on a given URL and display results"""
    loader = RedditLoader()
    nlp = EnhancedNLPEngine()
    viz = EnhancedVizEngine()
    
    # Initialize summarizer
    ai_summarizer = AISummarizer()
    
    with st.status("🔍 Analyzing...", expanded=True) as status:
        # Fetch data
        status.update(label="🔄 Fetching data from Reddit (using Mirrors)...")
        raw_data = loader.fetch(url)
        if not raw_data['success']:
            status.update(label="❌ Failed", state="error")
            st.error(f"Error: {raw_data['error']}")
            return None
        
        # Process comments
        status.update(label=f"🧠 Analyzing {len(raw_data['comments'])} comments...")
        processed_comments = nlp.process_batch(raw_data['comments'])
        df = pd.DataFrame(processed_comments)
        
        if df.empty:
            status.update(label="⚠️ No comments to analyze", state="error")
            st.warning("Bài viết này không có bình luận nào để phân tích.")
            return None
            
        df = df[df['word_count'] >= 3]  # Filter short comments
        
        # --- NEW: GENERATE SUMMARY WITH GEMINI ---
        status.update(label="🤖 Generating AI Summary... (Using Gemini)")
        
        # Get selftext from meta (safely)
        post_body = raw_data['meta'].get('selftext', '')
        
        summary_text = ai_summarizer.generate_summary(
            title=raw_data['meta']['title'],
            body=post_body,
            top_comments=raw_data['comments']
        )
        
        # Save to session
        st.session_state.current_analysis = {
            'df': df, 
            'meta': raw_data['meta'],
            'summary': summary_text,  # Stored summary
            'processed_at': datetime.now()
        }
        
        # Save to history
        hist_entry = {
            'id': str(time.time()), 
            'url': url, 
            'title': raw_data['meta']['title'][:50] + "...",
            'sub': raw_data['meta']['subreddit'],
            'comments': len(df),
            'timestamp': datetime.now()
        }
        if not any(h['url'] == url for h in st.session_state.history):
            st.session_state.history.append(hist_entry)
        
        status.update(label=f"✅ Analyzed {len(df)} comments", state="complete")

    # Display Results
    meta = raw_data['meta']
    df = st.session_state.current_analysis['df']
    summary = st.session_state.current_analysis.get('summary', 'No summary available')
    
    # KPIs
    st.markdown("### 🏆 Executive Summary")
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric("Total Engagement", f"{meta['score']:,}")
    with k2:
        st.metric("Comments Analyzed", f"{len(df):,}")
    with k3:
        avg_pol = df['polarity'].mean()
        sentiment = "😊 Positive" if avg_pol > 0.1 else "😟 Negative" if avg_pol < -0.1 else "😐 Neutral"
        st.metric("Avg Sentiment", f"{avg_pol:.3f}", delta=sentiment)
    with k4:
        st.metric("Avg Words", f"{df['word_count'].mean():.0f}")

    # Analysis Tabs - Added "Summary" Tab
    tab_summary, tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Overview", "🧠 Emotions", "🔬 Comments"])
    
    with tab_summary:
        st.markdown("### 🤖 AI Executive Summary (Powered by Gemini)")
        st.info("Bản tóm tắt này được tạo tự động bởi AI dựa trên tiêu đề, nội dung bài viết và các bình luận hàng đầu.")
        
        st.markdown(f"""
        <div style="background-color: #2D3748; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; color: #E2E8F0; font-size: 1.1em; line-height: 1.6;">
            {summary}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📌 Key Details")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**Title:** {meta['title']}")
            st.markdown(f"**Author:** {meta['author']}")
        with col_s2:
            st.markdown(f"**Subreddit:** r/{meta['subreddit']}")
            st.markdown(f"**Posted:** {meta['created']}")
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.plot_sentiment_distribution(df), use_container_width=True)
        with col2:
            timeline_fig = viz.plot_sentiment_timeline(df)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        radar_fig = viz.plot_emotion_radar(df)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Emotion statistics
        st.markdown("#### Emotion Frequency")
        all_emotions = [e for sublist in df['emotions'] for e in sublist]
        emotion_counts = Counter(all_emotions)
        
        if emotion_counts:
            cols = st.columns(len(emotion_counts))
            for idx, (emotion, count) in enumerate(emotion_counts.most_common()):
                with cols[idx]:
                    percentage = (count / len(df)) * 100
                    st.metric(f"{emotion}", f"{count}", f"{percentage:.1f}%")
    
    with tab3:
        # Simple comment viewer
        for idx, row in df.head(10).iterrows():
            sentiment_color = {
                'Positive': '#00D4AA',
                'Slightly Positive': '#4AE8C5', 
                'Neutral': '#FFD166',
                'Slightly Negative': '#FF9E64',
                'Negative': '#FF6B6B'
            }.get(row['sentiment'], '#888')
            
            st.markdown(f"""
            <div class="comment-card" style="border-left-color: {sentiment_color}">
                <div style="display: flex; justify-content: space-between;">
                    <b>👤 {row['author']}</b>
                    <span>⬆️ {row['score']} • {row['sentiment_emoji']} {row['sentiment']}</span>
                </div>
                <div style="margin: 10px 0; color: #666;">
                    Emotions: {', '.join(row['emotions'])} • Words: {row['word_count']} • Polarity: {row['polarity']:.3f}
                </div>
                <div>{row['body'][:200]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    return df

def show_single_analysis():
    """Single URL analysis functionality"""
    st.markdown("### 🔗 Analyze Reddit Thread")
    
    # Check if we have a URL from trending posts analysis
    if (st.session_state.get('trending_analysis_triggered') and 
        st.session_state.get('trending_analysis_url')):
        
        url = st.session_state.trending_analysis_url
        
        # Reset the trigger so it doesn't loop
        st.session_state.trending_analysis_triggered = False
        st.session_state.trending_analysis_url = None
        
        # Update current URL input logic
        st.session_state.url = url
        st.session_state.single_analysis_url = url
        
        # Perform analysis immediately
        st.info(f"🔍 Analyzing trending post: {url}")
        perform_analysis(url)
        return
    
    # Normal URL input
    url = st.session_state.get('url', '')
    
    # URL Input
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        url = st.text_input(
            "🔗 Reddit Thread URL", 
            value=url,
            placeholder="https://www.reddit.com/r/...",
            help="Paste any Reddit thread URL",
            key="single_analysis_url"
        )
    with col_btn:
        st.write("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 ANALYZE", type="primary", use_container_width=True, key="single_analysis_btn")

    # REMOVED QUICK DEMOS AS REQUESTED

    # Analysis Processing
    if analyze_btn and url:
        st.session_state.url = url
        perform_analysis(url)

    # Empty state for authenticated users
    elif not analyze_btn and not st.session_state.get('current_analysis'):
        st.markdown("---")
        st.markdown("""
        ## 🚀 Ready to Analyze Reddit Threads
        
        **Paste a Reddit URL above to get started with:**
        
        - 🧠 **Sentiment Analysis** - Understand discussion moods
        - 😊 **Emotion Detection** - Identify specific emotions  
        - 📊 **Visual Analytics** - Interactive charts & insights
        - ⏳ **Temporal Trends** - See how sentiment evolves
        
        ### 💡 Pro Tips
        - Add groups in sidebar for personalized tracking
        - Export analysis for reports
        """)

# --- ENHANCED MAIN APP ---

def main():
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "trending_analysis_triggered" not in st.session_state:
        st.session_state.trending_analysis_triggered = False
    if "trending_analysis_url" not in st.session_state:
        st.session_state.trending_analysis_url = None
    
    # NAVIGATION OPTIONS
    NAV_DASHBOARD = "🏠 Dashboard"
    NAV_TRENDING = "🔥 Trending Posts"
    NAV_ANALYSIS = "📈 Trend Analysis"
    NAV_SINGLE = "🔗 Single Analysis"
    NAV_OPTIONS = [NAV_DASHBOARD, NAV_TRENDING, NAV_ANALYSIS, NAV_SINGLE]

    # Initialize Active Tab
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = NAV_DASHBOARD

    # --- ENHANCED SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: white;">⚙️ Control Panel</h1>
            <p style="opacity: 0.7; color: white;">Advanced Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        if AUTH_AVAILABLE:
            show_auth_section()
        else:
            st.warning("🔒 Authentication unavailable")

    # --- MAIN CONTENT ---
    if not st.session_state.authenticated:
        show_welcome_page()
        return

    # Authenticated user content - Enhanced design
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Reddit Analytics Pro</h1>
        <p>Advanced NLP • Emotional Intelligence • Professional Insights • Real-Time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced user welcome
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ **Welcome, {st.session_state.user['username']}!** All premium features unlocked! 🚀")
    with col2:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

    # Enhanced user groups display
    if AUTH_AVAILABLE and st.session_state.authenticated:
        try:
            db = user_db_manager.get_session()
            user_service = UserService(db)
            user_groups = user_service.get_user_groups(st.session_state.user["id"])
            if user_groups:
                st.markdown("### 🎯 Your Tracked Communities")
                cols = st.columns(min(4, len(user_groups)))
                for i, group in enumerate(user_groups[:4]):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>r/{group.subreddit}</h3>
                            <p>{group.group_name}</p>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading groups: {e}")

    # Main Navigation Tabs (Replaced st.tabs with st.radio for programmatic control)
    # Using 'label_visibility="collapsed"' to hide the label "Navigation"
    selected_nav = st.radio(
        "Navigation", 
        NAV_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab"
    )

    st.markdown("---")

    if selected_nav == NAV_DASHBOARD:
        show_dashboard()
    
    elif selected_nav == NAV_TRENDING:
        show_trending_posts()
    
    elif selected_nav == NAV_ANALYSIS:
        show_trend_analysis()
    
    elif selected_nav == NAV_SINGLE:
        show_single_analysis()

def show_dashboard():
    """Main dashboard view"""
    st.markdown("### 📊 Quick Access")
    
    # Quick actions with proper navigation
    col1, col2, col3 = st.columns(3)
    
    # --- MODIFIED: USE CALLBACKS TO AVOID SESSION STATE ERROR ---
    with col1:
        st.button(
            "🔥 View Trending Posts", 
            use_container_width=True, 
            key="dashboard_trending",
            on_click=switch_tab,
            args=("🔥 Trending Posts",)
        )
    
    with col2:
        st.button(
            "📈 Analyze Trends", 
            use_container_width=True, 
            key="dashboard_trends",
            on_click=switch_tab,
            args=("📈 Trend Analysis",)
        )
    
    with col3:
        st.button(
            "🔍 Single Analysis", 
            use_container_width=True, 
            key="dashboard_single",
            on_click=switch_tab,
            args=("🔗 Single Analysis",)
        )
    
    # Recent activity or quick stats
    st.markdown("### 📈 Recent Activity")
    
    if st.session_state.history:
        st.markdown("#### 📋 Analysis History")
        for hist in st.session_state.history[-3:]:  # Show last 3 analyses
            st.markdown(f"""
            <div class="trend-card">
                <strong>{hist['title']}</strong><br>
                <small>r/{hist['sub']} • {hist['comments']} comments • {hist['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No analysis history yet. Start by analyzing a Reddit thread!")

if __name__ == "__main__":
    main()