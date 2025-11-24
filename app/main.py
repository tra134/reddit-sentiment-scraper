# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import time
import sys
import os

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional NLP imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Import authentication modules
try:
    from core.user_database import user_db_manager
    from core.auth import authenticate_user, logout
    from services.user_service import UserService
    AUTH_AVAILABLE = True
except ImportError as e:
    print(f"Authentication modules not available: {e}")
    AUTH_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Reddit Analytics Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK THEME ---
st.markdown("""
<style>
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
    .analysis-section {
        background: #2D3748;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid #4A5568;
        color: #E2E8F0;
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
        
        ⭐ **Accuracy**  
        Advanced algorithms for precise sentiment analysis
        
        ⭐ **Speed**  
        Real-time processing of large datasets
        
        ⭐ **Depth**  
        Multi-dimensional analysis beyond basic sentiment
        
        ⭐ **Usability**  
        Intuitive interface for all skill levels
        """)

# --- ENHANCED CORE FUNCTIONALITY CLASSES ---

class RedditLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch(self, url):
        clean_url = (url.split('?')[0] if '?' in url else url.rstrip('/')) + '.json'
        try:
            with st.spinner('🔄 Fetching Reddit data...'):
                resp = self.session.get(clean_url, timeout=20)
                if resp.status_code != 200: 
                    return {'success': False, 'error': f'HTTP {resp.status_code} - Unable to fetch data'}
                
                data = resp.json()
                post = data[0]['data']['children'][0]['data']
                comments = []
                
                def process_comments(children_list, depth=0):
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
                            if 'replies' in d and d['replies'] and isinstance(d['replies'], dict):
                                process_comments(d['replies']['data']['children'], depth + 1)
                
                process_comments(data[1]['data']['children'])
                
                return {
                    'success': True,
                    'meta': {
                        'title': post.get('title'),
                        'subreddit': post.get('subreddit'),
                        'score': post.get('score'),
                        'upvote_ratio': post.get('upvote_ratio'),
                        'comments_count': post.get('num_comments'),
                        'author': post.get('author'),
                        'created': datetime.fromtimestamp(post.get('created_utc')),
                        'url': post.get('url'),
                        'permalink': post.get('permalink')
                    },
                    'comments': comments
                }
        except Exception as e:
            return {'success': False, 'error': f'Connection error: {str(e)}'}

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
        progress_bar = st.progress(0)
        total_comments = len(comments)
        
        for i, comment in enumerate(comments):
            analysis = self.analyze_text(comment['body'])
            results.append(comment | analysis)
            progress_bar.progress((i + 1) / total_comments)
        
        progress_bar.empty()
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

    # Enhanced URL Input Section
    st.markdown("### 🔗 Analyze Reddit Thread")
    
    input_col, btn_col = st.columns([4, 1])
    with input_col:
        url = st.text_input(
            "Paste Reddit URL here", 
            value=st.session_state.get('url', ''),
            placeholder="https://www.reddit.com/r/technology/comments/...",
            help="Paste any Reddit thread URL for comprehensive analysis",
            label_visibility="collapsed"
        )
    with btn_col:
        st.write("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 START ANALYSIS", type="primary", use_container_width=True)

    # Enhanced Demo URLs
    with st.expander("🎯 Quick Analysis Templates", expanded=False):
        st.markdown("### Popular Community Templates")
        demos = {
            "💻 Technology": "https://www.reddit.com/r/technology/comments/example/",
            "👨‍💻 Programming": "https://www.reddit.com/r/programming/comments/example/", 
            "🐍 Python": "https://www.reddit.com/r/Python/comments/example/",
            "📱 Apple": "https://www.reddit.com/r/apple/comments/example/",
            "🎮 Gaming": "https://www.reddit.com/r/gaming/comments/example/"
        }
        
        cols = st.columns(len(demos))
        for idx, (name, demo_url) in enumerate(demos.items()):
            with cols[idx]:
                if st.button(name, key=f"demo_{idx}", use_container_width=True):
                    st.session_state.url = demo_url
                    st.rerun()

    # Enhanced Analysis Processing
    if analyze_btn and url:
        st.session_state.url = url
        loader = RedditLoader()
        nlp = EnhancedNLPEngine()
        viz = EnhancedVizEngine()
        
        with st.status("🔍 Starting Comprehensive Analysis...", expanded=True) as status:
            # Fetch data
            status.update(label="🔄 Fetching thread data from Reddit...")
            raw_data = loader.fetch(url)
            
            if not raw_data['success']:
                status.update(label="❌ Analysis Failed", state="error")
                st.error(f"Error: {raw_data['error']}")
                return
            
            # Process comments
            status.update(label=f"🧠 Analyzing {len(raw_data['comments'])} comments with AI...")
            processed_comments = nlp.process_batch(raw_data['comments'])
            df = pd.DataFrame(processed_comments)
            df = df[df['word_count'] >= 3]  # Filter short comments
            
            # Save to session
            st.session_state.current_analysis = {
                'df': df, 
                'meta': raw_data['meta'],
                'processed_at': datetime.now()
            }
            
            # Save to history
            hist_entry = {
                'id': str(time.time()), 
                'url': url, 
                'title': raw_data['meta']['title'][:100] + "...",
                'sub': raw_data['meta']['subreddit'],
                'comments': len(df),
                'timestamp': datetime.now()
            }
            if not any(h['url'] == url for h in st.session_state.history):
                st.session_state.history.append(hist_entry)
            
            status.update(label=f"✅ Successfully analyzed {len(df)} comments", state="complete")

        # Enhanced Results Display
        meta = raw_data['meta']
        df = st.session_state.current_analysis['df']
        
        # Enhanced KPIs
        st.markdown("### 🏆 Executive Summary")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Score</h3>
                <h2>{meta['score']:,}</h2>
                <p>Post Engagement</p>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💬 Comments</h3>
                <h2>{len(df):,}</h2>
                <p>Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            avg_pol = df['polarity'].mean()
            sentiment = "😊 Positive" if avg_pol > 0.1 else "😟 Negative" if avg_pol < -0.1 else "😐 Neutral"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎭 Avg Sentiment</h3>
                <h2>{avg_pol:.3f}</h2>
                <p>{sentiment}</p>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            avg_words = df['word_count'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Avg Words</h3>
                <h2>{avg_words:.0f}</h2>
                <p>Per Comment</p>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            emotion_count = len(set([e for sublist in df['emotions'] for e in sublist]))
            st.markdown(f"""
            <div class="metric-card">
                <h3>😊 Emotions</h3>
                <h2>{emotion_count}</h2>
                <p>Detected</p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview Dashboard", "🧠 Emotional Analysis", "🔬 Comments Explorer", "📈 Advanced Metrics"])
        
        with tab1:
            st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
            st.markdown("### 📊 Sentiment & Engagement Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(viz.plot_sentiment_distribution(df), use_container_width=True)
            with col2:
                timeline_fig = viz.plot_sentiment_timeline(df)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
            st.markdown("### 😊 Emotional Intelligence Analysis")
            
            radar_fig = viz.plot_emotion_radar(df)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Emotion statistics
            st.markdown("#### Emotion Frequency Distribution")
            all_emotions = [e for sublist in df['emotions'] for e in sublist]
            emotion_counts = Counter(all_emotions)
            
            if emotion_counts:
                em_cols = st.columns(len(emotion_counts))
                for idx, (emotion, count) in enumerate(emotion_counts.most_common()):
                    with em_cols[idx]:
                        percentage = (count / len(df)) * 100
                        st.metric(f"{emotion}", f"{count}", f"{percentage:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
            st.markdown("### 🔬 Detailed Comments Analysis")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=df['sentiment'].unique(),
                    default=df['sentiment'].unique()
                )
            with col2:
                emotion_filter = st.multiselect(
                    "Filter by Emotion", 
                    options=list(set([e for sublist in df['emotions'] for e in sublist])),
                    default=[]
                )
            with col3:
                min_score = st.slider("Minimum Score", min_value=0, max_value=int(df['score'].max()), value=0)
            
            # Apply filters
            filtered_df = df[df['sentiment'].isin(sentiment_filter)]
            filtered_df = filtered_df[filtered_df['score'] >= min_score]
            
            if emotion_filter:
                filtered_df = filtered_df[filtered_df['emotions'].apply(lambda x: any(em in emotion_filter for em in x))]
            
            st.markdown(f"**Showing {len(filtered_df)} of {len(df)} comments**")
            
            # Enhanced comment display
            for idx, row in filtered_df.head(20).iterrows():
                sentiment_color = {
                    'Positive': '#00D4AA',
                    'Slightly Positive': '#4AE8C5',
                    'Neutral': '#FFD166', 
                    'Slightly Negative': '#FF9E64',
                    'Negative': '#FF6B6B'
                }.get(row['sentiment'], '#888')
                
                emotions_display = ", ".join(row['emotions']) if row['emotions'] else "Neutral"
                
                st.markdown(f"""
                <div class="comment-card" style="border-left-color: {sentiment_color}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div>
                            <strong>👤 {row['author']}</strong>
                            <span style="margin-left: 10px; background: {sentiment_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                                {row['sentiment_emoji']} {row['sentiment']}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                                ⬆️ {row['score']}
                            </span>
                        </div>
                    </div>
                    <div style="margin: 10px 0; color: #CBD5E0; font-size: 0.9em;">
                        🎭 <strong>Emotions:</strong> {emotions_display} • 
                        📊 <strong>Polarity:</strong> {row['polarity']:.3f} • 
                        📝 <strong>Words:</strong> {row['word_count']}
                    </div>
                    <div style="color: #E2E8F0; line-height: 1.5;">{row['body'][:300]}{'...' if len(row['body']) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
            st.markdown("### 📈 Advanced Engagement Metrics")
            st.plotly_chart(viz.plot_engagement_metrics(df, meta), use_container_width=True)
            
            # Additional statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Basic Statistics")
                st.dataframe(df[['score', 'word_count', 'polarity', 'subjectivity']].describe(), use_container_width=True)
            with col2:
                st.markdown("#### 🏆 Top Contributors")
                top_authors = df['author'].value_counts().head(10)
                st.dataframe(pd.DataFrame({'Author': top_authors.index, 'Comments': top_authors.values}), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced empty state for authenticated users
    elif not analyze_btn:
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: white;">🚀 Ready for Advanced Reddit Analysis</h2>
            <p style="font-size: 1.1em; opacity: 0.8; color: white;">Paste a Reddit URL above to unlock powerful insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What You'll Discover
            
            - **🧠 Sentiment Intelligence**: Deep understanding of discussion moods
            - **😊 Emotional Patterns**: AI-powered emotion classification
            - **📊 Engagement Analytics**: Comprehensive metrics and trends
            - **⏳ Temporal Evolution**: How sentiment changes over time
            - **🔍 Behavioral Insights**: User engagement patterns
            
            ### 💼 Professional Use Cases
            
            - **Market Research**: Understand customer opinions
            - **Brand Monitoring**: Track sentiment across communities  
            - **Community Management**: Identify influencers and issues
            - **Competitive Analysis**: Compare with other communities
            - **Content Strategy**: Data-driven decision making
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Advanced Features
            
            #### 📈 Visualization
            - Interactive dashboards
            - Real-time charts
            - Professional reports
            - Export capabilities
            
            #### 🔧 Analysis Tools
            - Smart filtering
            - Comparative analysis
            - Trend detection
            - Pattern recognition
            
            #### 💾 Data Management
            - Historical tracking
            - Group management
            - Export options
            - Secure storage
            """)

if __name__ == "__main__":
    main()