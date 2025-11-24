import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import time
import requests # Cần thư viện này để tải data thật

# --- CONFIG ---
st.set_page_config(
    page_title="Reddit Real Data Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. SENTIMENT ANALYZER (Bộ phân tích cảm xúc) ---
class SimpleSentimentAnalyzer:
    """Rule-based sentiment analysis"""
    
    def __init__(self):
        # Từ điển đơn giản để demo
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'amazing', 'fantastic',
            'wonderful', 'perfect', 'love', 'like', 'nice', 'cool', 'best',
            'impressed', 'recommend', 'clean', 'fast', 'helpful', 'smart',
            'agree', 'right', 'beautiful', 'thank', 'thanks', 'useful'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'stupid', 'ridiculous', 'useless', 'waste', 'poor',
            'slow', 'buggy', 'broken', 'garbage', 'trash', 'pain', 'error',
            'wrong', 'ugly', 'sad', 'boring', 'messy', 'fail'
        }
    
    def analyze(self, text):
        if not text:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.5 + (0.1 * min(positive_count, 5))
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = -0.5 - (0.1 * min(negative_count, 5))
        else:
            sentiment = 'neutral'
            score = 0.0
            
        confidence = 0.6 + (0.05 * (positive_count + negative_count))
        confidence = min(0.95, confidence)
            
        return {
            'sentiment': sentiment,
            'score': float(score),
            'confidence': float(confidence)
        }

sentiment_analyzer = SimpleSentimentAnalyzer()

# --- 2. REAL DATA LOADER (Lõi xử lý mới) ---
class RealRedditLoader:
    """
    Tải dữ liệu THẬT từ Reddit thông qua JSON endpoint.
    Không cần API Key, nhưng cần User-Agent để tránh bị chặn.
    """
    
    def fetch_data(self, url):
        # 1. Chuẩn bị URL json
        if not url.endswith('.json'):
            # Xử lý trường hợp URL có tham số query (vd: ?utm_source=...)
            if '?' in url:
                parts = url.split('?')
                json_url = parts[0] + '.json?' + parts[1]
            else:
                json_url = url.rstrip('/') + '.json'
        else:
            json_url = url

        # 2. Fake User-Agent (Quan trọng để Reddit không chặn request)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }

        try:
            response = requests.get(json_url, headers=headers, timeout=10)
            
            # Kiểm tra lỗi
            if response.status_code == 429:
                return {'success': False, 'error': 'Too Many Requests. Reddit is blocking us temporarily. Please wait 1 minute.'}
            if response.status_code != 200:
                return {'success': False, 'error': f'Failed to load data. Status code: {response.status_code}'}

            data = response.json()
            
            # 3. Parse dữ liệu Post (Phần tử đầu tiên trong list JSON)
            post_info = data[0]['data']['children'][0]['data']
            
            parsed_post = {
                'title': post_info.get('title', 'No Title'),
                'author': post_info.get('author', 'Unknown'),
                'subreddit': post_info.get('subreddit', 'Unknown'),
                'score': post_info.get('score', 0),
                'upvote_ratio': post_info.get('upvote_ratio', 0.0),
                'num_comments': post_info.get('num_comments', 0),
                'url': url,
                'created_utc': datetime.fromtimestamp(post_info.get('created_utc', 0))
            }
            
            # 4. Parse dữ liệu Comments (Phần tử thứ 2 trong list JSON)
            comments_raw = data[1]['data']['children']
            parsed_comments = []
            
            for item in comments_raw:
                # Chỉ lấy comment thật (kind = t1), bỏ qua "load more" (kind = more)
                if item['kind'] == 't1':
                    c_data = item['data']
                    body = c_data.get('body', '')
                    
                    if body and body != '[deleted]' and body != '[removed]':
                        parsed_comments.append({
                            'body': body,
                            'author': c_data.get('author', 'Unknown'),
                            'score': c_data.get('score', 0),
                            'timestamp': datetime.fromtimestamp(c_data.get('created_utc', 0))
                        })
            
            # Giới hạn số lượng comment để demo nhanh
            parsed_comments = parsed_comments[:100]

            return {
                'success': True,
                'post_data': parsed_post,
                'comments': parsed_comments,
                'source': 'Reddit Live JSON'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

real_loader = RealRedditLoader()

# --- 3. UI FUNCTIONS ---

def display_results(data):
    post = data['post_data']
    comments = data['comments']
    
    # Header Info
    st.success(f"✅ Data fetched successfully from: r/{post['subreddit']}")
    
    # Metrics Overview
    st.header("📝 Post Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score (Upvotes)", post['score'])
    c2.metric("Comments (Fetched)", len(comments))
    c3.metric("Subreddit", f"r/{post['subreddit']}")
    c4.metric("Upvote Ratio", f"{post['upvote_ratio'] * 100:.0f}%")
    
    st.subheader(f"Title: {post['title']}")
    st.caption(f"Posted by u/{post['author']} at {post['created_utc']}")
    
    st.divider()
    
    # Sentiment Analysis Logic
    st.header("📊 Sentiment Analysis (On Real Comments)")
    
    if not comments:
        st.warning("No readable comments found for this post.")
        return

    # Chạy vòng lặp phân tích
    results = []
    progress_bar = st.progress(0)
    
    for i, comment in enumerate(comments):
        # AI Logic chạy ở đây
        analysis = sentiment_analyzer.analyze(comment['body'])
        
        # Gán kết quả vào comment
        comment['sentiment'] = analysis['sentiment']
        comment['confidence'] = analysis['confidence']
        results.append(analysis['sentiment'])
        
        # Update UI progress
        if i % 5 == 0:
            progress_bar.progress((i + 1) / len(comments))
            
    progress_bar.empty() # Xóa thanh loading khi xong
    
    # Thống kê kết quả
    counts = pd.Series(results).value_counts()
    
    # Metrics Sentiment
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive Comments", counts.get('positive', 0))
    c2.metric("Negative Comments", counts.get('negative', 0))
    c3.metric("Neutral Comments", counts.get('neutral', 0))
    
    # Chart
    if len(results) > 0:
        st.bar_chart(counts)
    
    # Detailed List
    st.subheader("💬 Real Comments Detail")
    
    # Filter
    filter_opt = st.selectbox("Filter by Sentiment:", ["All", "positive", "negative", "neutral"])
    
    filtered_comments = comments
    if filter_opt != "All":
        filtered_comments = [c for c in comments if c['sentiment'] == filter_opt]
        
    st.write(f"Showing **{len(filtered_comments)}** comments:")
    
    # Render List
    for c in filtered_comments:
        color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        color = color_map.get(c['sentiment'], 'gray')
        icon_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐'}
        icon = icon_map.get(c['sentiment'], '😐')
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 15px; margin-bottom: 15px; background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
                <div style="font-weight: bold; margin-bottom: 5px;">
                    {icon} {c['sentiment'].upper()} <span style="font-weight: normal; font-size: 0.8em; color: #555;">(Score: {c['score']} | Conf: {c['confidence']:.2f})</span>
                </div>
                <div style="font-style: italic;">"{c['body']}"</div>
                <div style="font-size: 0.8em; color: #666; margin-top: 5px;">— u/{c['author']}</div>
            </div>
            """, unsafe_allow_html=True)

# --- MAIN APP LOGIC ---

def main():
    st.title("🔥 Reddit Real Data Analyzer")
    st.markdown("This tool fetches **LIVE DATA** from Reddit (No API Key needed).")
    
    # FORM INPUT
    with st.form("main_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            url = st.text_input("Paste Reddit Link:", placeholder="https://www.reddit.com/r/python/comments/...")
        with col2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🚀 Fetch Real Data", type="primary", use_container_width=True)
            
    # XỬ LÝ KHI BẤM NÚT
    if submitted and url:
        if "reddit.com" not in url:
            st.error("❌ Please enter a valid Reddit URL.")
        else:
            with st.spinner("Connecting to Reddit & Downloading data..."):
                # Gọi hàm tải dữ liệu thật
                result = real_loader.fetch_data(url)
                
                if result['success']:
                    # Lưu vào session để không mất khi reload
                    st.session_state['real_data'] = result
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # HIỂN THỊ KẾT QUẢ TỪ SESSION
    if 'real_data' in st.session_state:
        display_results(st.session_state['real_data'])
        
    # FOOTER
    st.divider()
    with st.expander("ℹ️ How this works (Technical)"):
        st.write("""
        1. **JSON Endpoint:** Appends `.json` to your URL to access Reddit's public API.
        2. **Requests:** Downloads the raw JSON data containing post info and comments.
        3. **Parsing:** Extracts relevant fields (Author, Body, Score) from the nested JSON structure.
        4. **Analysis:** Runs the text through a local sentiment analysis engine.
        """)

if __name__ == "__main__":
    main()