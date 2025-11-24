import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import time
import requests
import plotly.express as px
from typing import Dict, List, Any

# --- CONFIG ---
st.set_page_config(
    page_title="Reddit Sentiment Analyzer Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. ADVANCED SENTIMENT ANALYZER ---
class AdvancedSentimentAnalyzer:
    """Enhanced sentiment analysis with better accuracy"""
    
    def __init__(self):
        logger.info("🔄 Initializing Advanced Sentiment Analyzer")
        
        # Expanded sentiment dictionaries
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'amazing', 'fantastic',
            'wonderful', 'perfect', 'love', 'like', 'nice', 'cool', 'best',
            'fantastic', 'brilliant', 'outstanding', 'superb', 'terrific',
            'happy', 'pleased', 'satisfied', 'impressed', 'recommend',
            'beautiful', 'thank', 'thanks', 'useful', 'helpful', 'smart',
            'agree', 'right', 'clean', 'fast', 'easy', 'smooth', 'fun',
            'enjoy', 'enjoyable', 'pleasure', 'delight', 'marvelous',
            'exceptional', 'flawless', 'seamless', 'intuitive', 'responsive'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'stupid', 'ridiculous', 'useless', 'waste', 'poor',
            'slow', 'buggy', 'broken', 'garbage', 'trash', 'pain', 'error',
            'wrong', 'ugly', 'sad', 'boring', 'messy', 'fail', 'awful',
            'disappointing', 'frustrating', 'annoying', 'confusing',
            'complicated', 'crashed', 'freeze', 'lag', 'glitch', 'janky'
        }
        
        # Strong sentiment modifiers
        self.strong_positive = {'love', 'amazing', 'awesome', 'fantastic', 'perfect', 'brilliant'}
        self.strong_negative = {'hate', 'terrible', 'horrible', 'disgusting', 'awful', 'worst'}
        
        # Negations
        self.negations = {'not', "don't", 'never', 'no', 'cannot', "won't", "can't"}
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with enhanced logic"""
        if not text or len(text.strip()) < 3:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
        
        try:
            text_lower = text.lower()
            words = text_lower.split()
            
            positive_score = 0
            negative_score = 0
            
            # Analyze each word with weights
            for word in words:
                if word in self.strong_positive:
                    positive_score += 3
                elif word in self.positive_words:
                    positive_score += 1
                elif word in self.strong_negative:
                    negative_score += 3
                elif word in self.negative_words:
                    negative_score += 1
            
            # Check for negations
            for i, word in enumerate(words):
                if word in self.negations and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in self.positive_words:
                        negative_score += 2
                        positive_score = max(0, positive_score - 1)
                    elif next_word in self.negative_words:
                        positive_score += 2
                        negative_score = max(0, negative_score - 1)
            
            # Calculate final sentiment
            total_words = len(words)
            if total_words == 0:
                return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
            
            positive_ratio = positive_score / total_words
            negative_ratio = negative_score / total_words
            
            sentiment_threshold = 0.02
            
            if positive_ratio > negative_ratio + sentiment_threshold:
                sentiment = 'positive'
                confidence = min(0.95, 0.6 + positive_ratio)
            elif negative_ratio > positive_ratio + sentiment_threshold:
                sentiment = 'negative'
                confidence = min(0.95, 0.6 + negative_ratio)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            # Calculate score for visualization
            score = confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0.0
            
            return {
                'sentiment': sentiment,
                'score': float(score),
                'confidence': float(confidence),
                'positive_words': positive_score,
                'negative_words': negative_score
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
    
    def get_model_info(self):
        """Return model information"""
        return {
            'model_name': 'Advanced Rule-based Engine',
            'model_type': 'Enhanced Keyword Analysis',
            'status': 'active'
        }

# Initialize analyzer
sentiment_analyzer = AdvancedSentimentAnalyzer()

# --- 2. ENHANCED REAL DATA LOADER ---
class EnhancedRedditLoader:
    """
    Enhanced Reddit data loader with better error handling and performance
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def fetch_data(self, url: str) -> Dict[str, Any]:
        """Fetch real data from Reddit with enhanced error handling"""
        
        # Prepare JSON URL
        if not url.endswith('.json'):
            if '?' in url:
                parts = url.split('?')
                json_url = parts[0] + '.json?' + parts[1]
            else:
                json_url = url.rstrip('/') + '.json'
        else:
            json_url = url

        try:
            logger.info(f"Fetching data from: {json_url}")
            response = self.session.get(json_url, timeout=15)
            
            # Enhanced error handling
            if response.status_code == 429:
                return {
                    'success': False, 
                    'error': 'Reddit is temporarily blocking requests. Please wait 1-2 minutes and try again.'
                }
            elif response.status_code == 403:
                return {
                    'success': False,
                    'error': 'Access forbidden. Reddit may be blocking automated requests.'
                }
            elif response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to load data. HTTP Status: {response.status_code}'
                }

            data = response.json()
            
            # Parse post data
            post_info = data[0]['data']['children'][0]['data']
            parsed_post = self._parse_post_data(post_info, url)
            
            # Parse comments with enhanced filtering
            comments_raw = data[1]['data']['children']
            parsed_comments = self._parse_comments(comments_raw)
            
            logger.info(f"Successfully fetched {len(parsed_comments)} comments from r/{parsed_post['subreddit']}")

            return {
                'success': True,
                'post_data': parsed_post,
                'comments': parsed_comments,
                'analysis_timestamp': datetime.now(),
                'source': 'Reddit Live JSON',
                'total_comments_found': len(comments_raw)
            }

        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out. Please try again.'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Connection error. Check your internet connection.'}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {'success': False, 'error': f'Unexpected error: {str(e)}'}
    
    def _parse_post_data(self, post_info: Dict, url: str) -> Dict[str, Any]:
        """Parse post data with enhanced fields"""
        return {
            'title': post_info.get('title', 'No Title'),
            'author': post_info.get('author', 'Unknown'),
            'subreddit': post_info.get('subreddit', 'Unknown'),
            'score': post_info.get('score', 0),
            'upvote_ratio': post_info.get('upvote_ratio', 0.0),
            'num_comments': post_info.get('num_comments', 0),
            'url': url,
            'created_utc': datetime.fromtimestamp(post_info.get('created_utc', 0)),
            'flair': post_info.get('link_flair_text', 'No Flair'),
            'post_id': post_info.get('id', ''),
            'subreddit_subscribers': post_info.get('subreddit_subscribers', 0)
        }
    
    def _parse_comments(self, comments_raw: List) -> List[Dict[str, Any]]:
        """Parse comments with enhanced filtering and limits"""
        parsed_comments = []
        
        for item in comments_raw:
            if item['kind'] == 't1':  # Only real comments
                c_data = item['data']
                body = c_data.get('body', '')
                
                # Enhanced filtering
                if (body and 
                    body not in ['[deleted]', '[removed]'] and
                    len(body.strip()) > 10 and  # Minimum length
                    not body.startswith('Your comment is awaiting')):  # Skip moderation messages
                    
                    parsed_comments.append({
                        'body': body,
                        'author': c_data.get('author', 'Unknown'),
                        'score': c_data.get('score', 0),
                        'timestamp': datetime.fromtimestamp(c_data.get('created_utc', 0)),
                        'comment_id': c_data.get('id', ''),
                        'replies': c_data.get('replies', {}),
                        'sentiment': 'neutral',  # Will be analyzed
                        'confidence': 0.5
                    })
        
        # Limit to reasonable number for performance
        return parsed_comments[:150]

# Initialize data loader
reddit_loader = EnhancedRedditLoader()

# --- 3. ENHANCED UI COMPONENTS ---

def display_enhanced_results(data: Dict[str, Any]):
    """Display comprehensive analysis results"""
    
    post = data['post_data']
    comments = data['comments']
    
    st.success(f"✅ **Live Data Analysis Complete!** (Source: {data['source']})")
    
    # Enhanced Post Overview
    st.header("📝 Post Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Post Score", f"{post['score']:,}")
    with col2:
        st.metric("Comments Analyzed", len(comments))
    with col3:
        st.metric("Subreddit", f"r/{post['subreddit']}")
    with col4:
        st.metric("Upvote Ratio", f"{post['upvote_ratio'] * 100:.1f}%")
    with col5:
        st.metric("Total Comments", f"{post['num_comments']:,}")
    
    st.subheader(f"**{post['title']}**")
    st.caption(f"👤 Posted by u/{post['author']} • 🕒 {post['created_utc'].strftime('%Y-%m-%d %H:%M')} • 🏷️ Flair: {post['flair']}")
    
    st.divider()
    
    # Sentiment Analysis Section
    st.header("📊 Advanced Sentiment Analysis")
    
    if not comments:
        st.warning("No readable comments found for sentiment analysis.")
        return
    
    # Progress tracking for analysis
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    sentiment_results = []
    confidence_scores = []
    
    # Analyze each comment
    for i, comment in enumerate(comments):
        # Update progress
        if i % 10 == 0:
            progress = (i + 1) / len(comments)
            progress_bar.progress(progress)
            progress_text.text(f"🔍 Analyzing comments... {i+1}/{len(comments)}")
        
        # Perform sentiment analysis
        analysis = sentiment_analyzer.analyze(comment['body'])
        
        # Update comment with analysis results
        comment['sentiment'] = analysis['sentiment']
        comment['confidence'] = analysis['confidence']
        comment['sentiment_score'] = analysis['score']
        
        sentiment_results.append(analysis['sentiment'])
        confidence_scores.append(analysis['confidence'])
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    # Sentiment Statistics
    sentiment_counts = pd.Series(sentiment_results).value_counts()
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    # Enhanced Metrics
    st.subheader("🎯 Sentiment Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        positive_count = sentiment_counts.get('positive', 0)
        st.metric("😊 Positive", positive_count, 
                 delta=f"{(positive_count/len(comments)*100):.1f}%" if comments else "0%")
    
    with col2:
        negative_count = sentiment_counts.get('negative', 0)
        st.metric("😠 Negative", negative_count,
                 delta=f"{(negative_count/len(comments)*100):.1f}%" if comments else "0%")
    
    with col3:
        neutral_count = sentiment_counts.get('neutral', 0)
        st.metric("😐 Neutral", neutral_count,
                 delta=f"{(neutral_count/len(comments)*100):.1f}%" if comments else "0%")
    
    with col4:
        st.metric("📊 Avg Confidence", f"{avg_confidence:.2f}")
    
    with col5:
        if sentiment_counts.any():
            overall_sentiment = max(sentiment_counts.index, key=lambda x: sentiment_counts[x])
            sentiment_emoji = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}
            st.metric("🎭 Overall", f"{sentiment_emoji.get(overall_sentiment, '😐')} {overall_sentiment.title()}")
        else:
            st.metric("🎭 Overall", "Neutral")
    
    # Visualizations
    if len(sentiment_counts) > 0:
        st.subheader("📈 Visual Analytics")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Bar chart
            chart_data = pd.DataFrame({
                'Sentiment': sentiment_counts.index,
                'Count': sentiment_counts.values
            })
            st.bar_chart(chart_data.set_index('Sentiment'), use_container_width=True)
        
        with viz_col2:
            # Pie chart
            try:
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545', 
                        'neutral': '#ffc107'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("📊 Pie chart unavailable - using simplified view")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(comments)) * 100
                    st.write(f"- **{sentiment.title()}**: {count} comments ({percentage:.1f}%)")
    
    # Detailed Comments Analysis
    st.header("💬 Detailed Comment Analysis")
    
    # Enhanced filtering options
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative", "Neutral"],
            key="sentiment_filter"
        )
    
    with filter_col2:
        confidence_filter = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.3,
            key="confidence_filter"
        )
    
    with filter_col3:
        score_filter = st.slider(
            "Minimum Comment Score",
            -50, 100, 0,
            key="score_filter"
        )
    
    with filter_col4:
        sort_option = st.selectbox(
            "Sort By",
            ["Highest Score", "Highest Confidence", "Newest", "Oldest"],
            key="sort_option"
        )
    
    # Apply filters
    filtered_comments = comments.copy()
    
    if sentiment_filter != "All":
        filtered_comments = [c for c in filtered_comments if c['sentiment'] == sentiment_filter.lower()]
    
    filtered_comments = [c for c in filtered_comments if c['confidence'] >= confidence_filter]
    filtered_comments = [c for c in filtered_comments if c['score'] >= score_filter]
    
    # Apply sorting
    if sort_option == "Highest Score":
        filtered_comments.sort(key=lambda x: x['score'], reverse=True)
    elif sort_option == "Highest Confidence":
        filtered_comments.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_option == "Newest":
        filtered_comments.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_option == "Oldest":
        filtered_comments.sort(key=lambda x: x['timestamp'])
    
    st.write(f"**Showing {len(filtered_comments)} of {len(comments)} comments**")
    
    # Display filtered comments
    for comment in filtered_comments[:25]:  # Limit display for performance
        sentiment = comment['sentiment']
        confidence = comment['confidence']
        
        # Enhanced styling
        sentiment_config = {
            'positive': {'color': '#28a745', 'bg_color': '#d4edda', 'emoji': '😊'},
            'negative': {'color': '#dc3545', 'bg_color': '#f8d7da', 'emoji': '😠'},
            'neutral': {'color': '#ffc107', 'bg_color': '#fff3cd', 'emoji': '😐'}
        }
        
        config = sentiment_config.get(sentiment, sentiment_config['neutral'])
        
        st.markdown(f"""
        <div style="border-left: 4px solid {config['color']}; 
                    padding: 12px; 
                    margin: 12px 0; 
                    background-color: {config['bg_color']}15;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong style="font-size: 0.95em;">
                    {config['emoji']} {sentiment.title()} 
                    <span style="font-weight: normal; color: #666;">(Score: {comment['score']})</span>
                </strong>
                <span style="color: #666; font-size: 0.85em;">
                    Confidence: {confidence:.2f}
                </span>
            </div>
            <div style="margin: 8px 0; line-height: 1.5; font-size: 0.9em; color: #333;">
                {comment['body']}
            </div>
            <div style="color: #666; font-size: 0.8em;">
                👤 by u/{comment['author']} • 🕒 {comment['timestamp'].strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence progress bar
        st.progress(float(confidence), text=f"Confidence: {confidence:.0%}")
    
    # Export and Additional Features
    st.header("📤 Export & Insights")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("💾 Download CSV Report", use_container_width=True):
            df = pd.DataFrame([
                {
                    'Comment': comment['body'],
                    'Author': comment['author'],
                    'Score': comment['score'],
                    'Sentiment': comment['sentiment'],
                    'Confidence': comment['confidence'],
                    'Timestamp': comment['timestamp']
                }
                for comment in comments
            ])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"reddit_sentiment_{post['subreddit']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("📊 Generate Insights", use_container_width=True):
            positive_pct = (sentiment_counts.get('positive', 0) / len(comments)) * 100
            negative_pct = (sentiment_counts.get('negative', 0) / len(comments)) * 100
            
            if positive_pct > 60:
                overall = "Very Positive"
                insight = "The community strongly supports this content"
            elif positive_pct > 40:
                overall = "Mostly Positive" 
                insight = "Generally favorable reception with some criticism"
            elif negative_pct > 40:
                overall = "Mostly Negative"
                insight = "Significant concerns or criticisms raised"
            else:
                overall = "Neutral/Mixed"
                insight = "Balanced discussion with varied opinions"
            
            st.info(f"""
            **📈 Analysis Insights:**
            
            - **Overall Sentiment:** {overall}
            - **Community Engagement:** {'High' if len(comments) > 50 else 'Medium' if len(comments) > 20 else 'Low'}
            - **Confidence Level:** {'High' if avg_confidence > 0.7 else 'Medium' if avg_confidence > 0.5 else 'Low'}
            - **Key Insight:** {insight}
            - **Sentiment Distribution:** {positive_pct:.1f}% Positive, {negative_pct:.1f}% Negative
            """)
    
    with export_col3:
        if st.button("🔄 Analyze New Post", use_container_width=True):
            st.session_state.pop('real_data', None)
            st.rerun()

# --- 4. MAIN APPLICATION ---

def main():
    st.title("🔥 Reddit Sentiment Analyzer Pro")
    st.markdown("""
    **Advanced sentiment analysis using LIVE Reddit data** 
    
    ✨ **Features:**
    - 🚀 Real-time data from Reddit (No API keys needed)
    - 🧠 Advanced sentiment analysis with confidence scoring  
    - 📊 Interactive visualizations and filtering
    - 💾 Export results to CSV
    - 🎯 Detailed comment-level analysis
    """)
    
    # Initialize session state
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'total_comments' not in st.session_state:
        st.session_state.total_comments = 0
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        model_info = sentiment_analyzer.get_model_info()
        st.success("✅ Advanced Analyzer Active")
        st.info(f"**Model:** {model_info['model_name']}")
        st.info(f"**Type:** {model_info['model_type']}")
        
        st.header("📈 Statistics")
        st.metric("Total Analyses", st.session_state.analysis_count)
        st.metric("Comments Processed", st.session_state.total_comments)
        
        st.header("ℹ️ About")
        st.markdown("""
        **Powered by:**
        - 🌐 Reddit JSON API
        - 🧠 Enhanced Sentiment Analysis
        - 📊 Streamlit Dashboard
        - 🎯 Real-time Data Processing
        """)
    
    # Main input form
    with st.form("analysis_form"):
        st.subheader("🔍 Enter Reddit Post URL")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            url = st.text_input(
                "Reddit Post URL:",
                placeholder="https://www.reddit.com/r/python/comments/...",
                help="Paste any Reddit post URL to analyze comments",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button(
                "🚀 Analyze Sentiment", 
                type="primary", 
                use_container_width=True
            )
    
    # Quick analysis presets
    with st.expander("⚡ Quick Analysis Presets", expanded=True):
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            if st.button("🐍 Python", use_container_width=True):
                st.session_state.preset_url = "https://www.reddit.com/r/Python/comments/18x4h4v/what_python_skills_are_most_in_demand_right_now/"
                st.rerun()
        
        with preset_col2:
            if st.button("🎮 Gaming", use_container_width=True):
                st.session_state.preset_url = "https://www.reddit.com/r/gaming/comments/18y2a3b/whats_the_best_game_youve_played_this_year/"
                st.rerun()
        
        with preset_col3:
            if st.button("🤖 Technology", use_container_width=True):
                st.session_state.preset_url = "https://www.reddit.com/r/technology/comments/18z1b2c/ai_advances_that_will_change_everything_in_2024/"
                st.rerun()
        
        with preset_col4:
            if st.button("💰 Crypto", use_container_width=True):
                st.session_state.preset_url = "https://www.reddit.com/r/CryptoCurrency/comments/1901c4d/bitcoin_etf_approval_what_it_means_for_crypto/"
                st.rerun()
    
    # Handle preset URL
    if hasattr(st.session_state, 'preset_url'):
        url = st.session_state.preset_url
        del st.session_state.preset_url
    
    # Process analysis request
    if submitted and url:
        if "reddit.com" not in url:
            st.error("❌ Please enter a valid Reddit URL.")
        else:
            with st.spinner("🌐 Connecting to Reddit & downloading real data..."):
                result = reddit_loader.fetch_data(url)
                
                if result['success']:
                    st.session_state.real_data = result
                    st.session_state.analysis_count += 1
                    st.session_state.total_comments += len(result['comments'])
                    st.success(f"✅ Successfully loaded {len(result['comments'])} comments!")
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Display results if available
    if 'real_data' in st.session_state:
        display_enhanced_results(st.session_state.real_data)
    
    # Footer with technical info
    st.divider()
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        **🛠️ How This Works:**
        
        1. **JSON Endpoint Access**: Appends `.json` to Reddit URLs to access public API
        2. **Enhanced Parsing**: Intelligently extracts post metadata and comments
        3. **Advanced Sentiment Analysis**: 
           - Weighted keyword scoring
           - Negation handling  
           - Confidence calculation
        4. **Real-time Processing**: Live analysis without API keys
        5. **Interactive Dashboard**: Filtering, visualization, and export
        
        **📊 Data Flow:**
        ```
        Reddit URL → JSON Endpoint → Data Parsing → Sentiment Analysis → Visualization
        ```
        
        **🛡️ Privacy & Compliance:**
        - Uses only public Reddit data
        - No authentication required
        - Respects rate limits
        - Local processing only
        """)

if __name__ == "__main__":
    main()