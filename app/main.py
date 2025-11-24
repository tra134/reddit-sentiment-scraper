import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check ML availability
try:
    import torch
    import transformers
    ML_AVAILABLE = True
    logger.info("✅ ML packages available")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️ ML packages not available: {e}")

# Simple fallback sentiment analyzer
class SimpleSentimentAnalyzer:
    """Rule-based sentiment analysis as fallback"""
    
    def __init__(self):
        logger.info("🔄 Initializing rule-based sentiment analyzer")
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'amazing', 'fantastic',
            'wonderful', 'perfect', 'love', 'like', 'nice', 'cool', 'best',
            'fantastic', 'brilliant', 'outstanding', 'superb', 'terrific',
            'happy', 'pleased', 'satisfied', 'impressed', 'recommend'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'stupid', 'ridiculous', 'useless', 'waste', 'poor',
            'disappointing', 'annoying', 'frustrating', 'broken', 'sad',
            'angry', 'mad', 'upset', 'displeased', 'garbage', 'trash'
        }
    
    def analyze(self, text):
        """Analyze sentiment and return dict with 'sentiment' key"""
        if not text or len(text.strip()) < 3:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
        
        try:
            text_lower = text.lower()
            words = text_lower.split()
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            
            total_words = len(words)
            if total_words == 0:
                return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
            
            sentiment_score = (positive_count - negative_count) / total_words
            
            if sentiment_score > 0.1:
                sentiment = 'positive'
                confidence = min(abs(sentiment_score) + 0.3, 0.95)
            elif sentiment_score < -0.1:
                sentiment = 'negative'
                confidence = min(abs(sentiment_score) + 0.3, 0.95)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return {
                'sentiment': sentiment,  # KEY: Always use 'sentiment'
                'score': float(sentiment_score),
                'confidence': float(confidence),
                'positive_words': positive_count,
                'negative_words': negative_count
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}

# Initialize analyzer
try:
    if ML_AVAILABLE:
        from app.ml.sentiment_analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("✅ ML Sentiment Analyzer loaded")
    else:
        sentiment_analyzer = SimpleSentimentAnalyzer()
        logger.info("✅ Rule-based Sentiment Analyzer loaded")
except Exception as e:
    logger.error(f"❌ Failed to initialize sentiment analyzer: {e}")
    sentiment_analyzer = SimpleSentimentAnalyzer()
    logger.info("🔄 Using fallback rule-based analyzer")

def main():
    st.set_page_config(
        page_title="Reddit Sentiment Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Reddit Sentiment Analyzer Pro")
    st.markdown("""
    Analyze sentiment in Reddit posts and comments using advanced AI models.
    Get insights into community opinions and reactions.
    """)
    
    # Sidebar with status and info
    with st.sidebar:
        st.header("📊 Status")
        
        if ML_AVAILABLE:
            st.success("✅ AI Analysis Available")
            try:
                model_info = sentiment_analyzer.get_model_info()
                st.info(f"Model: {model_info.get('model_name', 'Unknown')}")
                st.info(f"Type: {model_info.get('model_type', 'Unknown')}")
            except:
                st.info("Model: AI Transformer")
        else:
            st.info("📝 Using Rule-based Analysis")
            st.info("Model: Rule-based Engine")
        
        st.header("⚙️ Settings")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick", "Standard", "Detailed"],
            index=1
        )
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes sentiment in Reddit content using:
        - **AI Models**: Transformer-based sentiment analysis
        - **Rule-based**: Keyword matching fallback
        - **Real-time**: Live analysis and visualization
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Analyze Post", "📈 Dashboard", "ℹ️ Help"])
    
    with tab1:
        analyze_post_tab()
    
    with tab2:
        dashboard_tab()
    
    with tab3:
        help_tab()

def analyze_post_tab():
    """Main analysis interface"""
    st.header("📊 Analyze Reddit Post")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        post_url = st.text_input(
            "Enter Reddit Post URL:",
            placeholder="https://www.reddit.com/r/python/comments/...",
            help="Paste the full URL of any Reddit post"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_comments = st.slider("Max Comments to Analyze", 10, 100, 25)
            include_replies = st.checkbox("Include Reply Comments", value=True)
        with col2:
            min_comment_score = st.number_input("Minimum Comment Score", -10, 100, 0)
            sort_by = st.selectbox("Sort Comments By", ["Top", "New", "Controversial"])
    
    if analyze_btn and post_url:
        analyze_post(post_url, max_comments, include_replies, min_comment_score, sort_by)
    elif analyze_btn:
        st.error("❌ Please enter a Reddit URL")

def analyze_post(post_url, max_comments=25, include_replies=True, min_score=0, sort_by="Top"):
    """Analyze a Reddit post with comprehensive sentiment analysis"""
    
    # Validate URL
    if not post_url.startswith(('https://www.reddit.com/', 'https://reddit.com/')):
        st.error("❌ Please enter a valid Reddit URL")
        return
    
    with st.spinner("🕸️ Scraping Reddit post and analyzing sentiment..."):
        try:
            # Mock data - Replace this with actual Reddit API calls
            mock_data = generate_mock_reddit_data()
            
            if mock_data['success']:
                display_analysis_results(mock_data, post_url)
            else:
                st.error("❌ Failed to fetch post data")
                
        except Exception as e:
            logger.error(f"Error analyzing post: {e}")
            st.error(f"❌ Error analyzing post: {str(e)}")

def generate_mock_reddit_data():
    """Generate realistic mock Reddit data for demonstration"""
    import random
    from datetime import datetime, timedelta
    
    comments = []
    sentiments = ['positive', 'negative', 'neutral']
    comment_templates = {
        'positive': [
            "This is absolutely amazing! Great work!",
            "I love this so much, thank you for sharing!",
            "Fantastic project, very impressive work!",
            "This solved all my problems, you're a lifesaver!",
            "Incredible! Better than I expected!"
        ],
        'negative': [
            "This is terrible, completely useless.",
            "Worst implementation I've ever seen.",
            "I hate how this works, very frustrating.",
            "Poor quality, would not recommend to anyone.",
            "Disappointing results, expected much better."
        ],
        'neutral': [
            "Interesting approach, needs more testing.",
            "Looks okay but has some limitations.",
            "Standard implementation, nothing special.",
            "It works as described, no major issues.",
            "Average performance, meets basic requirements."
        ]
    }
    
    # Generate random comments
    for i in range(25):
        sentiment = random.choice(sentiments)
        template = random.choice(comment_templates[sentiment])
        comments.append({
            'body': template,
            'score': random.randint(-5, 50),
            'sentiment': sentiment,
            'author': f'user_{random.randint(1000, 9999)}',
            'timestamp': datetime.now() - timedelta(hours=random.randint(1, 168))
        })
    
    return {
        'success': True,
        'post_data': {
            'title': 'Showcase: Building a Reddit Sentiment Analyzer with Python and Streamlit',
            'author': 'python_dev',
            'subreddit': 'Python',
            'score': 247,
            'upvote_ratio': 0.92,
            'num_comments': 25,
            'created_utc': datetime.now() - timedelta(hours=24)
        },
        'comments': comments,
        'analysis_timestamp': datetime.now()
    }

def display_analysis_results(data, post_url):
    """Display comprehensive analysis results"""
    
    post_data = data['post_data']
    comments = data['comments']
    
    # Header with post info
    st.success("✅ Analysis Complete!")
    
    # Post overview
    st.header("📝 Post Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Post Score", f"{post_data['score']} ⬆️")
    with col2:
        st.metric("Comments", post_data['num_comments'])
    with col3:
        st.metric("Subreddit", f"r/{post_data['subreddit']}")
    with col4:
        st.metric("Upvote Ratio", f"{post_data['upvote_ratio'] * 100:.1f}%")
    
    st.subheader(post_data['title'])
    st.caption(f"Posted by u/{post_data['author']}")
    
    # Sentiment analysis
    st.header("📊 Sentiment Analysis")
    
    # Analyze each comment
    sentiment_results = []
    confidence_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, comment in enumerate(comments):
        status_text.text(f"Analyzing comment {i+1}/{len(comments)}...")
        progress_bar.progress((i + 1) / len(comments))
        
        try:
            analysis = sentiment_analyzer.analyze(comment['body'])
            # DEBUG: Ensure we're using the correct key
            sentiment = analysis.get('sentiment', 'neutral')
            confidence = analysis.get('confidence', 0.5)
            
            sentiment_results.append(sentiment)
            confidence_scores.append(confidence)
            
            # Add analysis to comment data
            comment['analysis'] = analysis
            comment['display_sentiment'] = sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing comment {i}: {e}")
            sentiment_results.append('neutral')
            confidence_scores.append(0.5)
            comment['display_sentiment'] = 'neutral'
    
    status_text.text("✅ Analysis complete!")
    
    # Sentiment summary
    sentiment_counts = pd.Series(sentiment_results).value_counts()
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Positive", sentiment_counts.get('positive', 0))
    with col2:
        st.metric("Negative", sentiment_counts.get('negative', 0))
    with col3:
        st.metric("Neutral", sentiment_counts.get('neutral', 0))
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        if len(sentiment_counts) > 0:
            chart_data = pd.DataFrame({
                'Count': sentiment_counts.values
            }, index=sentiment_counts.index)
            st.bar_chart(chart_data)
    
    with col2:
        # Sentiment distribution pie chart
        if len(sentiment_counts) > 0:
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comments
    st.header("💬 Comment Analysis Details")
    
    for i, comment in enumerate(comments):
        sentiment = comment.get('display_sentiment', 'neutral')
        analysis = comment.get('analysis', {})
        confidence = analysis.get('confidence', 0.5)
        
        # Color code based on sentiment
        if sentiment == 'positive':
            border_color = "🟢"
            color = "green"
        elif sentiment == 'negative':
            border_color = "🔴" 
            color = "red"
        else:
            border_color = "🟡"
            color = "gray"
        
        with st.expander(f"{border_color} Score: {comment['score']} - {sentiment.title()} (Confidence: {confidence:.2f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(comment['body'])
                st.caption(f"by u/{comment['author']}")
            
            with col2:
                # Sentiment indicator
                st.metric("Sentiment", sentiment.title())
                st.progress(confidence)
    
    # Export options
    st.header("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Download CSV Report"):
            # Create downloadable DataFrame
            df = pd.DataFrame([
                {
                    'comment': comment['body'],
                    'score': comment['score'],
                    'sentiment': comment.get('display_sentiment', 'neutral'),
                    'confidence': comment.get('analysis', {}).get('confidence', 0.5),
                    'author': comment['author']
                }
                for comment in comments
            ])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="reddit_sentiment_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🖨️ Generate PDF Report"):
            st.info("PDF export feature coming soon!")

def dashboard_tab():
    """Dashboard with analytics and insights"""
    st.header("📈 Sentiment Analytics Dashboard")
    
    # Placeholder for dashboard content
    st.info("🏗️ Dashboard under construction")
    st.write("Future features:")
    st.write("• Historical sentiment trends")
    st.write("• Subreddit comparison analytics") 
    st.write("• User sentiment patterns")
    st.write("• Real-time sentiment monitoring")

def help_tab():
    """Help and documentation"""
    st.header("ℹ️ User Guide")
    
    st.markdown("""
    ## How to Use This Tool
    
    ### 🔍 Analyzing Posts
    1. **Paste Reddit URL**: Copy any Reddit post URL and paste it in the input field
    2. **Configure Settings**: Adjust analysis depth and filters in Advanced Options
    3. **Run Analysis**: Click the "Analyze Sentiment" button
    4. **Review Results**: Explore sentiment breakdown and individual comments
    
    ### 📊 Understanding Results
    
    - **Sentiment Scores**: Positive, Negative, or Neutral classification
    - **Confidence Levels**: How certain the model is about each classification
    - **Comment Analysis**: Detailed breakdown of each comment's sentiment
    - **Visualizations**: Charts and graphs showing sentiment distribution
    
    ### ⚙️ Advanced Features
    
    - **AI-Powered Analysis**: Uses transformer models for accurate sentiment detection
    - **Rule-based Fallback**: Keyword matching when AI models are unavailable
    - **Export Options**: Download results as CSV for further analysis
    - **Custom Filters**: Filter by comment score, sort order, and more
    
    ### ❓ Troubleshooting
    
    **Common Issues:**
    - **ML Models Not Loading**: The app will automatically use rule-based analysis
    - **Connection Errors**: Check your internet connection and Reddit URL
    - **Slow Analysis**: Large posts with many comments may take longer to process
    
    **Need Help?**
    - Check the console for detailed error messages
    - Ensure all required packages are installed
    - Verify Reddit URLs are properly formatted
    """)

# Import for visualizations (optional)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, using basic charts")

if __name__ == "__main__":
    main()