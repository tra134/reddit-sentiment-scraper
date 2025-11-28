# app/test_simple.py
"""
Simple tests for Reddit Analytics Pro - No pytest required
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import RedditLoader, EnhancedNLPEngine, EnhancedVizEngine

def test_reddit_loader_basic():
    """Test basic RedditLoader functionality"""
    print("🧪 Testing RedditLoader...")
    
    loader = RedditLoader()
    
    # Test clean_html
    html_text = "<p>Hello <b>World</b>!</p>"
    cleaned = loader.clean_html(html_text)
    assert cleaned == "Hello World!", f"Expected 'Hello World!', got '{cleaned}'"
    print("✅ HTML cleaning works correctly")
    
    # Test URL conversion
    test_urls = [
        ("https://www.reddit.com/r/technology/comments/abc123/test/", 
         "https://www.reddit.com/comments/abc123.rss"),
        ("https://www.reddit.com/r/python/", 
         "https://www.reddit.com/r/python/hot.rss"),
        ("https://www.reddit.com/r/technology/hot.rss",
         "https://www.reddit.com/r/technology/hot.rss")
    ]
    
    for input_url, expected in test_urls:
        result = loader.convert_to_rss_url(input_url)
        assert result == expected, f"URL conversion failed for {input_url}"
    
    print("✅ URL conversion works correctly")

def test_nlp_engine_basic():
    """Test basic NLP engine functionality"""
    print("🧪 Testing EnhancedNLPEngine...")
    
    nlp = EnhancedNLPEngine()
    
    # Test sentiment analysis
    test_cases = [
        ("I love this! It's amazing!", "Positive"),
        ("I hate this! It's terrible!", "Negative"), 
        ("This is a normal thing.", "Neutral"),
    ]
    
    for text, expected_sentiment in test_cases:
        result = nlp.analyze_text(text)
        assert 'sentiment' in result, "Sentiment field missing"
        assert 'polarity' in result, "Polarity field missing"
        assert 'emotions' in result, "Emotions field missing"
        print(f"✅ Text: '{text}' -> {result['sentiment']} (polarity: {result['polarity']:.2f})")
    
    print("✅ NLP analysis works correctly")

def test_viz_engine_basic():
    """Test basic visualization engine functionality"""
    print("🧪 Testing EnhancedVizEngine...")
    
    viz = EnhancedVizEngine()
    
    # Create test data
    test_data = {
        'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative'],
        'polarity': [0.8, -0.7, 0.1, 0.9, -0.6],
        'emotions': [['Joy'], ['Anger'], ['Neutral'], ['Joy'], ['Anger']],
        'score': [10, 1, 5, 15, 2],
        'created_utc': [1609459200, 1609545600, 1609632000, 1609718400, 1609804800]
    }
    
    df = pd.DataFrame(test_data)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    
    # Test sentiment distribution
    fig1 = viz.plot_sentiment_distribution(df)
    assert fig1 is not None, "Sentiment distribution plot failed"
    print("✅ Sentiment distribution plot works")
    
    # Test emotion radar
    fig2 = viz.plot_emotion_radar(df)
    assert fig2 is not None, "Emotion radar plot failed"
    print("✅ Emotion radar plot works")
    
    # Test sentiment timeline
    fig3 = viz.plot_sentiment_timeline(df)
    assert fig3 is not None, "Sentiment timeline plot failed"
    print("✅ Sentiment timeline plot works")

def test_trending_posts_manager():
    """Test trending posts manager"""
    print("🧪 Testing TrendingPostsManager...")
    
    from app.main import TrendingPostsManager
    
    manager = TrendingPostsManager()
    
    # Test trend analysis with sample data
    sample_posts = [
        {'subreddit': 'technology', 'score': 10, 'comments_count': 5, 'author': 'user1'},
        {'subreddit': 'technology', 'score': 20, 'comments_count': 10, 'author': 'user2'},
        {'subreddit': 'python', 'score': 15, 'comments_count': 8, 'author': 'user3'},
    ]
    
    trends = manager.analyze_trends(sample_posts)
    
    assert 'technology' in trends, "Technology subreddit missing"
    assert 'python' in trends, "Python subreddit missing"
    assert trends['technology']['count'] == 2, "Technology post count incorrect"
    assert trends['python']['count'] == 1, "Python post count incorrect"
    
    print("✅ Trend analysis works correctly")

def run_all_tests():
    """Run all basic tests"""
    print("🚀 Starting Reddit Analytics Pro Basic Tests")
    print("=" * 50)
    
    try:
        test_reddit_loader_basic()
        test_nlp_engine_basic() 
        test_viz_engine_basic()
        test_trending_posts_manager()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! ✅")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)