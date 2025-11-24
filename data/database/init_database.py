#!/usr/bin/env python3
"""
Database initialization script for Reddit Sentiment Analyzer Pro
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import DatabaseManager, Base
from app.models.database_models import RedditPost, Comment, AnalysisResult, ScrapingSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with required tables"""
    try:
        # This will create all tables defined in models
        db_manager = DatabaseManager()
        
        # Test connection by creating a session
        session = db_manager.get_session()
        session.close()
        
        logger.info("✅ Database initialized successfully")
        logger.info("📊 Tables created:")
        logger.info("   - reddit_posts")
        logger.info("   - comments") 
        logger.info("   - analysis_results")
        logger.info("   - scraping_sessions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    try:
        from datetime import datetime, timedelta
        import random
        
        db_manager = DatabaseManager()
        
        # Create sample post
        sample_post = RedditPost(
            post_id="sample_post_001",
            title="Sample Reddit Post for Testing",
            author="test_user",
            subreddit="testing",
            url="https://reddit.com/r/testing/comments/sample_post_001",
            permalink="/r/testing/comments/sample_post_001/sample_reddit_post_for_testing/",
            content="This is a sample post content for testing the sentiment analysis system.",
            score=150,
            upvote_ratio=0.92,
            num_comments=3,
            created_utc=datetime.now() - timedelta(days=1),
            collected_at=datetime.now(),
            is_analyzed=True
        )
        
        db_manager.save_post(sample_post.to_dict())
        
        # Create sample comments
        sample_comments = [
            {
                'comment_id': 'sample_comment_001',
                'post_id': 'sample_post_001',
                'author': 'user1',
                'body': 'This is a great post! I really enjoyed reading it.',
                'score': 25,
                'sentiment_label': 'positive',
                'sentiment_score': 0.8,
                'emotion_label': 'joy',
                'emotion_scores': {'joy': 0.7, 'neutral': 0.3},
                'created_utc': datetime.now() - timedelta(hours=23),
                'is_analyzed': True
            },
            {
                'comment_id': 'sample_comment_002',
                'post_id': 'sample_post_001',
                'author': 'user2',
                'body': 'I have some concerns about this topic.',
                'score': 5,
                'sentiment_label': 'negative',
                'sentiment_score': -0.6,
                'emotion_label': 'fear',
                'emotion_scores': {'fear': 0.5, 'neutral': 0.3, 'sadness': 0.2},
                'created_utc': datetime.now() - timedelta(hours=22),
                'is_analyzed': True
            },
            {
                'comment_id': 'sample_comment_003',
                'post_id': 'sample_post_001',
                'author': 'user3',
                'body': 'Interesting perspective. Looking forward to more discussions.',
                'score': 12,
                'sentiment_label': 'neutral',
                'sentiment_score': 0.1,
                'emotion_label': 'neutral',
                'emotion_scores': {'neutral': 0.8, 'surprise': 0.2},
                'created_utc': datetime.now() - timedelta(hours=21),
                'is_analyzed': True
            }
        ]
        
        db_manager.save_comments(sample_comments)
        
        # Create sample analysis result
        sample_analysis = {
            'post_id': 'sample_post_001',
            'analysis_type': 'sentiment_summary',
            'results': {
                'total_comments': 3,
                'sentiment_distribution': {
                    'positive': 1,
                    'negative': 1, 
                    'neutral': 1
                },
                'overall_sentiment': 'neutral',
                'average_sentiment_score': 0.1
            },
            'summary': 'Sample analysis showing balanced sentiment distribution'
        }
        
        db_manager.save_analysis_result(sample_analysis)
        
        logger.info("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample data creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Initializing Reddit Sentiment Analyzer Database...")
    
    if init_database():
        print("✅ Database setup completed successfully!")
        
        # Ask if user wants sample data
        response = input("📝 Create sample data? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if create_sample_data():
                print("✅ Sample data created successfully!")
            else:
                print("❌ Failed to create sample data")
    else:
        print("❌ Database setup failed!")
        sys.exit(1)