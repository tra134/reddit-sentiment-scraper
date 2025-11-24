import pytest
from datetime import datetime
from app.models.database_models import RedditPost, Comment, AnalysisResult, ScrapingSession
from app.models.scraped_models import ScrapedPost, ScrapedComment, ScrapingSession as ScrapedSession
from app.models.analysis_models import (
    SentimentResult, EmotionResult, AspectResult, AnalysisSummary, CommentAnalysis,
    SentimentLabel, EmotionLabel
)

class TestDatabaseModels:
    """Test cases for database models"""
    
    def test_reddit_post_creation(self):
        """Test RedditPost model creation"""
        post = RedditPost(
            post_id="test123",
            title="Test Post",
            author="test_user",
            subreddit="test",
            url="https://reddit.com/test",
            permalink="/r/test/comments/test123",
            content="Test content",
            score=100,
            upvote_ratio=0.95,
            num_comments=50,
            created_utc=datetime.now()
        )
        
        assert post.post_id == "test123"
        assert post.title == "Test Post"
        assert post.is_analyzed == False
    
    def test_reddit_post_to_dict(self):
        """Test RedditPost to_dict method"""
        post = RedditPost(
            post_id="test123",
            title="Test Post",
            author="test_user",
            subreddit="test",
            url="https://reddit.com/test",
            permalink="/r/test/comments/test123",
            content="Test content",
            created_utc=datetime.now()
        )
        
        post_dict = post.to_dict()
        assert isinstance(post_dict, dict)
        assert post_dict['post_id'] == "test123"
        assert 'created_utc' in post_dict
    
    def test_comment_creation(self):
        """Test Comment model creation"""
        comment = Comment(
            comment_id="comment123",
            post_id="test123",
            author="commenter",
            body="This is a test comment",
            score=10,
            sentiment_label="positive",
            sentiment_score=0.8,
            created_utc=datetime.now()
        )
        
        assert comment.comment_id == "comment123"
        assert comment.post_id == "test123"
        assert comment.sentiment_label == "positive"
    
    def test_comment_analysis_update(self):
        """Test Comment analysis update method"""
        comment = Comment(
            comment_id="comment123",
            post_id="test123",
            author="commenter",
            body="Test comment",
            created_utc=datetime.now()
        )
        
        analysis_data = {
            'sentiment_label': 'positive',
            'sentiment_score': 0.9,
            'emotion_label': 'joy',
            'emotion_scores': {'joy': 0.8, 'neutral': 0.2},
            'aspects': [{'aspect': 'quality', 'sentiment': 'positive'}]
        }
        
        comment.update_analysis_results(analysis_data)
        
        assert comment.is_analyzed == True
        assert comment.sentiment_label == 'positive'
        assert comment.sentiment_score == 0.9

class TestScrapedModels:
    """Test cases for scraped data models"""
    
    def test_scraped_post_creation(self):
        """Test ScrapedPost creation"""
        post = ScrapedPost(
            post_id="test123",
            title="Test Post",
            author="test_user",
            subreddit="test",
            url="https://reddit.com/test",
            permalink="/r/test/comments/test123",
            content="Test content"
        )
        
        assert post.is_valid() == True
        assert post.collected_at is not None
    
    def test_scraped_post_engagement(self):
        """Test ScrapedPost engagement calculations"""
        post = ScrapedPost(
            post_id="test123",
            title="Test Post",
            author="test_user",
            subreddit="test",
            url="https://reddit.com/test",
            permalink="/r/test/comments/test123",
            content="Test content",
            score=100,
            num_comments=10,
            upvote_ratio=0.95
        )
        
        assert post.get_engagement_ratio() == 10.0
        assert post.get_approval_rate() == 95.0
    
    def test_scraped_comment_validation(self):
        """Test ScrapedComment validation"""
        valid_comment = ScrapedComment(
            comment_id="comment123",
            post_id="test123",
            author="user",
            body="Valid comment content"
        )
        
        deleted_comment = ScrapedComment(
            comment_id="comment456",
            post_id="test123",
            author="[deleted]",
            body="[deleted]"
        )
        
        assert valid_comment.is_valid() == True
        assert deleted_comment.is_valid() == False
        assert deleted_comment.is_deleted() == True

class TestAnalysisModels:
    """Test cases for analysis models"""
    
    def test_sentiment_result(self):
        """Test SentimentResult functionality"""
        sentiment = SentimentResult(
            label=SentimentLabel.POSITIVE,
            score=0.8,
            confidence=0.9,
            model_used="roberta"
        )
        
        assert sentiment.is_positive() == True
        assert sentiment.is_negative() == False
        assert sentiment.get_sentiment_strength() == 0.8
        assert sentiment.get_confidence_level() == "very_high"
    
    def test_emotion_result(self):
        """Test EmotionResult functionality"""
        emotion = EmotionResult(
            dominant_emotion=EmotionLabel.JOY,
            emotion_scores={
                'joy': 0.7,
                'neutral': 0.2,
                'sadness': 0.1
            },
            model_used="emotion_model"
        )
        
        top_emotions = emotion.get_top_emotions(2)
        assert len(top_emotions) == 2
        assert top_emotions[0]['emotion'] == 'joy'
        assert emotion.has_strong_emotion() == True
    
    def test_analysis_summary(self):
        """Test AnalysisSummary functionality"""
        sentiment = SentimentResult(
            label=SentimentLabel.POSITIVE,
            score=0.7,
            confidence=0.8,
            model_used="roberta"
        )
        
        summary = AnalysisSummary(
            analysis_id="analysis123",
            post_id="post123",
            total_comments=100,
            analyzed_comments=95,
            sentiment_distribution={
                'positive': 60,
                'negative': 20,
                'neutral': 20
            },
            emotion_distribution={
                'joy': 40,
                'neutral': 30,
                'anger': 10,
                'sadness': 10,
                'surprise': 5
            },
            aspect_summary={
                'quality': {'total_mentions': 25, 'dominant_sentiment': 'positive'},
                'price': {'total_mentions': 15, 'dominant_sentiment': 'negative'}
            },
            engagement_metrics={
                'average_score': 15.5,
                'total_engagement': 1550,
                'comments_per_hour': 12.3
            },
            overall_sentiment=sentiment,
            dominant_emotion=EmotionLabel.JOY,
            analysis_duration=30.5
        )
        
        sentiment_pct = summary.get_sentiment_percentages()
        assert sentiment_pct['positive'] == 60.0
        assert summary.get_success_rate() == 95.0
        
        insights = summary.get_key_insights()
        assert len(insights) > 0
        
        recommendations = summary.get_recommendations()
        assert isinstance(recommendations, list)

if __name__ == "__main__":
    pytest.main([__file__])