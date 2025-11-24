from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class RedditPost(Base):
    """SQLAlchemy model for storing Reddit post information"""
    
    __tablename__ = 'reddit_posts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    author = Column(String(100))
    subreddit = Column(String(100), nullable=False)
    url = Column(Text, nullable=False)
    permalink = Column(Text, nullable=False)
    content = Column(Text)
    score = Column(Integer, default=0)
    upvote_ratio = Column(Float, default=0.0)
    num_comments = Column(Integer, default=0)
    created_utc = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=func.now())
    is_analyzed = Column(Boolean, default=False)
    
    # Relationships
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    analyses = relationship("AnalysisResult", back_populates="post", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<RedditPost(id={self.id}, title='{self.title[:50]}...', subreddit='{self.subreddit}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'post_id': self.post_id,
            'title': self.title,
            'author': self.author,
            'subreddit': self.subreddit,
            'url': self.url,
            'permalink': self.permalink,
            'content': self.content,
            'score': self.score,
            'upvote_ratio': self.upvote_ratio,
            'num_comments': self.num_comments,
            'created_utc': self.created_utc.isoformat() if self.created_utc else None,
            'collected_at': self.collected_at.isoformat() if self.collected_at else None,
            'is_analyzed': self.is_analyzed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RedditPost':
        """Create instance from dictionary"""
        return cls(**data)
    
    def update_from_scraped_data(self, scraped_data: Dict[str, Any]):
        """Update post from scraped data"""
        self.title = scraped_data.get('title', self.title)
        self.author = scraped_data.get('author', self.author)
        self.score = scraped_data.get('score', self.score)
        self.upvote_ratio = scraped_data.get('upvote_ratio', self.upvote_ratio)
        self.num_comments = scraped_data.get('num_comments', self.num_comments)
        self.content = scraped_data.get('content', self.content)

class Comment(Base):
    """SQLAlchemy model for storing Reddit comments"""
    
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    comment_id = Column(String(100), unique=True, nullable=False, index=True)
    post_id = Column(String(100), ForeignKey('reddit_posts.post_id'), nullable=False, index=True)
    author = Column(String(100))
    body = Column(Text, nullable=False)
    score = Column(Integer, default=0)
    sentiment_label = Column(String(20))
    sentiment_score = Column(Float)
    emotion_label = Column(String(50))
    emotion_scores = Column(JSON)
    aspects = Column(JSON)
    created_utc = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=func.now())
    is_analyzed = Column(Boolean, default=False)
    
    # Relationships
    post = relationship("RedditPost", back_populates="comments")
    
    def __repr__(self):
        return f"<Comment(id={self.id}, author='{self.author}', sentiment='{self.sentiment_label}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'comment_id': self.comment_id,
            'post_id': self.post_id,
            'author': self.author,
            'body': self.body,
            'score': self.score,
            'sentiment_label': self.sentiment_label,
            'sentiment_score': self.sentiment_score,
            'emotion_label': self.emotion_label,
            'emotion_scores': self.emotion_scores,
            'aspects': self.aspects,
            'created_utc': self.created_utc.isoformat() if self.created_utc else None,
            'collected_at': self.collected_at.isoformat() if self.collected_at else None,
            'is_analyzed': self.is_analyzed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Comment':
        """Create instance from dictionary"""
        return cls(**data)
    
    def update_analysis_results(self, analysis_data: Dict[str, Any]):
        """Update comment with analysis results"""
        self.sentiment_label = analysis_data.get('sentiment_label')
        self.sentiment_score = analysis_data.get('sentiment_score')
        self.emotion_label = analysis_data.get('emotion_label')
        self.emotion_scores = analysis_data.get('emotion_scores')
        self.aspects = analysis_data.get('aspects')
        self.is_analyzed = True
    
    def get_sentiment_confidence(self) -> float:
        """Get confidence level for sentiment analysis"""
        if self.sentiment_score is None:
            return 0.0
        return abs(self.sentiment_score)
    
    def get_top_emotion(self) -> Optional[str]:
        """Get the dominant emotion"""
        if not self.emotion_scores:
            return self.emotion_label
        return max(self.emotion_scores.items(), key=lambda x: x[1])[0] if self.emotion_scores else None

class AnalysisResult(Base):
    """SQLAlchemy model for storing analysis results"""
    
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(100), ForeignKey('reddit_posts.post_id'), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)  # 'sentiment', 'aspect', 'emotion', 'summary'
    results = Column(JSON, nullable=False)
    summary = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    post = relationship("RedditPost", back_populates="analyses")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, post_id='{self.post_id}', type='{self.analysis_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'post_id': self.post_id,
            'analysis_type': self.analysis_type,
            'results': self.results,
            'summary': self.summary,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create instance from dictionary"""
        return cls(**data)
    
    def get_metric(self, metric_path: str, default: Any = None) -> Any:
        """Get nested metric from results"""
        try:
            keys = metric_path.split('.')
            value = self.results
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class ScrapingSession(Base):
    """SQLAlchemy model for tracking scraping sessions"""
    
    __tablename__ = 'scraping_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed
    posts_scraped = Column(Integer, default=0)
    comments_scraped = Column(Integer, default=0)
    errors_encountered = Column(Integer, default=0)
    session_data = Column(JSON)  # Additional session metadata
    
    def __repr__(self):
        return f"<ScrapingSession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'posts_scraped': self.posts_scraped,
            'comments_scraped': self.comments_scraped,
            'errors_encountered': self.errors_encountered,
            'session_data': self.session_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapingSession':
        """Create instance from dictionary"""
        return cls(**data)
    
    def mark_completed(self):
        """Mark session as completed"""
        self.status = 'completed'
        self.end_time = datetime.now()
    
    def mark_failed(self, error_message: str = None):
        """Mark session as failed"""
        self.status = 'failed'
        self.end_time = datetime.now()
        if error_message and self.session_data:
            if 'errors' not in self.session_data:
                self.session_data['errors'] = []
            self.session_data['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'message': error_message
            })
    
    def increment_posts_scraped(self, count: int = 1):
        """Increment posts scraped count"""
        self.posts_scraped += count
    
    def increment_comments_scraped(self, count: int = 1):
        """Increment comments scraped count"""
        self.comments_scraped += count
    
    def increment_errors(self, count: int = 1):
        """Increment error count"""
        self.errors_encountered += count
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None
    
    def get_success_rate(self) -> float:
        """Get success rate for the session"""
        total_operations = self.posts_scraped + self.comments_scraped
        if total_operations == 0:
            return 0.0
        successful_operations = total_operations - self.errors_encountered
        return successful_operations / total_operations