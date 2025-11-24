from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

from config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()

class RedditPost(Base):
    """Model for storing Reddit post information"""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
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

class Comment(Base):
    """Model for storing Reddit comments"""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
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

class AnalysisResult(Base):
    """Model for storing analysis results"""
    
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(100), ForeignKey('reddit_posts.post_id'), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)
    results = Column(JSON, nullable=False)
    summary = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    post = relationship("RedditPost", back_populates="analyses")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'post_id': self.post_id,
            'analysis_type': self.analysis_type,
            'results': self.results,
            'summary': self.summary,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ScrapingSession(Base):
    """Model for tracking scraping sessions"""
    
    __tablename__ = 'scraping_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    status = Column(String(20), default='running')
    posts_scraped = Column(Integer, default=0)
    comments_scraped = Column(Integer, default=0)
    errors_encountered = Column(Integer, default=0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'posts_scraped': self.posts_scraped,
            'comments_scraped': self.comments_scraped,
            'errors_encountered': self.errors_encountered
        }

class DatabaseManager:
    """Database management class with CRUD operations"""
    
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_post(self, post_data: Dict[str, Any]) -> Optional[RedditPost]:
        """Save or update Reddit post"""
        session = self.get_session()
        try:
            # Check if post exists
            existing_post = session.query(RedditPost).filter(
                RedditPost.post_id == post_data['post_id']
            ).first()
            
            if existing_post:
                # Update existing post
                for key, value in post_data.items():
                    setattr(existing_post, key, value)
                post = existing_post
            else:
                # Create new post
                post = RedditPost(**post_data)
                session.add(post)
            
            session.commit()
            return post
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving post: {e}")
            return None
        finally:
            session.close()
    
    def save_comments(self, comments_data: List[Dict[str, Any]]) -> int:
        """Save multiple comments"""
        session = self.get_session()
        saved_count = 0
        
        try:
            for comment_data in comments_data:
                # Check if comment exists
                existing_comment = session.query(Comment).filter(
                    Comment.comment_id == comment_data['comment_id']
                ).first()
                
                if existing_comment:
                    # Update existing comment
                    for key, value in comment_data.items():
                        setattr(existing_comment, key, value)
                else:
                    # Create new comment
                    comment = Comment(**comment_data)
                    session.add(comment)
                    saved_count += 1
            
            session.commit()
            return saved_count
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving comments: {e}")
            return 0
        finally:
            session.close()
    
    def get_post_by_id(self, post_id: str) -> Optional[RedditPost]:
        """Get post by post_id"""
        session = self.get_session()
        try:
            return session.query(RedditPost).filter(RedditPost.post_id == post_id).first()
        finally:
            session.close()
    
    def get_comments_by_post(self, post_id: str, limit: int = 1000) -> List[Comment]:
        """Get comments for a specific post"""
        session = self.get_session()
        try:
            return session.query(Comment).filter(
                Comment.post_id == post_id
            ).order_by(Comment.score.desc()).limit(limit).all()
        finally:
            session.close()
    
    def save_analysis_result(self, analysis_data: Dict[str, Any]) -> bool:
        """Save analysis results"""
        session = self.get_session()
        try:
            analysis = AnalysisResult(**analysis_data)
            session.add(analysis)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving analysis result: {e}")
            return False
        finally:
            session.close()
    
    def get_analysis_results(self, post_id: str, analysis_type: str = None) -> List[AnalysisResult]:
        """Get analysis results for a post"""
        session = self.get_session()
        try:
            query = session.query(AnalysisResult).filter(AnalysisResult.post_id == post_id)
            if analysis_type:
                query = query.filter(AnalysisResult.analysis_type == analysis_type)
            return query.order_by(AnalysisResult.created_at.desc()).all()
        finally:
            session.close()
    
    def create_scraping_session(self, session_id: str) -> ScrapingSession:
        """Create a new scraping session"""
        session = self.get_session()
        try:
            scraping_session = ScrapingSession(session_id=session_id)
            session.add(scraping_session)
            session.commit()
            return scraping_session
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error creating scraping session: {e}")
            raise
        finally:
            session.close()
    
    def update_scraping_session(self, session_id: str, **kwargs) -> bool:
        """Update scraping session"""
        session = self.get_session()
        try:
            scraping_session = session.query(ScrapingSession).filter(
                ScrapingSession.session_id == session_id
            ).first()
            
            if scraping_session:
                for key, value in kwargs.items():
                    setattr(scraping_session, key, value)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error updating scraping session: {e}")
            return False
        finally:
            session.close()

# Global database instance
db_manager = DatabaseManager()