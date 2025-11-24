from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

@dataclass
class ScrapedPost:
    """Data class for scraped Reddit post data"""
    
    post_id: str
    title: str
    author: Optional[str]
    subreddit: str
    url: str
    permalink: str
    content: Optional[str]
    score: int = 0
    upvote_ratio: float = 0.0
    num_comments: int = 0
    created_utc: datetime = None
    collected_at: datetime = None
    flair: Optional[str] = None
    awards: List[str] = None
    nsfw: bool = False
    spoiler: bool = False
    locked: bool = False
    distinguished: Optional[str] = None
    
    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()
        if self.awards is None:
            self.awards = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedPost':
        """Create from dictionary"""
        return cls(**data)
    
    def is_valid(self) -> bool:
        """Check if post data is valid"""
        return all([
            self.post_id,
            self.title,
            self.subreddit,
            self.url
        ])
    
    def get_engagement_ratio(self) -> float:
        """Calculate engagement ratio"""
        if self.num_comments == 0:
            return 0.0
        return self.score / self.num_comments
    
    def get_approval_rate(self) -> float:
        """Calculate approval rate based on upvote ratio"""
        return self.upvote_ratio * 100 if self.upvote_ratio else 0.0

@dataclass
class ScrapedComment:
    """Data class for scraped Reddit comment data"""
    
    comment_id: str
    post_id: str
    author: Optional[str]
    body: str
    score: int = 0
    created_utc: datetime = None
    collected_at: datetime = None
    parent_id: Optional[str] = None
    is_submitter: bool = False
    depth: int = 0
    permalink: Optional[str] = None
    awards: List[str] = None
    controversiality: float = 0.0
    
    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()
        if self.awards is None:
            self.awards = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedComment':
        """Create from dictionary"""
        return cls(**data)
    
    def is_valid(self) -> bool:
        """Check if comment data is valid"""
        return all([
            self.comment_id,
            self.post_id,
            self.body.strip() not in ['[deleted]', '[removed]']
        ])
    
    def is_deleted(self) -> bool:
        """Check if comment is deleted or removed"""
        return self.body.strip() in ['[deleted]', '[removed]'] or self.author in ['[deleted]', None]
    
    def get_controversy_level(self) -> str:
        """Get controversy level based on controversiality score"""
        if self.controversiality > 0.7:
            return 'high'
        elif self.controversiality > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def get_body_length(self) -> int:
        """Get length of comment body"""
        return len(self.body) if self.body else 0
    
    def get_word_count(self) -> int:
        """Get word count of comment body"""
        return len(self.body.split()) if self.body else 0

@dataclass
class ScrapingSession:
    """Data class for scraping session information"""
    
    session_id: str
    start_time: datetime = None
    end_time: Optional[datetime] = None
    status: str = 'running'  # running, completed, failed
    target_url: Optional[str] = None
    posts_scraped: int = 0
    comments_scraped: int = 0
    errors_encountered: int = 0
    session_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.session_data is None:
            self.session_data = {}
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapingSession':
        """Create from dictionary"""
        return cls(**data)
    
    def mark_completed(self):
        """Mark session as completed"""
        self.status = 'completed'
        self.end_time = datetime.now()
    
    def mark_failed(self, error_message: str = None):
        """Mark session as failed"""
        self.status = 'failed'
        self.end_time = datetime.now()
        if error_message:
            self.session_data['last_error'] = error_message
    
    def increment_posts(self, count: int = 1):
        """Increment posts scraped count"""
        self.posts_scraped += count
    
    def increment_comments(self, count: int = 1):
        """Increment comments scraped count"""
        self.comments_scraped += count
    
    def increment_errors(self, count: int = 1):
        """Increment error count"""
        self.errors_encountered += count
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        end_time = self.end_time or datetime.now()
        if self.start_time:
            return (end_time - self.start_time).total_seconds()
        return None
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total_operations = self.posts_scraped + self.comments_scraped
        if total_operations == 0:
            return 0.0
        successful_operations = total_operations - self.errors_encountered
        return successful_operations / total_operations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'duration_seconds': self.get_duration(),
            'posts_scraped': self.posts_scraped,
            'comments_scraped': self.comments_scraped,
            'errors_encountered': self.errors_encountered,
            'success_rate': self.get_success_rate(),
            'status': self.status
        }

@dataclass
class ScrapingBatch:
    """Data class for batch scraping operations"""
    
    batch_id: str
    session_id: str
    urls: List[str]
    start_time: datetime = None
    end_time: Optional[datetime] = None
    status: str = 'pending'  # pending, running, completed, failed
    results: List[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def add_result(self, result: Dict[str, Any]):
        """Add scraping result"""
        self.results.append(result)
    
    def add_error(self, error: Dict[str, Any]):
        """Add scraping error"""
        self.errors.append(error)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get batch progress"""
        total = len(self.urls)
        completed = len(self.results) + len(self.errors)
        return {
            'total': total,
            'completed': completed,
            'remaining': total - completed,
            'completion_percentage': (completed / total * 100) if total > 0 else 0,
            'successful': len(self.results),
            'failed': len(self.errors)
        }