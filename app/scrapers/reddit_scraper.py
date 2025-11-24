from typing import List, Dict, Any, Optional
import logging
import re
from urllib.parse import urlparse

from .base_scraper import BaseScraper
from app.services.web_scraper_service import AdvancedWebScraperService

logger = logging.getLogger(__name__)

class RedditScraper(BaseScraper):
    """Reddit-specific scraper implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("reddit_scraper", config)
        self.web_scraper = AdvancedWebScraperService(
            use_proxies=config.get('use_proxies', True) if config else True
        )
        self.setup_reddit_patterns()
    
    def setup_reddit_patterns(self):
        """Setup Reddit-specific URL patterns"""
        self.post_patterns = [
            r'https?://(?:www\.)?reddit\.com/r/\w+/comments/[\w-]+/',
            r'https?://(?:www\.)?reddit\.com/r/\w+/comments/[\w-]+/[\w-]+/',
            r'https?://redd\.it/[\w]+'
        ]
        
        self.subreddit_patterns = [
            r'https?://(?:www\.)?reddit\.com/r/(\w+)/?',
            r'https?://(?:old\.)?reddit\.com/r/(\w+)/?'
        ]
    
    def validate_target(self, target: str) -> bool:
        """Validate if target is a valid Reddit URL"""
        if not target:
            return False
        
        # Check if it matches any Reddit post pattern
        for pattern in self.post_patterns:
            if re.match(pattern, target, re.IGNORECASE):
                return True
        
        # Check if it matches subreddit pattern
        for pattern in self.subreddit_patterns:
            if re.match(pattern, target, re.IGNORECASE):
                return True
        
        return False
    
    def scrape(self, target: str, **kwargs) -> Dict[str, Any]:
        """Scrape Reddit content"""
        try:
            self.logger.info(f"Scraping Reddit target: {target}")
            
            # Pre-process target
            processed_target = self.pre_process(target)
            
            # Determine content type and scrape accordingly
            if self._is_post_url(processed_target):
                return self._scrape_post(processed_target, **kwargs)
            elif self._is_subreddit_url(processed_target):
                return self._scrape_subreddit(processed_target, **kwargs)
            else:
                return self.handle_error(
                    ValueError("Unsupported Reddit URL type"), 
                    processed_target
                )
                
        except Exception as e:
            return self.handle_error(e, target)
    
    def _is_post_url(self, url: str) -> bool:
        """Check if URL is a Reddit post"""
        return any(re.match(pattern, url, re.IGNORECASE) for pattern in self.post_patterns)
    
    def _is_subreddit_url(self, url: str) -> bool:
        """Check if URL is a subreddit"""
        return any(re.match(pattern, url, re.IGNORECASE) for pattern in self.subreddit_patterns)
    
    def _scrape_post(self, post_url: str, **kwargs) -> Dict[str, Any]:
        """Scrape a single Reddit post"""
        max_comments = kwargs.get('max_comments', 500)
        
        result = self.web_scraper.scrape_reddit_post(post_url, max_comments)
        return self.post_process(result)
    
    def _scrape_subreddit(self, subreddit_url: str, **kwargs) -> Dict[str, Any]:
        """Scrape a subreddit (multiple posts)"""
        limit = kwargs.get('limit', 10)
        sort_by = kwargs.get('sort_by', 'hot')
        
        # Extract subreddit name from URL
        subreddit_name = self._extract_subreddit_name(subreddit_url)
        
        if not subreddit_name:
            return self.handle_error(
                ValueError("Could not extract subreddit name from URL"),
                subreddit_url
            )
        
        # This would be implemented to scrape multiple posts from a subreddit
        # For now, return a placeholder result
        return self.post_process({
            'success': True,
            'subreddit': subreddit_name,
            'posts_scraped': 0,
            'message': f"Subreddit scraping for r/{subreddit_name} would be implemented here",
            'content_type': 'subreddit'
        })
    
    def _extract_subreddit_name(self, url: str) -> Optional[str]:
        """Extract subreddit name from URL"""
        for pattern in self.subreddit_patterns:
            match = re.match(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add Reddit-specific post-processing"""
        processed_data = super().post_process(data)
        
        # Add Reddit-specific metadata
        if processed_data.get('success'):
            processed_data['platform'] = 'reddit'
            processed_data['content_type'] = self._determine_content_type(processed_data)
        
        return processed_data
    
    def _determine_content_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of Reddit content"""
        if 'subreddit' in data and 'posts_scraped' in data:
            return 'subreddit'
        elif 'post_data' in data:
            return 'post'
        else:
            return 'unknown'