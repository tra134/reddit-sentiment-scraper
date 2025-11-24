from typing import List, Dict, Any
import logging
import time

from .base_scraper import BaseScraper
from app.utils.text_processor import text_processor

logger = logging.getLogger(__name__)

class CommentScraper(BaseScraper):
    """Specialized scraper for comment extraction and processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("comment_scraper", config)
        self.setup_comment_filters()
    
    def setup_comment_filters(self):
        """Setup filters for comment processing"""
        self.min_comment_length = self.config.get('min_comment_length', 3)
        self.max_comment_length = self.config.get('max_comment_length', 1000)
        self.exclude_authors = ['AutoModerator', '[deleted]', '[removed]']
        self.exclude_flairs = ['Moderator', 'Admin', 'Bot']
    
    def validate_target(self, target: str) -> bool:
        """Validate if target can be processed by comment scraper"""
        # Comment scraper works with already scraped data, not URLs
        return isinstance(target, (dict, list))
    
    def scrape(self, target: Any, **kwargs) -> Dict[str, Any]:
        """Process and clean comments data"""
        try:
            if isinstance(target, dict):
                # Single post with comments
                return self._process_post_comments(target, **kwargs)
            elif isinstance(target, list):
                # List of comments
                return self._process_comment_list(target, **kwargs)
            else:
                raise ValueError("Unsupported target type for comment scraping")
                
        except Exception as e:
            return self.handle_error(e, str(target))
    
    def _process_post_comments(self, post_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process comments from a post data structure"""
        comments = post_data.get('comments', [])
        processed_comments = self._process_comment_list(comments, **kwargs)
        
        result = {
            'success': True,
            'post_id': post_data.get('post_id'),
            'original_comment_count': len(comments),
            'processed_comment_count': len(processed_comments.get('comments', [])),
            'comments': processed_comments.get('comments', []),
            'processing_stats': processed_comments.get('processing_stats', {})
        }
        
        return self.post_process(result)
    
    def _process_comment_list(self, comments: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Process a list of comments"""
        start_time = time.time()
        stats = {
            'total_comments': len(comments),
            'filtered_comments': 0,
            'cleaned_comments': 0,
            'errors': 0
        }
        
        processed_comments = []
        
        for comment in comments:
            try:
                processed_comment = self._process_single_comment(comment, **kwargs)
                if processed_comment:
                    processed_comments.append(processed_comment)
                    stats['cleaned_comments'] += 1
                else:
                    stats['filtered_comments'] += 1
            except Exception as e:
                stats['errors'] += 1
                self.logger.warning(f"Error processing comment: {e}")
        
        stats['processing_time'] = time.time() - start_time
        stats['success_rate'] = stats['cleaned_comments'] / max(stats['total_comments'], 1)
        
        return {
            'success': True,
            'comments': processed_comments,
            'processing_stats': stats
        }
    
    def _process_single_comment(self, comment: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Process a single comment"""
        # Apply filters
        if not self._passes_filters(comment):
            return None
        
        # Clean comment text
        cleaned_body = self._clean_comment_text(comment.get('body', ''))
        
        if not cleaned_body or len(cleaned_body) < self.min_comment_length:
            return None
        
        # Create processed comment
        processed_comment = comment.copy()
        processed_comment['cleaned_body'] = cleaned_body
        processed_comment['body_length'] = len(cleaned_body)
        processed_comment['word_count'] = len(cleaned_body.split())
        
        # Add text statistics
        text_stats = text_processor.get_text_statistics(comment.get('body', ''))
        processed_comment['text_statistics'] = text_stats
        
        # Add processing metadata
        processed_comment['processed_at'] = time.time()
        processed_comment['processing_version'] = '1.0'
        
        return processed_comment
    
    def _passes_filters(self, comment: Dict[str, Any]) -> bool:
        """Check if comment passes all filters"""
        # Author filter
        author = comment.get('author', '')
        if author in self.exclude_authors:
            return False
        
        # Length filter
        body = comment.get('body', '')
        if len(body) < self.min_comment_length or len(body) > self.max_comment_length:
            return False
        
        # Content filter (remove deleted/removed comments)
        if body in ['[deleted]', '[removed]']:
            return False
        
        # Score filter (optional)
        score = comment.get('score', 0)
        if score < self.config.get('min_score', -100):
            return False
        
        return True
    
    def _clean_comment_text(self, text: str) -> str:
        """Clean comment text using text processor"""
        return text_processor.clean_text(
            text,
            remove_stopwords=self.config.get('remove_stopwords', False),
            lemmatize=self.config.get('lemmatize', False),
            min_length=self.min_comment_length
        )
    
    def batch_process(self, comments: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Batch process comments for efficiency"""
        result = self._process_comment_list(comments, **kwargs)
        return result.get('comments', [])
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comment processing statistics"""
        return {
            'min_comment_length': self.min_comment_length,
            'max_comment_length': self.max_comment_length,
            'excluded_authors': self.exclude_authors,
            'filters_applied': ['author', 'length', 'content', 'score']
        }