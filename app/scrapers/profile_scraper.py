from typing import List, Dict, Any, Optional
import logging
import re
import time

from .base_scraper import BaseScraper
from app.services.web_scraper_service import AdvancedWebScraperService

logger = logging.getLogger(__name__)

class ProfileScraper(BaseScraper):
    """Scraper for Reddit user profiles"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("profile_scraper", config)
        self.web_scraper = AdvancedWebScraperService(
            use_proxies=config.get('use_proxies', True) if config else True
        )
        self.setup_profile_patterns()
    
    def setup_profile_patterns(self):
        """Setup Reddit profile URL patterns"""
        self.profile_patterns = [
            r'https?://(?:www\.)?reddit\.com/user/([\w-]+)/?',
            r'https?://(?:old\.)?reddit\.com/user/([\w-]+)/?',
            r'^/user/([\w-]+)$'
        ]
    
    def validate_target(self, target: str) -> bool:
        """Validate if target is a valid Reddit profile"""
        if not target:
            return False
        
        # Check if it matches any profile pattern
        for pattern in self.profile_patterns:
            if re.match(pattern, target, re.IGNORECASE):
                return True
        
        # Also accept plain usernames
        if re.match(r'^[\w-]+$', target):
            return True
        
        return False
    
    def scrape(self, target: str, **kwargs) -> Dict[str, Any]:
        """Scrape Reddit user profile"""
        try:
            self.logger.info(f"Scraping Reddit profile: {target}")
            
            # Extract username from target
            username = self._extract_username(target)
            if not username:
                return self.handle_error(ValueError("Could not extract username"), target)
            
            # Build profile URL
            profile_url = f"https://www.reddit.com/user/{username}/"
            
            # Scrape profile data
            profile_data = self._scrape_profile_data(profile_url, **kwargs)
            
            return self.post_process(profile_data)
            
        except Exception as e:
            return self.handle_error(e, target)
    
    def _extract_username(self, target: str) -> Optional[str]:
        """Extract username from various target formats"""
        # Try URL patterns first
        for pattern in self.profile_patterns:
            match = re.match(pattern, target, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If it's just a username, return as is
        if re.match(r'^[\w-]+$', target):
            return target
        
        return None
    
    def _scrape_profile_data(self, profile_url: str, **kwargs) -> Dict[str, Any]:
        """Scrape data from profile page"""
        # Note: This is a simplified implementation
        # In a real scenario, you would use the web scraper to extract profile data
        
        max_posts = kwargs.get('max_posts', 10)
        
        # This would be implemented to actually scrape the profile
        # For now, return mock data
        return {
            'success': True,
            'username': profile_url.split('/')[-2],
            'profile_url': profile_url,
            'karma': 15420,
            'account_age_days': 780,
            'post_karma': 12450,
            'comment_karma': 2970,
            'trophies': ['Verified Email', 'Three-Year Club'],
            'recent_posts': [],
            'recent_comments': [],
            'subreddits_active': ['python', 'programming', 'MachineLearning'],
            'scraped_at': time.time()
        }
    
    def scrape_multiple_profiles(self, usernames: List[str], **kwargs) -> Dict[str, Any]:
        """Scrape multiple user profiles"""
        results = {
            'success': True,
            'profiles_scraped': 0,
            'profiles': [],
            'errors': []
        }
        
        for username in usernames:
            try:
                profile_data = self.scrape(username, **kwargs)
                if profile_data.get('success'):
                    results['profiles'].append(profile_data)
                    results['profiles_scraped'] += 1
                else:
                    results['errors'].append({
                        'username': username,
                        'error': profile_data.get('error', 'Unknown error')
                    })
            except Exception as e:
                results['errors'].append({
                    'username': username,
                    'error': str(e)
                })
            
            # Rate limiting
            time.sleep(kwargs.get('delay_between_profiles', 2))
        
        return results
    
    def get_profile_statistics(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from profile data"""
        karma = profile_data.get('karma', 0)
        account_age_days = profile_data.get('account_age_days', 1)
        
        return {
            'karma_per_day': round(karma / account_age_days, 2),
            'post_ratio': round(profile_data.get('post_karma', 0) / max(karma, 1), 2),
            'comment_ratio': round(profile_data.get('comment_karma', 0) / max(karma, 1), 2),
            'engagement_level': self._calculate_engagement_level(karma, account_age_days),
            'subreddit_diversity': len(profile_data.get('subreddits_active', []))
        }
    
    def _calculate_engagement_level(self, karma: int, account_age_days: int) -> str:
        """Calculate user engagement level"""
        karma_per_day = karma / max(account_age_days, 1)
        
        if karma_per_day > 10:
            return 'high'
        elif karma_per_day > 2:
            return 'medium'
        else:
            return 'low'