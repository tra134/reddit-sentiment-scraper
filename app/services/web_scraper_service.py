from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
import requests
import pandas as pd
import time
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import random
import re
from urllib.parse import urljoin, urlparse
import os

from config import Config
from app.models.scraped_models import RedditPost, RedditComment, ScrapingSession
from app.utils.browser_manager import BrowserManager
from app.utils.rate_limiter import RateLimiter
from app.utils.error_handler import ScrapingErrorHandler
from app.services.proxy_service import ProxyService

logger = logging.getLogger(__name__)

class AdvancedWebScraperService:
    """Advanced web scraping service for Reddit with anti-detection features"""
    
    def __init__(self, use_proxies: bool = True):
        self.config = Config.get_scraping_config()
        self.browser_config = Config.get_browser_config()
        self.use_proxies = use_proxies
        
        # Initialize services
        self.browser_manager = BrowserManager()
        self.rate_limiter = RateLimiter(max_requests=30, time_window=60)  # 30 requests/minute
        self.error_handler = ScrapingErrorHandler()
        
        if use_proxies and Config.PROXY['enabled']:
            self.proxy_service = ProxyService()
        else:
            self.proxy_service = None
        
        self.driver = None
        self.current_session = None
        
    def initialize_scraper(self) -> bool:
        """Initialize the web scraper with configured settings"""
        try:
            self.driver = self.browser_manager.create_driver(
                headless=self.config['headless'],
                user_agent=self.browser_config['user_agent'],
                disable_images=self.browser_config['disable_images']
            )
            
            if self.proxy_service and self.use_proxies:
                proxy = self.proxy_service.get_working_proxy()
                if proxy:
                    self.browser_manager.set_proxy(self.driver, proxy)
            
            self.current_session = ScrapingSession(
                started_at=datetime.now(),
                status='active',
                pages_scraped=0
            )
            
            logger.info("✅ Web scraper initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize web scraper: {e}")
            return False
    
    def scrape_reddit_post(self, post_url: str, max_comments: int = None) -> Dict[str, Any]:
        """
        Scrape a Reddit post and its comments
        """
        if max_comments is None:
            max_comments = self.config['max_comments_per_post']
        
        start_time = time.time()
        
        try:
            if not self.driver:
                self.initialize_scraper()
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            logger.info(f"🕸️ Scraping post: {post_url}")
            
            # Navigate to post
            self.driver.get(post_url)
            time.sleep(3)  # Initial load wait
            
            # Check for blocking
            if self._is_blocked():
                logger.warning("🚫 Blocked by Reddit, rotating proxy...")
                if self._rotate_proxy():
                    self.driver.get(post_url)
                    time.sleep(3)
                else:
                    raise ScrapingError("Unable to bypass blocking")
            
            # Extract post data
            post_data = self._extract_post_data()
            
            # Scroll to load comments
            self._scroll_to_load_comments()
            
            # Extract comments
            comments = self._extract_comments(max_comments)
            
            execution_time = round(time.time() - start_time, 2)
            
            return {
                'success': True,
                'message': f"Successfully scraped {len(comments)} comments",
                'post_data': post_data,
                'comments': comments,
                'comments_count': len(comments),
                'execution_time': execution_time,
                'post_id': post_data.get('post_id'),
                'session_id': self.current_session.id if self.current_session else None
            }
            
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            execution_time = round(time.time() - start_time, 2)
            return {
                'success': False,
                'message': f"Scraping error: {str(e)}",
                'post_data': None,
                'comments': [],
                'comments_count': 0,
                'execution_time': execution_time
            }
    
    def _extract_post_data(self) -> Dict[str, Any]:
        """Extract detailed post information"""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Multiple selectors for robustness
            title_selectors = [
                'h1[class*="Post"]',
                '[data-adclicklocation="title"]',
                'div[class*="Post"] h1',
                'shreddit-post h1'
            ]
            
            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract other post details
            post_data = {
                'post_id': self._extract_post_id(),
                'title': title,
                'author': self._extract_author(soup),
                'subreddit': self._extract_subreddit(soup),
                'score': self._extract_score(soup),
                'upvote_ratio': self._extract_upvote_ratio(soup),
                'num_comments': self._extract_comment_count(soup),
                'created_utc': self._extract_post_time(soup),
                'url': self.driver.current_url,
                'permalink': self.driver.current_url,
                'content': self._extract_post_content(soup),
                'flair': self._extract_flair(soup),
                'awards': self._extract_awards(soup)
            }
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return {}
    
    def _extract_comments(self, max_comments: int) -> List[Dict[str, Any]]:
        """Extract comments from the post"""
        comments = []
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Multiple comment selectors
            comment_selectors = [
                'div[class*="Comment"]',
                'shreddit-comment',
                '[data-testid="comment"]',
                'div[thingid*="t1_"]'  # Comment thing ID
            ]
            
            comment_elements = []
            for selector in comment_selectors:
                elements = soup.select(selector)
                if elements:
                    comment_elements = elements
                    break
            
            for i, comment_elem in enumerate(comment_elements[:max_comments]):
                try:
                    comment_data = self._extract_single_comment(comment_elem)
                    if comment_data and comment_data.get('body'):
                        comments.append(comment_data)
                    
                    # Progress logging
                    if (i + 1) % 50 == 0:
                        logger.info(f"📝 Extracted {i + 1} comments...")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract comment {i}: {e}")
                    continue
            
            return comments
            
        except Exception as e:
            logger.error(f"Error extracting comments: {e}")
            return []
    
    def _extract_single_comment(self, comment_elem) -> Dict[str, Any]:
        """Extract data from a single comment element"""
        try:
            # Comment ID
            comment_id = self._extract_comment_id(comment_elem)
            
            # Author
            author_selectors = [
                'a[href*="/user/"]',
                '[class*="Author"]',
                '[data-testid="comment_author"]'
            ]
            
            author = None
            for selector in author_selectors:
                author_elem = comment_elem.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    break
            
            # Comment body
            body_selectors = [
                'div[class*="CommentBody"]',
                '[data-testid="comment"]',
                'div[class*="md"]'
            ]
            
            body = None
            for selector in body_selectors:
                body_elem = comment_elem.select_one(selector)
                if body_elem:
                    body = body_elem.get_text(strip=True)
                    break
            
            # Score
            score_selectors = [
                'div[class*="Score"]',
                '[id*="vote-arrows"]',
                '[data-testid="vote-arrows"]'
            ]
            
            score = 0
            for selector in score_selectors:
                score_elem = comment_elem.select_one(selector)
                if score_elem:
                    score_text = score_elem.get_text(strip=True)
                    # Extract numbers from score text
                    numbers = re.findall(r'\d+', score_text)
                    if numbers:
                        score = int(numbers[0])
                        break
            
            # Timestamp
            time_selectors = [
                'time',
                '[data-testid="comment_timestamp"]',
                'a[data-click-id="timestamp"]'
            ]
            
            created_utc = datetime.now()
            for selector in time_selectors:
                time_elem = comment_elem.select_one(selector)
                if time_elem:
                    time_text = time_elem.get('datetime') or time_elem.get_text(strip=True)
                    if time_text:
                        created_utc = self._parse_timestamp(time_text)
                    break
            
            return {
                'comment_id': comment_id or f"unknown_{int(time.time())}_{random.randint(1000, 9999)}",
                'author': author or '[deleted]',
                'body': body or '',
                'score': score,
                'created_utc': created_utc,
                'permalink': f"{self.driver.current_url}{comment_id}" if comment_id else self.driver.current_url
            }
            
        except Exception as e:
            logger.warning(f"Error extracting single comment: {e}")
            return {}
    
    def _scroll_to_load_comments(self):
        """Scroll to load all comments"""
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            for i in range(10):  # Max 10 scroll attempts
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(self.config['scroll_pause_time'])
                
                # Check if we've reached the bottom
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
                logger.info(f"📜 Scrolling... ({i + 1}/10)")
                
        except Exception as e:
            logger.warning(f"Scrolling interrupted: {e}")
    
    def _is_blocked(self) -> bool:
        """Check if we're being blocked by Reddit"""
        blocked_indicators = [
            "sorry, we have failed you",
            "rate limited",
            "access denied",
            "cloudflare",
            "captcha"
        ]
        
        page_source = self.driver.page_source.lower()
        return any(indicator in page_source for indicator in blocked_indicators)
    
    def _rotate_proxy(self) -> bool:
        """Rotate to a new proxy"""
        if not self.proxy_service:
            return False
        
        try:
            new_proxy = self.proxy_service.get_working_proxy()
            if new_proxy:
                self.browser_manager.set_proxy(self.driver, new_proxy)
                logger.info(f"🔄 Rotated to new proxy: {new_proxy}")
                return True
        except Exception as e:
            logger.error(f"Proxy rotation failed: {e}")
        
        return False
    
    # Helper extraction methods
    def _extract_post_id(self) -> str:
        """Extract post ID from URL"""
        try:
            url_parts = self.driver.current_url.split('/')
            if 'comments' in url_parts:
                post_index = url_parts.index('comments') + 1
                if post_index < len(url_parts):
                    return url_parts[post_index]
        except:
            pass
        return f"scraped_{int(time.time())}"
    
    def _extract_author(self, soup) -> str:
        """Extract post author"""
        author_selectors = [
            'a[href*="/user/"]',
            '[data-testid="post_author"]',
            '[class*="Author"]'
        ]
        
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)
        return '[deleted]'
    
    def _extract_subreddit(self, soup) -> str:
        """Extract subreddit name"""
        subreddit_selectors = [
            'a[href*="/r/"]',
            '[data-testid="subreddit"]',
            '[class*="Subreddit"]'
        ]
        
        for selector in subreddit_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if text.startswith('r/'):
                    return text[2:]
                return text
        return 'unknown'
    
    def _extract_score(self, soup) -> int:
        """Extract post score"""
        score_selectors = [
            'div[class*="Score"]',
            '[data-testid="post-score"]',
            '[id*="vote-arrows"]'
        ]
        
        for selector in score_selectors:
            elem = soup.select_one(selector)
            if elem:
                score_text = elem.get_text(strip=True)
                numbers = re.findall(r'\d+', score_text)
                if numbers:
                    return int(numbers[0])
        return 0
    
    def _extract_upvote_ratio(self, soup) -> float:
        """Extract upvote ratio"""
        # This is tricky without API, we'll estimate
        return 0.85  # Default estimate
    
    def _extract_comment_count(self, soup) -> int:
        """Extract comment count"""
        count_selectors = [
            'a[href*="#comment"]',
            '[data-testid="comments-count"]',
            '[class*="Comments"]'
        ]
        
        for selector in count_selectors:
            elem = soup.select_one(selector)
            if elem:
                count_text = elem.get_text(strip=True)
                numbers = re.findall(r'\d+', count_text)
                if numbers:
                    return int(numbers[0])
        return 0
    
    def _extract_post_time(self, soup) -> datetime:
        """Extract post timestamp"""
        time_selectors = [
            'time',
            '[data-testid="post_timestamp"]',
            'a[data-click-id="timestamp"]'
        ]
        
        for selector in time_selectors:
            elem = soup.select_one(selector)
            if elem:
                time_text = elem.get('datetime') or elem.get_text(strip=True)
                if time_text:
                    return self._parse_timestamp(time_text)
        
        return datetime.now()
    
    def _extract_post_content(self, soup) -> str:
        """Extract post content/body"""
        content_selectors = [
            'div[class*="PostContent"]',
            '[data-testid="post-content"]',
            'div[class*="md"]'
        ]
        
        for selector in content_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)
        return ''
    
    def _extract_flair(self, soup) -> str:
        """Extract post flair"""
        flair_selectors = [
            '[class*="Flair"]',
            '[data-testid="post-flair"]'
        ]
        
        for selector in flair_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)
        return ''
    
    def _extract_awards(self, soup) -> List[str]:
        """Extract post awards"""
        award_selectors = [
            '[class*="Award"]',
            '[data-testid="awards"]'
        ]
        
        awards = []
        for selector in award_selectors:
            elems = soup.select(selector)
            for elem in elems:
                award_text = elem.get_text(strip=True)
                if award_text:
                    awards.append(award_text)
        return awards
    
    def _extract_comment_id(self, comment_elem) -> Optional[str]:
        """Extract comment ID"""
        # Try to get from element ID or data attributes
        for attr in ['id', 'data-thing-id', 'data-comment-id']:
            if comment_elem.has_attr(attr):
                value = comment_elem[attr]
                if 't1_' in value:
                    return value
                elif value:
                    return f"t1_{value}"
        
        return None
    
    def _parse_timestamp(self, time_text: str) -> datetime:
        """Parse various timestamp formats"""
        try:
            # ISO format
            if 'T' in time_text and 'Z' in time_text:
                return datetime.fromisoformat(time_text.replace('Z', '+00:00'))
            
            # Relative time (e.g., "5 hours ago")
            now = datetime.now()
            if 'ago' in time_text.lower():
                numbers = re.findall(r'\d+', time_text)
                if numbers:
                    num = int(numbers[0])
                    if 'minute' in time_text:
                        return now - timedelta(minutes=num)
                    elif 'hour' in time_text:
                        return now - timedelta(hours=num)
                    elif 'day' in time_text:
                        return now - timedelta(days=num)
                    elif 'week' in time_text:
                        return now - timedelta(weeks=num)
                    elif 'month' in time_text:
                        return now - timedelta(days=num * 30)
                    elif 'year' in time_text:
                        return now - timedelta(days=num * 365)
            
            return now
            
        except:
            return datetime.now()
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        if self.current_session:
            self.current_session.status = 'completed'
            self.current_session.ended_at = datetime.now()

# Custom exception
class ScrapingError(Exception):
    pass