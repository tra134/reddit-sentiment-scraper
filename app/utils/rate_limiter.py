import time
import threading
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting utility for controlling API and scraping requests"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = threading.Lock()
        
        # Domain-specific rate limits
        self.domain_limits = {
            'reddit.com': (30, 60),  # 30 requests per minute
            'default': (max_requests, time_window)
        }
    
    def acquire(self, domain: str = 'default') -> bool:
        """Acquire permission to make a request"""
        with self.lock:
            self._clean_old_requests()
            
            domain_limit, window = self.domain_limits.get(domain, self.domain_limits['default'])
            
            if len(self.requests) < domain_limit:
                self.requests.append(time.time())
                return True
            else:
                return False
    
    def wait_until_available(self, domain: str = 'default'):
        """Wait until a request can be made"""
        while not self.acquire(domain):
            sleep_time = self._get_wait_time(domain)
            logger.debug(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    def _get_wait_time(self, domain: str = 'default') -> float:
        """Calculate how long to wait until next request"""
        with self.lock:
            self._clean_old_requests()
            
            domain_limit, window = self.domain_limits.get(domain, self.domain_limits['default'])
            
            if len(self.requests) < domain_limit:
                return 0
            
            # Wait until the oldest request expires
            oldest_request = self.requests[0]
            wait_time = (oldest_request + window) - time.time()
            return max(wait_time, 0.1)  # Minimum 100ms
    
    def _clean_old_requests(self):
        """Remove requests outside the current time window"""
        current_time = time.time()
        self.requests = [
            req_time for req_time in self.requests
            if current_time - req_time < self.time_window
        ]
    
    def get_stats(self) -> Dict[str, any]:
        """Get current rate limiting statistics"""
        with self.lock:
            self._clean_old_requests()
            
            return {
                'current_requests': len(self.requests),
                'max_requests': self.max_requests,
                'time_window_seconds': self.time_window,
                'requests_per_minute': len(self.requests) / (self.time_window / 60),
                'oldest_request_seconds_ago': time.time() - self.requests[0] if self.requests else 0
            }
    
    def set_domain_limit(self, domain: str, max_requests: int, time_window: int):
        """Set custom rate limit for a domain"""
        self.domain_limits[domain] = (max_requests, time_window)
        logger.info(f"Set rate limit for {domain}: {max_requests} requests per {time_window} seconds")

class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on response success/failure rates"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        super().__init__(max_requests, time_window)
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 300  # 5 minutes
        
    def record_success(self):
        """Record a successful request"""
        self.success_count += 1
        self._maybe_adjust_limits()
    
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self._maybe_adjust_limits()
    
    def _maybe_adjust_limits(self):
        """Adjust rate limits based on success/failure ratio"""
        current_time = time.time()
        if current_time - self.last_adjustment < self.adjustment_interval:
            return
        
        total_requests = self.success_count + self.failure_count
        if total_requests == 0:
            return
        
        success_rate = self.success_count / total_requests
        
        # Adjust limits based on success rate
        if success_rate > 0.9:
            # Increase limits if success rate is high
            self.max_requests = min(self.max_requests * 1.1, 100)
            logger.info(f"Increased rate limit to {self.max_requests} due to high success rate")
        elif success_rate < 0.7:
            # Decrease limits if success rate is low
            self.max_requests = max(self.max_requests * 0.9, 10)
            logger.info(f"Decreased rate limit to {self.max_requests} due to low success rate")
        
        # Reset counters
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = current_time