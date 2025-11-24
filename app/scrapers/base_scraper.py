from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Abstract base class for all scrapers"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.setup()
    
    def setup(self):
        """Setup the scraper - can be overridden by subclasses"""
        self.logger.info(f"Initializing {self.name} scraper")
    
    @abstractmethod
    def scrape(self, target: str, **kwargs) -> Dict[str, Any]:
        """Main scraping method to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """Validate if the target can be scraped by this scraper"""
        pass
    
    def pre_process(self, target: str) -> str:
        """Pre-process the target before scraping"""
        return target.strip()
    
    def post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process scraped data"""
        data['scraped_at'] = datetime.now().isoformat()
        data['scraper_name'] = self.name
        return data
    
    def handle_error(self, error: Exception, target: str) -> Dict[str, Any]:
        """Handle scraping errors"""
        self.logger.error(f"Error scraping {target}: {error}")
        
        return {
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'target': target,
            'scraped_at': datetime.now().isoformat(),
            'scraper_name': self.name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        return {
            'name': self.name,
            'config': self.config,
            'status': 'active'
        }