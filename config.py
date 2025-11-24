import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import json

load_dotenv()

class Config:
    """Configuration for Web Scraper based Reddit Sentiment Analysis"""
    
    # Application
    APP_NAME = "Reddit Sentiment Scraper Pro"
    VERSION = "3.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'reddit-scraper-pro-secret-2024')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/database/reddit_data.db')
    
    # Web Scraping Configuration
    SCRAPING = {
        'headless': True,
        'timeout': 30,
        'max_retries': 3,
        'retry_delay': 2,
        'request_delay': 1.5,  # Delay between requests
        'max_comments_per_post': 1000,
        'scroll_pause_time': 2,
        'wait_for_element_timeout': 10
    }
    
    # Browser Configuration
    BROWSER = {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'window_size': '1920,1080',
        'disable_images': True,  # Faster loading
        'disable_javascript': False,
        'block_popups': True
    }
    
    # Proxy Configuration
    PROXY = {
        'enabled': True,
        'max_retries': 3,
        'timeout': 10,
        'rotate_on_failure': True,
        'proxy_file': 'proxies/proxy_list.json'
    }
    
    # ML Models Configuration
    ML_MODELS = {
        'sentiment': {
            'roberta': "cardiffnlp/twitter-roberta-base-sentiment-latest",
            'vader': "vader",
            'textblob': "textblob"
        },
        'emotion': {
            'roberta': "j-hartmann/emotion-english-distilroberta-base"
        },
        'aspect': {
            'keywords': ['price', 'quality', 'service', 'performance', 'design', 'battery', 'screen', 'camera']
        }
    }
    
    # Analysis Settings
    ANALYSIS = {
        'batch_size': 50,
        'max_text_length': 512,
        'confidence_threshold': 0.6,
        'enable_real_time': True
    }
    
    # Cache Settings
    CACHE = {
        'enabled': True,
        'duration': 3600,  # 1 hour
        'max_size': 1000
    }
    
    # Feature Flags
    FEATURES = {
        'ADVANCED_SCRAPING': True,
        'PROXY_SUPPORT': True,
        'REAL_TIME_ANALYSIS': True,
        'TREND_DETECTION': True,
        'MULTI_LANGUAGE': False,
        'EXPORT_CAPABILITIES': True
    }
    
    # Visualization
    VISUALIZATION = {
        'theme': 'plotly_dark',
        'colors': {
            'positive': '#00D26A',
            'neutral': '#FFB800', 
            'negative': '#FF4757',
            'joy': '#FF6B9D',
            'anger': '#FF4757',
            'sadness': '#5352ED',
            'fear': '#3742FA',
            'surprise': '#FF9F1A',
            'love': '#FF3838'
        },
        'chart_height': 600,
        'animation_duration': 500
    }
    
    @classmethod
    def get_scraping_config(cls) -> Dict[str, Any]:
        return cls.SCRAPING
    
    @classmethod
    def get_browser_config(cls) -> Dict[str, Any]:
        return cls.BROWSER
    
    @classmethod
    def save_config(cls):
        """Save configuration to JSON file"""
        config_dict = {
            'scraping': cls.SCRAPING,
            'browser': cls.BROWSER,
            'proxy': cls.PROXY,
            'analysis': cls.ANALYSIS
        }
        
        with open('scraper_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls):
        """Load configuration from JSON file"""
        try:
            with open('scraper_config.json', 'r') as f:
                config_dict = json.load(f)
                cls.SCRAPING.update(config_dict.get('scraping', {}))
                cls.BROWSER.update(config_dict.get('browser', {}))
                cls.PROXY.update(config_dict.get('proxy', {}))
                cls.ANALYSIS.update(config_dict.get('analysis', {}))
        except FileNotFoundError:
            cls.save_config()