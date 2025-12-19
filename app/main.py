import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
import time
import os
import sqlite3
import hashlib
import tempfile
import json
from collections import Counter
import threading
from pathlib import Path

# --- IMPORT ENHANCED UI MODULE ---
try:
    from visualizations.ui import (
        load_css,
        render_enhanced_sidebar,
        render_main_dashboard,
        render_enhanced_analysis_result,
        show_loading_animation,
        show_error_message,
        show_success_message,
        COLORS
    )
    UI_AVAILABLE = True
    print("✅ Enhanced UI module loaded successfully")
except ImportError as e:
    UI_AVAILABLE = False
    st.error(f"❌ Không thể tải module UI: {e}")

# --- IMPORT CACHE SERVICE ---
try:
    from services.cache_service import CacheService
    CACHE_SERVICE_AVAILABLE = True
    print("✅ CacheService loaded successfully")
except ImportError as e:
    CACHE_SERVICE_AVAILABLE = False
    print(f"⚠️ CacheService not available: {e}")

# --- IMPORT PROXY SERVICE ---
try:
    from services.proxy_service import ProxyService
    PROXY_SERVICE_AVAILABLE = True
    print("✅ ProxyService loaded successfully")
except ImportError as e:
    PROXY_SERVICE_AVAILABLE = False
    print(f"⚠️ ProxyService not available: {e}")

# --- IMPORT ANALYSIS SERVICE ---
try:
    from services.analysis_service import AnalysisService
    ANALYSIS_SERVICE_AVAILABLE = True
    print("✅ AnalysisService loaded successfully")
    
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        print("✅ Gemini API key found")
    else:
        print("⚠️ Gemini API key not found. Text summarization will use fallback methods.")
    
except ImportError as e:
    print(f"⚠️ AnalysisService not available: {e}")
    # Create a simple fallback
    class AnalysisService:
        def analyze_text(self, text):
            """Simple fallback sentiment analysis"""
            if not text:
                return {"sentiment": "neutral", "score": 0}
            
            positive_words = ['tốt', 'tuyệt', 'xuất sắc', 'hoàn hảo', 'thích', 'hài lòng']
            negative_words = ['xấu', 'tệ', 'dở', 'không thích', 'thất vọng', 'kém']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return {"sentiment": "positive", "score": 0.5}
            elif neg_count > pos_count:
                return {"sentiment": "negative", "score": -0.5}
            else:
                return {"sentiment": "neutral", "score": 0}
    
    ANALYSIS_SERVICE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Reddit Sentiment Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. DATABASE MANAGER WITH SUBREDDIT GROUPS
# ==========================================
db_lock = threading.Lock()

class DBManager:
    def __init__(self, db_name="reddit_sentiment.db"):
        # FIX: Streamlit Cloud compatible database path
        if os.environ.get('STREAMLIT_CLOUD') or 'STREAMLIT_SHARING' in os.environ:
            temp_dir = tempfile.gettempdir()
            db_path = os.path.join(temp_dir, db_name)
            print(f"🌐 Running on Streamlit Cloud. Database path: {db_path}")
        else:
            db_path = db_name
            print(f"💻 Running locally. Database path: {db_path}")
            
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with db_lock:
            c = self.conn.cursor()
            # Users table
            c.execute('''CREATE TABLE IF NOT EXISTS users 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          username TEXT UNIQUE, 
                          password TEXT,
                          created_at TEXT)''')
            
            # Subreddit groups table
            c.execute('''CREATE TABLE IF NOT EXISTS subreddit_groups 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          user_id INTEGER,
                          subreddit_name TEXT,
                          added_date TEXT,
                          last_updated TEXT)''')
            
            # Analysis history table
            c.execute('''CREATE TABLE IF NOT EXISTS analysis_history 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          user_id INTEGER,
                          title TEXT,
                          url TEXT,
                          subreddit TEXT,
                          sentiment_score REAL,
                          total_comments INTEGER,
                          analysis_date TEXT,
                          analysis_data TEXT)''')
            
            # Trending cache table
            c.execute('''CREATE TABLE IF NOT EXISTS trending_cache 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          subreddit TEXT,
                          data TEXT,
                          cached_at TEXT)''')
            
            self.conn.commit()

    # User management
    def register(self, username, password):
        with db_lock:
            c = self.conn.cursor()
            hashed = hashlib.sha256(password.encode()).hexdigest()
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                c.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)", 
                         (username, hashed, created_at))
                self.conn.commit()
                return True
            except: 
                return False

    def login(self, username, password):
        with db_lock:
            c = self.conn.cursor()
            hashed = hashlib.sha256(password.encode()).hexdigest()
            c.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, hashed))
            return c.fetchone()

    # Subreddit group management
    def add_subreddit_group(self, user_id, subreddit_name):
        """Add a subreddit to user's tracking list"""
        with db_lock:
            c = self.conn.cursor()
            
            # Clean subreddit name
            clean_name = subreddit_name.lower().strip()
            if clean_name.startswith('r/'):
                clean_name = clean_name[2:]
            if clean_name.startswith('/'):
                clean_name = clean_name[1:]
            clean_name = clean_name.rstrip('/').replace(' ', '')
            
            if not clean_name:
                return False
            
            # Check if already exists
            c.execute('''SELECT id FROM subreddit_groups 
                         WHERE user_id=? AND subreddit_name=?''', 
                      (user_id, clean_name))
            if c.fetchone():
                return False  # Already exists
            
            # Add new subreddit
            added_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                c.execute('''INSERT INTO subreddit_groups (user_id, subreddit_name, added_date) 
                             VALUES (?, ?, ?)''', 
                          (user_id, clean_name, added_date))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Error adding subreddit: {e}")
                return False

    def get_user_subreddit_groups(self, user_id):
        """Get all subreddit groups tracked by user"""
        with db_lock:
            c = self.conn.cursor()
            c.execute('''SELECT id, subreddit_name, added_date, last_updated 
                         FROM subreddit_groups 
                         WHERE user_id=? 
                         ORDER BY added_date DESC''', (user_id,))
            
            results = []
            for row in c.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'added_date': row[2],
                    'last_updated': row[3]
                })
            return results

    def delete_subreddit_group(self, user_id, subreddit_id):
        """Delete a subreddit from user's tracking list"""
        with db_lock:
            c = self.conn.cursor()
            try:
                # Verify ownership
                c.execute('''SELECT id FROM subreddit_groups 
                             WHERE id=? AND user_id=?''', 
                         (subreddit_id, user_id))
                if not c.fetchone():
                    return False
                
                # Delete
                c.execute('''DELETE FROM subreddit_groups WHERE id=?''', (subreddit_id,))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting subreddit: {e}")
                return False

    def update_subreddit_timestamp(self, subreddit_name):
        """Update last updated timestamp for a subreddit"""
        with db_lock:
            c = self.conn.cursor()
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('''UPDATE subreddit_groups 
                         SET last_updated=? 
                         WHERE subreddit_name=?''', 
                     (updated_at, subreddit_name))
            self.conn.commit()

    # Analysis history
    def save_analysis(self, user_id, title, url, subreddit, analysis_data):
        """Save Reddit analysis results"""
        with db_lock:
            c = self.conn.cursor()
            analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract metrics
            sentiment_score = analysis_data.get('overall_score', 0)
            total_comments = analysis_data.get('total_comments', 0)
            
            # Clean analysis data for JSON serialization
            clean_data = self._clean_for_json(analysis_data)
            
            try:
                data_json = json.dumps(clean_data)
                
                c.execute('''INSERT INTO analysis_history 
                            (user_id, title, url, subreddit, sentiment_score, 
                             total_comments, analysis_date, analysis_data) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (user_id, title, url, subreddit, sentiment_score,
                          total_comments, analysis_date, data_json))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Error saving analysis: {e}")
                return False

    def _clean_for_json(self, data):
        """Clean data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return data

    def get_user_history(self, user_id, limit=10):
        """Get user's analysis history"""
        with db_lock:
            c = self.conn.cursor()
            c.execute('''SELECT id, title, url, subreddit, sentiment_score, analysis_date 
                         FROM analysis_history 
                         WHERE user_id=? 
                         ORDER BY analysis_date DESC 
                         LIMIT ?''', (user_id, limit))
            
            results = []
            for row in c.fetchall():
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'url': row[2],
                    'subreddit': row[3],
                    'sentiment_score': row[4],
                    'analysis_date': row[5]
                })
            return results

    # Trending cache
    def cache_trending_data(self, subreddit, data):
        """Cache trending data for a subreddit"""
        with db_lock:
            c = self.conn.cursor()
            cached_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_json = json.dumps(data)
            
            # Delete old cache
            c.execute('''DELETE FROM trending_cache WHERE subreddit=?''', (subreddit,))
            
            # Insert new cache
            c.execute('''INSERT INTO trending_cache (subreddit, data, cached_at) 
                         VALUES (?, ?, ?)''', 
                     (subreddit, data_json, cached_at))
            self.conn.commit()

    def get_cached_trending(self, subreddit, max_age_minutes=30):
        """Get cached trending data if not too old"""
        with db_lock:
            c = self.conn.cursor()
            c.execute('''SELECT data, cached_at FROM trending_cache 
                         WHERE subreddit=?''', (subreddit,))
            result = c.fetchone()
            
            if result:
                data_json, cached_at = result
                cache_time = datetime.strptime(cached_at, "%Y-%m-%d %H:%M:%S")
                current_time = datetime.now()
                
                if (current_time - cache_time).total_seconds() < max_age_minutes * 60:
                    return json.loads(data_json)
            
            return None

    # User stats
    def get_user_stats(self, user_id):
        """Get user statistics for dashboard"""
        with db_lock:
            c = self.conn.cursor()
            
            # Total analyses
            c.execute('''SELECT COUNT(*) FROM analysis_history WHERE user_id=?''', (user_id,))
            total_analyses = c.fetchone()[0] or 0
            
            # Average sentiment
            c.execute('''SELECT AVG(sentiment_score) FROM analysis_history WHERE user_id=?''', (user_id,))
            avg_sentiment = c.fetchone()[0] or 0
            
            # Subreddit count
            c.execute('''SELECT COUNT(*) FROM subreddit_groups WHERE user_id=?''', (user_id,))
            subreddit_count = c.fetchone()[0] or 0
            
            return {
                'total_analyses': total_analyses,
                'avg_sentiment': avg_sentiment,
                'subreddit_count': subreddit_count
            }


# ==========================================
# 2. CACHE & PROXY MANAGERS
# ==========================================

class CacheManager:
    """Wrapper for CacheService with application-specific logic"""
    
    def __init__(self):
        if CACHE_SERVICE_AVAILABLE:
            self.cache_service = CacheService(
                cache_dir="data/cache",
                default_ttl=1800,  # 30 minutes
                max_memory_items=200
            )
            # Schedule periodic cleanup
            self.last_cleanup = time.time()
        else:
            self.cache_service = None
            self.memory_cache = {}
            self.cache_dir = Path("data/cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_reddit_post(self, url):
        """Get Reddit post from cache or fetch"""
        if not self.cache_service:
            return None
        
        cache_key = f"reddit_post:{hashlib.md5(url.encode()).hexdigest()}"
        return self.cache_service.get(cache_key)
    
    def set_reddit_post(self, url, data, ttl=1800):
        """Cache Reddit post data"""
        if not self.cache_service:
            return False
        
        cache_key = f"reddit_post:{hashlib.md5(url.encode()).hexdigest()}"
        return self.cache_service.set(cache_key, data, ttl)
    
    def get_trending(self, subreddit, timeframe='day'):
        """Get trending posts from cache"""
        if not self.cache_service:
            return None
        
        cache_key = f"trending:{subreddit}:{timeframe}"
        return self.cache_service.get(cache_key)
    
    def set_trending(self, subreddit, data, timeframe='day', ttl=900):
        """Cache trending posts"""
        if not self.cache_service:
            return False
        
        cache_key = f"trending:{subreddit}:{timeframe}"
        return self.cache_service.set(cache_key, data, ttl)
    
    def delete_trending_cache(self, subreddit, timeframe='day'):
        """Delete trending cache for a subreddit"""
        if not self.cache_service:
            return False
        
        cache_key = f"trending:{subreddit}:{timeframe}"
        return self.cache_service.delete(cache_key)
    
    def cleanup_if_needed(self):
        """Periodic cache cleanup"""
        if self.cache_service and time.time() - self.last_cleanup > 3600:  # Every hour
            cleaned = self.cache_service.cleanup_expired()
            self.last_cleanup = time.time()
            if cleaned > 0:
                print(f"[Cache] Cleaned {cleaned} expired entries")


class ProxyManager:
    """Wrapper for ProxyService with retry logic"""
    
    def __init__(self):
        self.proxy_service = None
        self.current_proxy = None
        self.proxy_failures = 0
        self.max_proxy_failures = 3
        
        if PROXY_SERVICE_AVAILABLE:
            try:
                # FIX: Check if proxy file exists
                proxy_file = 'proxies/proxy_list.json'
                if os.path.exists(proxy_file):
                    self.proxy_service = ProxyService(
                        proxy_file=proxy_file,
                        cooldown_seconds=30
                    )
                    # Validate proxies on startup
                    threading.Thread(target=self._validate_proxies_background, daemon=True).start()
                else:
                    print(f"⚠️ Proxy file not found: {proxy_file}. Running without proxy.")
            except Exception as e:
                print(f"⚠️ Failed to initialize ProxyService: {e}")
        else:
            print("ℹ️ Running without ProxyService")
    
    def _validate_proxies_background(self):
        """Validate proxies in background thread"""
        if self.proxy_service:
            try:
                working_count = self.proxy_service.validate_all(max_workers=5)
                print(f"✅ Proxy validation completed. {working_count} proxies available")
            except Exception as e:
                print(f"⚠️ Proxy validation failed: {e}")
    
    def get_proxy(self):
        """Get a working proxy with rotation"""
        if not self.proxy_service:
            print("ℹ️ No proxy service available")
            return None
        
        # FIX: Handle case when no working proxies
        try:
            # If current proxy has too many failures, get new one
            if self.proxy_failures >= self.max_proxy_failures:
                self.current_proxy = None
                self.proxy_failures = 0
            
            if not self.current_proxy:
                self.current_proxy = self.proxy_service.get_working_proxy()
                self.proxy_failures = 0
            
            return self.current_proxy
        except Exception as e:
            print(f"⚠️ Error getting proxy: {e}")
            return None
    
    def mark_proxy_failure(self):
        """Mark current proxy as failed"""
        self.proxy_failures += 1
        self.current_proxy = None
        
        # Trigger proxy revalidation
        if self.proxy_service:
            threading.Thread(target=self._validate_proxies_background, daemon=True).start()
    
    def get_proxy_dict(self):
        """Get proxy dict for requests"""
        proxy = self.get_proxy()
        if not proxy:
            return None
        
        try:
            proxy_str = self.proxy_service.get_proxy_string(proxy)
            return {
                'http': proxy_str,
                'https': proxy_str
            }
        except Exception as e:
            print(f"⚠️ Error creating proxy dict: {e}")
            return None


# ==========================================
# 3. ENHANCED REDDIT API CLIENT WITH CACHE & PROXY
# ==========================================
class EnhancedRedditClient:
    """Enhanced client with cache, proxy, and rate limiting"""
    
    def __init__(self, cache_manager=None, proxy_manager=None):
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Reddit Sentiment Analyzer v2.0)'
        }
        
        # FIX: Updated mirror list with working mirrors
        self.mirrors = [
            # --- NHÓM 1: ĐỘ TIN CẬY CAO ---
            "https://safereddit.com",
            "https://l.opnxng.com",
            "https://redlib.vling.moe",
            "https://redlib.perennialte.ch",
            "https://redlib.kitty.is",
            
            # --- NHÓM 2: CÁC INSTANCE MỚI NỔI ---
            "https://rl.pcom.net",
            "https://redlib.ducks.party",
            "https://redlib.privacydev.net",
            "https://redlib.projectsegfau.lt",
            "https://redlib.nohost.network",
            
            # --- NHÓM 3: INSTANCE DỰ PHÒNG (DÀNH CHO QUỐC TẾ) ---
            "https://redlib.tux.pro",
            "https://redlib.matthew.sh",
            "https://redlib.freedit.eu",
            "https://redlib.backend.net",
            "https://redlib.crep.dev",
            
            # --- NHÓM 4: CÁC INSTANCE ÍT NGƯỜI BIẾT (TRÁNH RATE LIMIT) ---
            "https://redlib.seitan-ayumi.cf",
            "https://redlib.zapashny.cloud",
            "https://redlib.pisscloud.net",
            "https://redlib.slipfox.xyz",
            "https://redlib.no-logs.com",
            
            # --- NHÓM 5: DỰ PHÒNG CUỐI CÙNG ---
            "https://reddit.invidiou.sh",
            "https://libreddit.spike.codes",
            "https://www.reddit.com"
        ]
        
        self.cache_manager = cache_manager
        self.proxy_manager = proxy_manager
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        self.request_count = 0
        self.reset_time = time.time()
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset counter every hour
        if current_time - self.reset_time > 3600:
            self.request_count = 0
            self.reset_time = current_time
        
        # Limit to 60 requests per hour
        if self.request_count >= 60:
            wait_time = 3600 - (current_time - self.reset_time)
            if wait_time > 0:
                time.sleep(wait_time)
                self.request_count = 0
                self.reset_time = time.time()
        
        # Respect minimum interval
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def fetch_reddit_post(self, url):
        """Fetch a Reddit post with cache and proxy support"""
        # Check cache first
        if self.cache_manager:
            cached_data = self.cache_manager.get_reddit_post(url)
            if cached_data:
                print(f"[Cache] Hit for URL: {url[:50]}...")
                return cached_data, None
        
        try:
            self._rate_limit()
            
            # Clean and prepare URL
            if not url.startswith('http'):
                url = 'https://' + url
            
            if 'redd.it' in url:
                return None, "Please use full reddit.com URL"
            
            # Add .json suffix if needed
            if not url.endswith('.json'):
                url = url.rstrip('/') + '.json'
            
            # Try with proxy first, then without
            proxies = self._get_proxies()
            attempts = [proxies, None] if proxies else [None]
            
            for proxy_attempt in attempts:
                for mirror in self.mirrors:
                    try:
                        if 'reddit.com' in url:
                            path = url.split('reddit.com')[1]
                            target_url = mirror + path
                        else:
                            target_url = url
                        
                        print(f"[Request] Trying {mirror} with proxy: {proxy_attempt is not None}")
                        response = requests.get(
                            target_url, 
                            headers=self.headers, 
                            proxies=proxy_attempt,
                            timeout=10,
                            verify=True
                        )
                        
                        if response.status_code == 200:
                            parsed_data = self._parse_reddit_json(response.json())
                            
                            # Cache the result
                            if self.cache_manager and parsed_data:
                                self.cache_manager.set_reddit_post(url, parsed_data)
                            
                            return parsed_data, None
                        elif response.status_code == 429:
                            print(f"[Rate Limit] Hit for {mirror}")
                            time.sleep(2)  # Wait before trying next
                            
                    except requests.exceptions.ProxyError:
                        print(f"[Proxy] Failed for {mirror}")
                        if self.proxy_manager:
                            self.proxy_manager.mark_proxy_failure()
                        continue
                    except requests.exceptions.Timeout:
                        print(f"[Timeout] {mirror}")
                        continue
                    except requests.exceptions.ConnectionError:
                        print(f"[Connection] Failed for {mirror}")
                        continue
                    except Exception as e:
                        print(f"[Error] {mirror}: {e}")
                        continue
            
            return None, "Cannot fetch data from Reddit (all mirrors failed)"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def fetch_subreddit_trending(self, subreddit_name, limit=10, timeframe='day'):
        """Fetch trending posts with safety checks for JSON and Mirror rotation"""
        # Lấy danh sách proxy nếu có, nếu không trả về None để dùng kết nối trực tiếp
        proxies = self._get_proxies()
        
        for mirror in self.mirrors:
            try:
                # Chuẩn hóa URL để tránh lỗi dấu gạch chéo
                base_url = mirror.rstrip('/')
                url = f"{base_url}/r/{subreddit_name}/top.json?limit={limit}&t={timeframe}"
                
                print(f"[Trending] Đang thử nguồn: {mirror}")
                response = requests.get(url, headers=self.headers, proxies=proxies, timeout=5)
                
                # Kiểm tra phản hồi có phải là JSON hợp lệ không
                content_type = response.headers.get('Content-Type', '')
                if response.status_code == 200 and 'application/json' in content_type:
                    try:
                        data = response.json()
                        # Kiểm tra xem JSON có rỗng hoặc chứa thông báo lỗi không
                        if not data or 'data' not in data:
                            print(f"⚠️ Nguồn {mirror} trả về JSON rỗng hoặc không đúng cấu trúc")
                            continue
                    except Exception:
                        print(f"⚠️ Nguồn {mirror} trả về HTML giả dạng JSON")
                        continue
                    
                    posts = []
                    
                    # Kiểm tra cấu trúc JSON Reddit tiêu chuẩn
                    if 'data' in data and 'children' in data['data']:
                        for child in data['data']['children']:
                            p_data = child['data']
                            posts.append({
                                'id': p_data.get('id'),
                                'title': p_data.get('title', 'No Title'),
                                'author': p_data.get('author', 'unknown'),
                                'score': p_data.get('score', 0),
                                'comments_count': p_data.get('num_comments', 0),
                                'url': f"https://www.reddit.com{p_data.get('permalink', '')}",
                                'subreddit': subreddit_name,
                                'time_str': self._format_time(p_data.get('created_utc', time.time())),
                                'selftext': p_data.get('selftext', '')[:200]
                            })
                        
                        # Cache kết quả thành công
                        if self.cache_manager:
                            self.cache_manager.set_trending(subreddit_name, posts, timeframe)
                            
                        return posts, None
                else:
                    print(f"⚠️ Nguồn {mirror} trả về lỗi hoặc chặn truy cập (Status: {response.status_code})")
                    continue
                    
            except Exception as e:
                print(f"❌ Lỗi kết nối tới {mirror}: {str(e)}")
                continue
                
        return [], "Tất cả các nguồn dữ liệu (Mirrors) đều thất bại hoặc bị chặn"
    
    def _get_proxies(self):
        """Get proxy configuration"""
        if self.proxy_manager:
            return self.proxy_manager.get_proxy_dict()
        return None
    
    def _parse_reddit_json(self, data):
        """Parse Reddit JSON response"""
        try:
            if isinstance(data, list) and len(data) >= 2:
                # Post data
                post_data = data[0]['data']['children'][0]['data']
                
                # Comments data
                comments = []
                comments_data = data[1]['data']['children']
                
                for comment in comments_data:
                    if 'data' in comment and 'body' in comment['data']:
                        comment_data = comment['data']
                        comments.append({
                            'body': comment_data.get('body', ''),
                            'author': comment_data.get('author', 'Unknown'),
                            'score': comment_data.get('score', 0),
                            'created_utc': comment_data.get('created_utc', time.time()),
                            'id': comment_data.get('id', '')
                        })
                
                # Meta data
                meta = {
                    'title': post_data.get('title', 'No Title'),
                    'subreddit': post_data.get('subreddit', 'unknown'),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'author': post_data.get('author', '[deleted]'),
                    'created_utc': post_data.get('created_utc', time.time()),
                    'created_time': self._format_time(post_data.get('created_utc', time.time())),
                    'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                    'selftext': post_data.get('selftext', ''),
                    'upvote_ratio': post_data.get('upvote_ratio', 0),
                    'id': post_data.get('id', '')
                }
                
                return {'meta': meta, 'comments': comments}
            
        except Exception as e:
            print(f"Parse error: {e}")
        
        return None
    
    def _format_time(self, timestamp):
        """Format timestamp to relative time"""
        try:
            post_time = datetime.fromtimestamp(timestamp)
            now = datetime.now()
            diff = now - post_time
            
            if diff.days > 365:
                years = diff.days // 365
                return f"{years} năm trước"
            elif diff.days > 30:
                months = diff.days // 30
                return f"{months} tháng trước"
            elif diff.days > 0:
                return f"{diff.days} ngày trước"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours} giờ trước"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes} phút trước"
            else:
                return "Vừa xong"
        except:
            return "Không rõ"


# ==========================================
# 4. SENTIMENT ANALYZER WITH ANALYSIS SERVICE
# ==========================================
class SentimentAnalyzer:
    """Analyze sentiment from Reddit data using AnalysisService"""
    
    def __init__(self):
        self.analysis_service = AnalysisService() if ANALYSIS_SERVICE_AVAILABLE else None
        self.sentiment_labels = {
            "Rất tích cực": 1.0,
            "Tích cực": 0.6,
            "Trung lập": 0.0,
            "Tiêu cực": -0.6,
            "Rất tiêu cực": -1.0
        }
        
    
    def analyze_reddit_post(self, reddit_data):
        """Analyze a Reddit post and its comments"""
        if not reddit_data:
            return self._create_empty_result()
        
        meta = reddit_data.get('meta', {})
        comments = reddit_data.get('comments', [])
        
        # Clean meta data for JSON
        clean_meta = self._clean_meta_data(meta)
        
        # Analyze comments (limit for performance)
        max_comments = min(len(comments), 150)  # Increased limit for better analysis
        analyzed_comments = []
        sentiment_counts = {label: 0 for label in self.sentiment_labels.keys()}
        total_sentiment_score = 0
        all_keywords = []
        
        # Process comments in batches for better performance
        for i in range(0, max_comments, 20):
            batch = comments[i:i+20]
            for comment in batch:
                analysis = self._analyze_comment(comment)
                analyzed_comments.append(analysis)
                
                sentiment = analysis['sentiment']
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
                
                total_sentiment_score += analysis['score']
                
                # Collect keywords
                if 'keywords' in analysis:
                    all_keywords.extend(analysis['keywords'])
        
        # Calculate statistics
        total_comments = len(analyzed_comments)
        overall_score = total_sentiment_score / total_comments if total_comments > 0 else 0
        
        # Get top keywords
        keyword_counter = Counter(all_keywords)
        top_keywords = [{'word': word, 'count': count} 
                       for word, count in keyword_counter.most_common(10)]
        
        # Generate summary
        summary = self._generate_summary(clean_meta, sentiment_counts, overall_score, total_comments, top_keywords)
        
        # Prepare result
        result = {
            'meta': clean_meta,
            'comments': analyzed_comments,
            'statistics': {
                'total_comments': total_comments,
                'overall_score': overall_score,
                'sentiment_distribution': sentiment_counts,
                'positive_count': sentiment_counts["Tích cực"] + sentiment_counts["Rất tích cực"],
                'negative_count': sentiment_counts["Tiêu cực"] + sentiment_counts["Rất tiêu cực"],
                'neutral_count': sentiment_counts["Trung lập"],
                'top_keywords': top_keywords
            },
            'summary': summary,
            'analyzed_at': datetime.now().isoformat()
        }
        
        return result
    
    def _analyze_comment(self, comment):
        """Analyze sentiment of a single comment"""
        text = comment.get('body', '')
        
        # Use AnalysisService if available
        if self.analysis_service:
            try:
                analysis = self.analysis_service.analyze_text(text)
                
                # Map to our sentiment labels
                sentiment_map = {
                    'positive': 'Tích cực',
                    'negative': 'Tiêu cực',
                    'neutral': 'Trung lập',
                    'very_positive': 'Rất tích cực',
                    'very_negative': 'Rất tiêu cực'
                }
                
                sentiment = sentiment_map.get(analysis.get('sentiment', 'neutral'), 'Trung lập')
                score = analysis.get('score', 0)
                
                # Adjust score for very positive/negative
                if sentiment == 'Rất tích cực':
                    score = max(0.8, score)
                elif sentiment == 'Rất tiêu cực':
                    score = min(-0.8, score)
                
                return {
                    'text': text[:200] + ('...' if len(text) > 200 else ''),
                    'sentiment': sentiment,
                    'score': score,
                    'author': comment.get('author', 'Ẩn danh'),
                    'timestamp': datetime.fromtimestamp(comment.get('created_utc', time.time())).isoformat(),
                    'keywords': analysis.get('keywords', [])[:5]
                }
            except Exception as e:
                print(f"AnalysisService error: {e}")
        
        # Fallback to simple analysis
        return self._simple_analyze_comment(comment)
    
    def _simple_analyze_comment(self, comment):
        """Simple fallback sentiment analysis"""
        text = comment.get('body', '')
        
        # Simple keyword-based analysis
        positive_words = ['tốt', 'tuyệt', 'xuất sắc', 'hoàn hảo', 'thích', 'hài lòng', 'tuyệt vời', 'awesome', 'great', 'good', 'love', 'excellent']
        negative_words = ['xấu', 'tệ', 'dở', 'không thích', 'thất vọng', 'kém', 'tồi', 'bad', 'terrible', 'awful', 'hate', 'worst']
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if pos_score > neg_score:
            if pos_score - neg_score > 2:
                sentiment = "Rất tích cực"
                score = 0.8
            else:
                sentiment = "Tích cực"
                score = 0.4
        elif neg_score > pos_score:
            if neg_score - pos_score > 2:
                sentiment = "Rất tiêu cực"
                score = -0.8
            else:
                sentiment = "Tiêu cực"
                score = -0.4
        else:
            sentiment = "Trung lập"
            score = 0
        
        # Extract keywords
        keywords = list(set(re.findall(r'\b\w{4,}\b', text_lower)))[:6]
        
        return {
            'text': text[:200] + ('...' if len(text) > 200 else ''),
            'sentiment': sentiment,
            'score': score,
            'author': comment.get('author', 'Ẩn danh'),
            'timestamp': datetime.fromtimestamp(comment.get('created_utc', time.time())).isoformat(),
            'keywords': keywords
        }
    
    def _clean_meta_data(self, meta):
        """Clean meta data for JSON serialization"""
        clean_meta = {}
        for key, value in meta.items():
            if isinstance(value, datetime):
                clean_meta[key] = value.isoformat()
            elif key == 'created_utc':
                clean_meta[key] = value
            else:
                clean_meta[key] = value
        return clean_meta
    
    def _generate_summary(self, meta, sentiment_counts, overall_score, total_comments, top_keywords):
        """Generate analysis summary"""
        if total_comments == 0:
            return "Không có bình luận để phân tích."
        
        title = meta.get('title', 'Bài viết không có tiêu đề')
        subreddit = meta.get('subreddit', 'unknown')
        post_score = meta.get('score', 0)
        
        # Calculate percentages
        positive_percentage = (sentiment_counts["Tích cực"] + sentiment_counts["Rất tích cực"]) / total_comments * 100
        negative_percentage = (sentiment_counts["Tiêu cực"] + sentiment_counts["Rất tiêu cực"]) / total_comments * 100
        
        # Determine overall mood
        if overall_score > 0.3:
            mood = "rất tích cực"
            recommendation = "Cộng đồng đang rất ủng hộ bài viết này."
        elif overall_score > 0:
            mood = "tích cực"
            recommendation = "Phản hồi chủ yếu là tích cực."
        elif overall_score > -0.3:
            mood = "hơi tiêu cực"
            recommendation = "Có một số phản hồi tiêu cực cần lưu ý."
        else:
            mood = "rất tiêu cực"
            recommendation = "Bài viết nhận nhiều phản hồi tiêu cực."
        
        # Keywords section
        keywords_text = ""
        if top_keywords:
            top_5 = [k['word'] for k in top_keywords[:5]]
            keywords_text = f"\n**Từ khóa nổi bật:** {', '.join(top_5)}"
        
        summary = f"""
**Phân tích bài viết Reddit:**
- **Tiêu đề:** {title}
- **Subreddit:** r/{subreddit}
- **Điểm bài viết:** {post_score} ⭐
- **Tổng bình luận:** {total_comments} 💬

**Phân tích cảm xúc:**
- **Điểm cảm xúc trung bình:** {overall_score:.2f}
- **Tỷ lệ tích cực:** {positive_percentage:.1f}%
- **Tỷ lệ tiêu cực:** {negative_percentage:.1f}%
- **Tâm trạng chung:** {mood}
{keywords_text}

**Khuyến nghị:** {recommendation}

**Thống kê chi tiết:**
- Rất tích cực: {sentiment_counts['Rất tích cực']} bình luận
- Tích cực: {sentiment_counts['Tích cực']} bình luận
- Trung lập: {sentiment_counts['Trung lập']} bình luận
- Tiêu cực: {sentiment_counts['Tiêu cực']} bình luận
- Rất tiêu cực: {sentiment_counts['Rất tiêu cực']} bình luận
"""
        return summary
    
    def _create_empty_result(self):
        """Create empty result structure"""
        return {
            'meta': {},
            'comments': [],
            'statistics': {
                'total_comments': 0,
                'overall_score': 0,
                'sentiment_distribution': {label: 0 for label in self.sentiment_labels.keys()},
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'top_keywords': []
            },
            'summary': "Không có dữ liệu để phân tích.",
            'analyzed_at': datetime.now().isoformat()
        }


# ==========================================
# 5. TRENDING MANAGER WITH CACHE
# ==========================================
class TrendingManager:
    """Manage trending posts from tracked subreddits with cache"""
    
    def __init__(self, db_manager, cache_manager=None, proxy_manager=None):
        self.db = db_manager
        self.cache_manager = cache_manager
        self.proxy_manager = proxy_manager
        self.reddit_client = EnhancedRedditClient(cache_manager, proxy_manager)
        self.cache_duration = 30  # minutes
    
    def get_trending_for_subreddit(self, subreddit_name, force_refresh=False):
        """Get trending posts for a subreddit"""
        # Check cache first
        if not force_refresh:
            if self.cache_manager:
                cached_data = self.cache_manager.get_trending(subreddit_name)
                if cached_data:
                    print(f"[TrendingCache] Hit for {subreddit_name}")
                    return cached_data
            else:
                # Fallback to database cache
                cached_data = self.db.get_cached_trending(subreddit_name, self.cache_duration)
                if cached_data:
                    return cached_data
        
        # Fetch fresh data
        posts, error = self.reddit_client.fetch_subreddit_trending(subreddit_name, limit=10)
        
        if error:
            print(f"Error fetching trending for {subreddit_name}: {error}")
            return []
        
        # Update caches
        if self.cache_manager:
            self.cache_manager.set_trending(subreddit_name, posts)
        else:
            self.db.cache_trending_data(subreddit_name, posts)
        
        return posts
    
    def get_trending_for_user(self, user_id):
        """Get trending posts from all user's tracked subreddits"""
        subreddits = self.db.get_user_subreddit_groups(user_id)
        
        if not subreddits:
            return []
        
        all_posts = []
        
        # Use threading for parallel fetching
        def fetch_subreddit(subreddit):
            try:
                return self.get_trending_for_subreddit(subreddit['name'])
            except Exception as e:
                print(f"Error fetching {subreddit['name']}: {e}")
                return []
        
        # Fetch sequentially for now (to avoid rate limiting)
        for subreddit in subreddits:
            posts = fetch_subreddit(subreddit)
            for post in posts:
                post['group_name'] = subreddit['name']
            all_posts.extend(posts)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Sort by score (most popular first)
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        
        return all_posts[:20]  # Return top 20 posts


# ==========================================
# 6. MAIN APPLICATION
# ==========================================
class RedditSentimentApp:
    """Main application class"""
    
    def __init__(self):
        self.db = DBManager()
        self.cache_manager = CacheManager()
        self.proxy_manager = ProxyManager()
        self.reddit_client = EnhancedRedditClient(self.cache_manager, self.proxy_manager)
        self.analyzer = SentimentAnalyzer()
        self.trending_manager = TrendingManager(self.db, self.cache_manager, self.proxy_manager)
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'page' not in st.session_state:
            st.session_state.page = "Dashboard"
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'trending_posts' not in st.session_state:
            st.session_state.trending_posts = []
        if 'last_cache_cleanup' not in st.session_state:
            st.session_state.last_cache_cleanup = time.time()
    
    # ============ PAGE RENDERERS ============
    
    def render_login_page(self):
        """Render login page"""
        if UI_AVAILABLE:
            load_css()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 40px;">
                <h1 style="font-size: 2.5rem; margin-bottom: 10px;">💎 Reddit Sentiment Analyzer</h1>
                <p style="color: var(--text-sub);">Phân tích cảm xúc cộng đồng Reddit với AI</p>
                <p style="color: var(--text-sub); font-size: 0.9rem; margin-top: 10px;">
                    🚀 Enhanced with Cache & Proxy for Streamlit Cloud
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # System status
            with st.expander("📊 System Status"):
                col_status1, col_status2, col_status3 = st.columns(3)
                with col_status1:
                    st.metric("Cache", "✅" if CACHE_SERVICE_AVAILABLE else "❌")
                with col_status2:
                    st.metric("Proxy", "✅" if PROXY_SERVICE_AVAILABLE else "❌")
                with col_status3:
                    st.metric("AI Analysis", "✅" if ANALYSIS_SERVICE_AVAILABLE else "⚠️")
            
            # Login form
            with st.form("login_form"):
                st.markdown("### 🔐 Đăng nhập")
                
                username = st.text_input("Tên đăng nhập", placeholder="Nhập username")
                password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    login_btn = st.form_submit_button("🚀 Đăng nhập", use_container_width=True)
                with col_btn2:
                    register_btn = st.form_submit_button("📝 Đăng ký", use_container_width=True)
                
                if login_btn:
                    if username and password:
                        user = self.db.login(username, password)
                        if user:
                            st.session_state.user = {"id": user[0], "username": user[1]}
                            st.session_state.authenticated = True
                            if UI_AVAILABLE:
                                show_success_message(f"Chào mừng {user[1]}!")
                            st.rerun()
                        else:
                            if UI_AVAILABLE:
                                show_error_message("Sai tên đăng nhập hoặc mật khẩu")
                            else:
                                st.error("Sai thông tin đăng nhập")
                    else:
                        if UI_AVAILABLE:
                            show_error_message("Vui lòng nhập đầy đủ thông tin")
                
                if register_btn:
                    if username and password:
                        if len(username) < 3:
                            if UI_AVAILABLE:
                                show_error_message("Tên đăng nhập phải có ít nhất 3 ký tự")
                            else:
                                st.error("Tên đăng nhập phải có ít nhất 3 ký tự")
                        elif len(password) < 6:
                            if UI_AVAILABLE:
                                show_error_message("Mật khẩu phải có ít nhất 6 ký tự")
                            else:
                                st.error("Mật khẩu phải có ít nhất 6 ký tự")
                        elif self.db.register(username, password):
                            if UI_AVAILABLE:
                                show_success_message("Đăng ký thành công! Vui lòng đăng nhập.")
                            else:
                                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                        else:
                            if UI_AVAILABLE:
                                show_error_message("Tên đăng nhập đã tồn tại")
                            else:
                                st.error("Tên đăng nhập đã tồn tại")
                    else:
                        if UI_AVAILABLE:
                            show_error_message("Vui lòng nhập đầy đủ thông tin")
                        else:
                            st.error("Vui lòng nhập đầy đủ thông tin")
            
            # Demo mode
            st.markdown("---")
            if st.button("👀 Dùng thử Demo", use_container_width=True):
                st.session_state.user = {"id": 0, "username": "Demo User"}
                st.session_state.authenticated = True
                st.rerun()
    
    def render_dashboard_page(self):
        """Render main dashboard page"""
        user = st.session_state.user
        
        # Periodic cache cleanup
        if time.time() - st.session_state.last_cache_cleanup > 3600:
            if self.cache_manager:
                self.cache_manager.cleanup_if_needed()
            st.session_state.last_cache_cleanup = time.time()
        
        # Get user stats and subreddits
        user_stats = self.db.get_user_stats(user['id']) if user['id'] > 0 else {
            'total_analyses': 0,
            'avg_sentiment': 0,
            'subreddit_count': 0
        }
        
        user_subreddits = self.db.get_user_subreddit_groups(user['id']) if user['id'] > 0 else []
        
        # Get trending posts
        if not st.session_state.trending_posts and user['id'] > 0:
            with st.spinner("Đang tải bài viết trending (có cache & proxy)..."):
                trending_posts = self.trending_manager.get_trending_for_user(user['id'])
                st.session_state.trending_posts = trending_posts
        
        # Create dashboard data with real data
        dashboard_data = {
            'user_stats': user_stats,
            'trending_posts': st.session_state.trending_posts if user['id'] > 0 else [],
            'user_subreddits': user_subreddits,
            'timeline_data': pd.DataFrame(),
            'sentiment_data': pd.DataFrame(),
            'comments_data': pd.DataFrame()
        }
        
        # Render sidebar with subreddit management
        if UI_AVAILABLE:
            render_enhanced_sidebar(
                username=user.get('username', 'Guest'),
                groups=user_subreddits,
                logout_callback=self.logout,
                add_group_callback=lambda sub: self.add_subreddit_group(sub),
                delete_group_callback=lambda sub_id: self.delete_subreddit_group(sub_id)
            )
        
        # Render dashboard with real data and username
        if UI_AVAILABLE:
            render_main_dashboard(dashboard_data, username=user['username'])
        
        # Show trending posts section (if any)
        if user['id'] > 0 and st.session_state.trending_posts:
            self.render_trending_section()
        
        # Show subreddit management
        self.render_subreddit_management()
    
    def render_analysis_page(self):
        """Render Reddit URL analysis page"""
        user = st.session_state.user
        user_subreddits = self.db.get_user_subreddit_groups(user['id']) if user['id'] > 0 else []
        
        if UI_AVAILABLE:
            render_enhanced_sidebar(
                username=user['username'],
                groups=user_subreddits,
                logout_callback=self.logout,
                add_group_callback=lambda sub: self.add_subreddit_group(sub),
                delete_group_callback=lambda sub_id: self.delete_subreddit_group(sub_id)
            )
        
        st.markdown("## 🔍 Phân tích bài viết Reddit")
        
        # FIX: Add clear label for accessibility (điểm nghẽn 3)
        url = st.text_input(
            "Dán URL bài viết tại đây",
            value=st.session_state.get('analyze_url', ''),
            placeholder="https://www.reddit.com/r/technology/comments/abc123/title_here/",
            help="Dán URL đầy đủ của bài viết Reddit (phải chứa 'reddit.com')"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_btn = st.button("🚀 Bắt đầu phân tích", use_container_width=True, type="primary")
        
        with col2:
            use_proxy = st.checkbox("Sử dụng Proxy", value=True, help="Sử dụng proxy để tránh bị chặn")
        
        with col3:
            if st.session_state.get('current_analysis'):
                if st.button("📥 Xuất báo cáo", use_container_width=True):
                    self.export_analysis()
        
        if analyze_btn:
            if url and 'reddit.com' in url:
                # Temporarily disable proxy if not wanted
                original_proxy_state = self.proxy_manager.proxy_service is not None
                if not use_proxy:
                    self.reddit_client.proxy_manager = None
                
                self.analyze_reddit_url(url)
                
                # Restore proxy state
                if not use_proxy:
                    self.reddit_client.proxy_manager = self.proxy_manager
            else:
                if UI_AVAILABLE:
                    show_error_message("Vui lòng nhập URL Reddit hợp lệ (phải chứa 'reddit.com')")
                else:
                    st.error("Vui lòng nhập URL Reddit hợp lệ (phải chứa 'reddit.com')")
        
        # Show current analysis results
        if st.session_state.get('current_analysis'):
            st.markdown("---")
            st.markdown("### 📊 Kết quả phân tích")
            
            if UI_AVAILABLE:
                # Prepare data for UI
                analysis_result = st.session_state.current_analysis
                df = self.prepare_data_for_ui(analysis_result['comments'])
                
                ui_data = {
                    'meta': analysis_result['meta'],
                    'df': df,
                    'summary': analysis_result['summary'],
                    'statistics': analysis_result['statistics']
                }
                
                render_enhanced_analysis_result(ui_data)
            else:
                # Basic display
                analysis = st.session_state.current_analysis
                meta = analysis['meta']
                stats = analysis['statistics']
                
                st.write(f"**Tiêu đề:** {meta.get('title', 'N/A')}")
                st.write(f"**Subreddit:** r/{meta.get('subreddit', 'N/A')}")
                st.write(f"**Tác giả:** {meta.get('author', 'N/A')}")
                st.write(f"**Điểm:** {meta.get('score', 0)} ⭐")
                st.write(f"**Bình luận:** {stats['total_comments']} 💬")
                st.write(f"**Điểm cảm xúc TB:** {stats['overall_score']:.2f}")
                
                st.markdown("**Tóm tắt:**")
                st.write(analysis['summary'])
    
    # ============ SUBREDDIT MANAGEMENT ============
    
    def render_subreddit_management(self):
        """Render subreddit management section"""
        st.markdown("---")
        st.markdown("### 🏷️ Quản lý Subreddit Groups")
        
        user = st.session_state.user
        user_subreddits = self.db.get_user_subreddit_groups(user['id']) if user['id'] > 0 else []
        
        # Add new subreddit
        col1, col2 = st.columns([3, 1])
        with col1:
            new_subreddit = st.text_input(
                "Thêm subreddit mới",
                placeholder="Nhập tên subreddit (ví dụ: technology, programming)",
                help="Nhập tên subreddit không có r/ phía trước",
                key="new_subreddit_input"
            )
        
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("➕ Thêm", key="add_subreddit_btn", use_container_width=True):
                if new_subreddit:
                    if self.add_subreddit_group(new_subreddit):
                        if UI_AVAILABLE:
                            show_success_message(f"Đã thêm r/{new_subreddit} vào danh sách theo dõi")
                        else:
                            st.success(f"Đã thêm r/{new_subreddit} vào danh sách theo dõi")
                        st.rerun()
                else:
                    if UI_AVAILABLE:
                        show_error_message("Vui lòng nhập tên subreddit")
                    else:
                        st.error("Vui lòng nhập tên subreddit")
        
        # Current subreddits
        if user_subreddits:
            st.markdown("#### 📋 Subreddit đang theo dõi")
            
            for subreddit in user_subreddits:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    last_updated = subreddit.get('last_updated', 'Chưa cập nhật')
                    if last_updated and last_updated != 'Chưa cập nhật' and isinstance(last_updated, str):
                        last_updated = last_updated[:10]
                    elif last_updated and last_updated != 'Chưa cập nhật':
                        last_updated = str(last_updated)[:10]
                    
                    if UI_AVAILABLE:
                        st.markdown(f"""
                        <div style="padding: 12px; background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border);">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="font-size: 1.2rem;">🏷️</div>
                                <div>
                                    <div style="font-weight: 600;">r/{subreddit['name']}</div>
                                    <div style="font-size: 0.8rem; color: var(--text-sub);">
                                        Thêm: {subreddit['added_date'][:10] if subreddit['added_date'] else 'N/A'}
                                        {f" • Cập nhật: {last_updated}" if last_updated != 'Chưa cập nhật' else ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"**r/{subreddit['name']}**")
                        st.caption(f"Thêm: {subreddit['added_date'][:10] if subreddit['added_date'] else 'N/A'}")
                
                with col2:
                    if st.button("🔄 Làm mới", key=f"refresh_{subreddit['id']}", use_container_width=True):
                        self.refresh_subreddit(subreddit['name'])
                
                with col3:
                    if st.button("🗑️", key=f"delete_{subreddit['id']}", use_container_width=True, type="secondary"):
                        if self.delete_subreddit_group(subreddit['id']):
                            if UI_AVAILABLE:
                                show_success_message(f"Đã xóa r/{subreddit['name']}")
                            else:
                                st.success(f"Đã xóa r/{subreddit['name']}")
                            st.rerun()
        
        elif user['id'] > 0:  # Not demo user
            st.markdown("""
            <div style="background: var(--bg-card); padding: 20px; border-radius: 8px; border: 1px solid var(--border); margin: 20px 0;">
                <div style="font-weight: 600; color: var(--text-main); margin-bottom: 10px;">📌 Chưa theo dõi subreddit nào</div>
                <div style="color: var(--text-sub);">
                    Thêm subreddit bạn quan tâm để theo dõi trending và phân tích cảm xúc cộng đồng.
                    Ví dụ: technology, programming, science, news, etc.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_trending_section(self):
        """Render trending posts section"""
        
        trending_posts = st.session_state.get("trending_posts", [])
        if not trending_posts:
            return

        st.divider()
        st.subheader("🔥 Bài viết trending từ subreddits của bạn")

        # Group posts by subreddit
        posts_by_subreddit = {}
        for post in trending_posts:
            subreddit = post.get("subreddit", "unknown")
            clean_subreddit = subreddit.replace('r/', '').replace('/', '')
            posts_by_subreddit.setdefault(clean_subreddit, []).append(post)

        # Create tabs
        if posts_by_subreddit:
            tabs = st.tabs([f"r/{sub}" for sub in posts_by_subreddit.keys()])

            for tab, (subreddit, posts) in zip(tabs, posts_by_subreddit.items()):
                with tab:
                    for i, post in enumerate(posts[:5]):
                        with st.container(border=True):
                            st.caption(f"r/{subreddit} · {post.get('time_str', '')}")
                            st.markdown(f"**{post.get('title', '')}**")

                            if post.get("selftext"):
                                st.write(post["selftext"][:200] + "...")

                            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                            with col1:
                                st.metric("▲ Điểm", post.get("score", 0))

                            with col2:
                                st.metric("💬 Bình luận", post.get("comments_count", 0))

                            with col3:
                                st.write("👤 **Tác giả**")
                                st.write(post.get("author", "unknown"))

                            with col4:
                                btn_key = f"analyze_{subreddit}_{post.get('id', '')}_{i}"
                                if st.button(
                                    "Phân tích",
                                    key=btn_key,
                                    use_container_width=True
                                ):
                                    st.session_state.analyze_url = post.get("url")
                                    st.session_state.page = "Analysis"
                                    st.rerun()
        else:
            st.info("Chưa có bài viết trending để hiển thị.")
    
    # ============ HELPER METHODS ============
    
    def add_subreddit_group(self, subreddit_name):
        """Add a subreddit to user's tracking list"""
        user = st.session_state.user
        if user['id'] == 0:  # Demo user
            if UI_AVAILABLE:
                show_error_message("Tính năng không khả dụng cho tài khoản demo")
            else:
                st.error("Tính năng không khả dụng cho tài khoản demo")
            return False
        
        success = self.db.add_subreddit_group(user['id'], subreddit_name)
        if success:
            # Clear trending cache
            st.session_state.trending_posts = []
            if self.cache_manager:
                # Clear cache for this subreddit
                self.cache_manager.delete_trending_cache(subreddit_name)
        return success
    
    def delete_subreddit_group(self, subreddit_id):
        """Delete a subreddit from user's tracking list"""
        user = st.session_state.user
        if user['id'] == 0:  # Demo user
            if UI_AVAILABLE:
                show_error_message("Tính năng không khả dụng cho tài khoản demo")
            else:
                st.error("Tính năng không khả dụng cho tài khoản demo")
            return False
        
        # Get subreddit name before deletion
        user_subreddits = self.db.get_user_subreddit_groups(user['id'])
        subreddit_to_delete = None
        for sub in user_subreddits:
            if sub['id'] == subreddit_id:
                subreddit_to_delete = sub['name']
                break
        
        success = self.db.delete_subreddit_group(user['id'], subreddit_id)
        if success:
            # Clear trending cache
            st.session_state.trending_posts = []
            if self.cache_manager and subreddit_to_delete:
                # Clear cache for this subreddit
                self.cache_manager.delete_trending_cache(subreddit_to_delete)
        return success
    
    def refresh_subreddit(self, subreddit_name):
        """Refresh trending data for a subreddit"""
        # Clear cache for this subreddit
        if self.cache_manager:
            self.cache_manager.delete_trending_cache(subreddit_name)
        else:
            self.db.cache_trending_data(subreddit_name, [])
        
        # Update timestamp
        self.db.update_subreddit_timestamp(subreddit_name)
        
        # Clear trending posts to trigger refresh
        st.session_state.trending_posts = []
        
        if UI_AVAILABLE:
            show_success_message(f"Đã làm mới dữ liệu cho r/{subreddit_name}")
        else:
            st.success(f"Đã làm mới dữ liệu cho r/{subreddit_name}")
        st.rerun()
    
    def analyze_reddit_url(self, url):
        """Analyze a Reddit URL with enhanced features"""
        if UI_AVAILABLE:
            show_loading_animation("Đang tải dữ liệu từ Reddit (có cache & proxy)...")
        else:
            with st.spinner("Đang tải dữ liệu từ Reddit..."):
                pass
        
        # Fetch data from Reddit with enhanced client
        reddit_data, error = self.reddit_client.fetch_reddit_post(url)
        
        if error:
            if UI_AVAILABLE:
                show_error_message(f"Lỗi: {error}")
            else:
                st.error(f"Lỗi: {error}")
            return
        
        if not reddit_data:
            if UI_AVAILABLE:
                show_error_message("Không thể phân tích bài viết này")
            else:
                st.error("Không thể phân tích bài viết này")
            return
        
        # Analyze sentiment
        analysis_result = self.analyzer.analyze_reddit_post(reddit_data)
        
        # Save to database if not demo user
        user = st.session_state.user
        if user['id'] > 0:
            meta = analysis_result['meta']
            self.db.save_analysis(
                user['id'],
                meta['title'],
                meta['url'],
                meta['subreddit'],
                analysis_result
            )
        
        # Store in session
        st.session_state.current_analysis = analysis_result
        
        if UI_AVAILABLE:
            show_success_message("Phân tích hoàn tất!")
        else:
            st.success("Phân tích hoàn tất!")
        time.sleep(1)
        st.rerun()
    
    def export_analysis(self):
        """Export analysis results"""
        if not st.session_state.get('current_analysis'):
            if UI_AVAILABLE:
                show_error_message("Không có dữ liệu để xuất")
            else:
                st.error("Không có dữ liệu để xuất")
            return
        
        analysis = st.session_state.current_analysis
        meta = analysis['meta']
        
        # Create simple report
        report = f"""
# Báo cáo phân tích Reddit
**Tiêu đề:** {meta.get('title', 'N/A')}
**Subreddit:** r/{meta.get('subreddit', 'N/A')}
**URL:** {meta.get('url', 'N/A')}
**Thời gian phân tích:** {analysis.get('analyzed_at', 'N/A')}

## Thống kê
- Tổng bình luận: {analysis['statistics']['total_comments']}
- Điểm cảm xúc trung bình: {analysis['statistics']['overall_score']:.2f}
- Bình luận tích cực: {analysis['statistics']['positive_count']}
- Bình luận tiêu cực: {analysis['statistics']['negative_count']}
- Bình luận trung lập: {analysis['statistics']['neutral_count']}

## Tóm tắt
{analysis['summary']}
"""
        
        # Offer download
        st.download_button(
            label="📥 Tải báo cáo (TXT)",
            data=report,
            file_name=f"reddit_analysis_{meta.get('subreddit', 'report')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    def prepare_data_for_ui(self, comments):
        """Prepare data for UI visualizations"""
        data = []
        
        for idx, comment in enumerate(comments):
            # Parse timestamp if it's a string
            timestamp = comment.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            
            data.append({
                'id': idx + 1,
                'timestamp': timestamp,
                'body': comment.get('text', comment.get('body', '')),
                'author': comment.get('author', 'Ẩn danh'),
                'score': comment.get('score', 0),
                'polarity': comment.get('score', 0),
                'sentiment': comment.get('sentiment', 'Trung lập'),
                'source': 'Reddit'
            })
        
        return pd.DataFrame(data)
    
    def logout(self):
        """Logout user"""
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.page = "Login"
        st.session_state.trending_posts = []
        st.session_state.current_analysis = None
        st.rerun()
    
    def run(self):
        """Run the application"""
        # Initialize UI
        if UI_AVAILABLE:
            load_css()
        
        # Check authentication
        if not st.session_state.authenticated:
            self.render_login_page()
            return
        
        # Page routing
        page = st.session_state.page
        
        if page == "Dashboard":
            self.render_dashboard_page()
        elif page == "Analysis":
            self.render_analysis_page()
        else:
            # Default to dashboard
            st.session_state.page = "Dashboard"
            st.rerun()


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    """Main function"""
    # Add custom CSS for better appearance
    st.markdown("""
    <style>
    .stAlert {
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app = RedditSentimentApp()
    app.run()

if __name__ == "__main__":
    main()