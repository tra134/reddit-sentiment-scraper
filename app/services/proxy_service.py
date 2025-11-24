import requests
import json
import time
import logging
from typing import List, Dict, Optional
import random
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class ProxyService:
    """Manage proxy rotation and validation"""
    
    def __init__(self, proxy_file: str = None):
        self.proxy_file = proxy_file or 'proxies/proxy_list.json'
        self.proxies = []
        self.working_proxies = []
        self.failed_proxies = []
        self.last_update = None
        
        self.load_proxies()
    
    def load_proxies(self):
        """Load proxies from file"""
        try:
            if os.path.exists(self.proxy_file):
                with open(self.proxy_file, 'r') as f:
                    data = json.load(f)
                    self.proxies = data.get('proxies', [])
                    self.working_proxies = data.get('working_proxies', [])
                    self.last_update = data.get('last_update')
                
                logger.info(f"✅ Loaded {len(self.proxies)} proxies, {len(self.working_proxies)} working")
            else:
                logger.warning("❌ Proxy file not found, creating empty list")
                self._create_default_proxy_file()
                
        except Exception as e:
            logger.error(f"❌ Failed to load proxies: {e}")
            self._create_default_proxy_file()
    
    def _create_default_proxy_file(self):
        """Create default proxy file structure"""
        default_data = {
            "proxies": [],
            "working_proxies": [],
            "last_update": None
        }
        
        os.makedirs('proxies', exist_ok=True)
        with open(self.proxy_file, 'w') as f:
            json.dump(default_data, f, indent=2)
        
        self.proxies = []
        self.working_proxies = []
    
    def add_proxy(self, proxy: str, proxy_type: str = "http"):
        """Add a new proxy"""
        proxy_data = {
            "address": proxy,
            "type": proxy_type,
            "added_at": datetime.now().isoformat(),
            "success_count": 0,
            "fail_count": 0,
            "last_used": None,
            "last_checked": None,
            "response_time": None
        }
        
        self.proxies.append(proxy_data)
        self.save_proxies()
        logger.info(f"➕ Added proxy: {proxy}")
    
    def validate_proxy(self, proxy_data: Dict, timeout: int = 10) -> bool:
        """Validate if proxy is working"""
        try:
            proxy_address = proxy_data['address']
            proxy_type = proxy_data['type']
            
            proxies = {
                "http": f"{proxy_type}://{proxy_address}",
                "https": f"{proxy_type}://{proxy_address}"
            }
            
            start_time = time.time()
            response = requests.get(
                "https://httpbin.org/ip",
                proxies=proxies,
                timeout=timeout
            )
            response_time = round((time.time() - start_time) * 1000, 2)  # ms
            
            if response.status_code == 200:
                proxy_data['success_count'] += 1
                proxy_data['last_checked'] = datetime.now().isoformat()
                proxy_data['response_time'] = response_time
                
                # Add to working proxies if not already there
                if proxy_data not in self.working_proxies:
                    self.working_proxies.append(proxy_data)
                
                logger.info(f"✅ Proxy validated: {proxy_address} ({response_time}ms)")
                return True
            else:
                proxy_data['fail_count'] += 1
                logger.warning(f"❌ Proxy failed validation: {proxy_address}")
                return False
                
        except Exception as e:
            proxy_data['fail_count'] += 1
            logger.warning(f"❌ Proxy error: {proxy_address} - {e}")
            return False
    
    def validate_all_proxies(self, max_workers: int = 5):
        """Validate all proxies"""
        logger.info(f"🔍 Validating {len(self.proxies)} proxies...")
        
        validated_count = 0
        for proxy in self.proxies:
            if self.validate_proxy(proxy):
                validated_count += 1
        
        self.save_proxies()
        logger.info(f"✅ Validation complete: {validated_count}/{len(self.proxies)} working")
    
    def get_working_proxy(self) -> Optional[Dict]:
        """Get a random working proxy"""
        if not self.working_proxies:
            logger.warning("⚠️ No working proxies available")
            return None
        
        # Sort by response time and success rate
        sorted_proxies = sorted(
            self.working_proxies,
            key=lambda x: (
                x.get('response_time', 9999),
                -x.get('success_count', 0)
            )
        )
        
        # Pick from top 3 fastest
        if len(sorted_proxies) > 3:
            proxy = random.choice(sorted_proxies[:3])
        else:
            proxy = random.choice(sorted_proxies)
        
        proxy['last_used'] = datetime.now().isoformat()
        self.save_proxies()
        
        logger.info(f"🔧 Selected proxy: {proxy['address']} ({proxy.get('response_time', '?')}ms)")
        return proxy
    
    def get_proxy_string(self, proxy_data: Dict) -> str:
        """Convert proxy data to string format"""
        return f"{proxy_data['type']}://{proxy_data['address']}"
    
    def import_proxies_from_file(self, file_path: str):
        """Import proxies from text file"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different proxy formats
                        if '://' in line:
                            proxy_type, address = line.split('://')
                        else:
                            proxy_type = "http"
                            address = line
                        
                        self.add_proxy(address, proxy_type)
            
            logger.info(f"📥 Imported proxies from {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to import proxies: {e}")
    
    def export_working_proxies(self, file_path: str):
        """Export working proxies to file"""
        try:
            with open(file_path, 'w') as f:
                for proxy in self.working_proxies:
                    f.write(f"{proxy['type']}://{proxy['address']}\n")
            
            logger.info(f"💾 Exported {len(self.working_proxies)} working proxies to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export proxies: {e}")
    
    def get_statistics(self) -> Dict:
        """Get proxy statistics"""
        total = len(self.proxies)
        working = len(self.working_proxies)
        success_rate = (working / total * 100) if total > 0 else 0
        
        avg_response_time = 0
        if working > 0:
            avg_response_time = sum(
                p.get('response_time', 0) for p in self.working_proxies
            ) / working
        
        return {
            "total_proxies": total,
            "working_proxies": working,
            "success_rate": round(success_rate, 2),
            "average_response_time": round(avg_response_time, 2),
            "last_update": self.last_update
        }
    
    def save_proxies(self):
        """Save proxies to file"""
        try:
            data = {
                "proxies": self.proxies,
                "working_proxies": self.working_proxies,
                "last_update": datetime.now().isoformat()
            }
            
            with open(self.proxy_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_update = data['last_update']
            
        except Exception as e:
            logger.error(f"❌ Failed to save proxies: {e}")
    
    def cleanup_failed_proxies(self, max_failures: int = 5):
        """Remove proxies with too many failures"""
        initial_count = len(self.proxies)
        
        self.proxies = [
            p for p in self.proxies 
            if p.get('fail_count', 0) < max_failures
        ]
        
        removed_count = initial_count - len(self.proxies)
        if removed_count > 0:
            logger.info(f"🗑️ Removed {removed_count} failed proxies")
            self.save_proxies()