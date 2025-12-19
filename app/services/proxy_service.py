# ProxyService v2 - Production-grade
# Improvements:
# - ThreadPool validation (uses max_workers)
# - Proxy scoring & cooldown
# - Safe separation of proxies / working pool
# - Faster selection & cleanup

import requests
import json
import time
import logging
from typing import List, Dict, Optional
import random
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProxyService:
    """Advanced proxy rotation & validation service"""

    TEST_URLS = [
        "https://httpbin.org/ip",
        "https://api.ipify.org?format=json"
    ]

    def __init__(self, proxy_file: str = None, cooldown_seconds: int = 60):
        self.proxy_file = proxy_file or 'proxies/proxy_list.json'
        self.cooldown_seconds = cooldown_seconds

        self.proxies: List[Dict] = []
        self.working_proxies: Dict[str, Dict] = {}
        self.last_update = None

        self.load_proxies()

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    def load_proxies(self):
        if not os.path.exists(self.proxy_file):
            self._create_default_proxy_file()
            return

        try:
            with open(self.proxy_file, 'r') as f:
                data = json.load(f)
                self.proxies = data.get('proxies', [])
                self.working_proxies = {
                    p['address']: p for p in data.get('working_proxies', [])
                }
                self.last_update = data.get('last_update')

            logger.info(f"Loaded {len(self.proxies)} proxies, {len(self.working_proxies)} working")
        except Exception as e:
            logger.error(f"Failed loading proxies: {e}")
            self._create_default_proxy_file()

    def save_proxies(self):
        try:
            data = {
                'proxies': self.proxies,
                'working_proxies': list(self.working_proxies.values()),
                'last_update': datetime.now().isoformat()
            }
            with open(self.proxy_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.last_update = data['last_update']
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def _create_default_proxy_file(self):
        os.makedirs('proxies', exist_ok=True)
        with open(self.proxy_file, 'w') as f:
            json.dump({'proxies': [], 'working_proxies': [], 'last_update': None}, f, indent=2)

    # -------------------------------------------------
    # Core logic
    # -------------------------------------------------
    def add_proxy(self, address: str, proxy_type: str = 'http'):
        self.proxies.append({
            'address': address,
            'type': proxy_type,
            'success': 0,
            'fail': 0,
            'score': 0.0,
            'last_used': None,
            'last_checked': None,
            'cooldown_until': None,
            'response_time': None
        })
        self.save_proxies()

    def _in_cooldown(self, proxy: Dict) -> bool:
        cd = proxy.get('cooldown_until')
        return cd and datetime.now() < datetime.fromisoformat(cd)

    def validate_proxy(self, proxy: Dict, timeout: int = 8) -> bool:
        if self._in_cooldown(proxy):
            return False

        test_url = random.choice(self.TEST_URLS)
        proxies = {
            'http': f"{proxy['type']}://{proxy['address']}",
            'https': f"{proxy['type']}://{proxy['address']}"
        }

        try:
            start = time.time()
            r = requests.get(test_url, proxies=proxies, timeout=timeout)
            elapsed = (time.time() - start) * 1000

            if r.status_code == 200:
                proxy['success'] += 1
                proxy['score'] += 1
                proxy['response_time'] = round(elapsed, 2)
                proxy['last_checked'] = datetime.now().isoformat()
                self.working_proxies[proxy['address']] = proxy
                return True
            else:
                raise Exception('Bad status')

        except Exception:
            proxy['fail'] += 1
            proxy['score'] -= 1
            proxy['cooldown_until'] = (
                datetime.now() + timedelta(seconds=self.cooldown_seconds)
            ).isoformat()
            self.working_proxies.pop(proxy['address'], None)
            return False

    def validate_all(self, max_workers: int = 10):
        logger.info(f"Validating {len(self.proxies)} proxies with {max_workers} workers")
        ok = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.validate_proxy, p) for p in self.proxies]
            for f in as_completed(futures):
                if f.result():
                    ok += 1
        self.save_proxies()
        logger.info(f"Validation done: {ok}/{len(self.proxies)} working")

    # -------------------------------------------------
    # Selection
    # -------------------------------------------------
    def get_working_proxy(self) -> Optional[Dict]:
        if not self.working_proxies:
            return None

        candidates = list(self.working_proxies.values())
        candidates.sort(key=lambda p: (-p['score'], p.get('response_time', 9999)))

        proxy = random.choice(candidates[:min(5, len(candidates))])
        proxy['last_used'] = datetime.now().isoformat()
        self.save_proxies()
        return proxy

    def get_proxy_string(self, proxy: Dict) -> str:
        return f"{proxy['type']}://{proxy['address']}"

    # -------------------------------------------------
    # Maintenance
    # -------------------------------------------------
    def cleanup(self, max_fail: int = 5):
        before = len(self.proxies)
        self.proxies = [p for p in self.proxies if p['fail'] < max_fail]
        self.working_proxies = {
            p['address']: p for p in self.proxies if p['address'] in self.working_proxies
        }
        removed = before - len(self.proxies)
        if removed:
            logger.info(f"Removed {removed} dead proxies")
            self.save_proxies()