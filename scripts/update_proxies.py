#!/usr/bin/env python3
"""
Proxy Update Script for Reddit Sentiment Analyzer
Automatically fetches and verifies new proxies
"""

import requests
import re
import time
from typing import List
import json
from pathlib import Path

def fetch_proxies_from_source(source_url: str) -> List[str]:
    """Fetch proxies from online sources"""
    try:
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()
        
        # Extract IP:PORT patterns
        proxies = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', response.text)
        return list(set(proxies))  # Remove duplicates
        
    except Exception as e:
        print(f"❌ Failed to fetch from {source_url}: {e}")
        return []

def update_proxy_list():
    """Main function to update proxy list"""
    proxy_sources = [
        "https://www.proxy-list.download/api/v1/get?type=http",
        "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http",
        # Add more proxy sources here
    ]
    
    all_proxies = []
    
    print("🔄 Fetching proxies from sources...")
    for source in proxy_sources:
        proxies = fetch_proxies_from_source(source)
        all_proxies.extend(proxies)
        print(f"📥 Found {len(proxies)} proxies from {source}")
        time.sleep(1)  # Be respectful to the sources
    
    # Remove duplicates
    all_proxies = list(set(all_proxies))
    print(f"📊 Total unique proxies found: {len(all_proxies)}")
    
    # Save to file
    output_file = Path("proxies/new_proxies.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        for proxy in all_proxies:
            f.write(proxy + '\n')
    
    print(f"💾 Saved new proxies to: {output_file}")
    print("🎯 Run proxy_checker.py to verify these proxies")

if __name__ == "__main__":
    update_proxy_list()