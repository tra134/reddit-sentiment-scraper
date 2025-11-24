import requests
import concurrent.futures
from typing import List, Dict
import json
import time

def check_proxy(proxy: str, timeout: int = 5) -> bool:
    """Check if a proxy is working"""
    try:
        response = requests.get(
            'https://httpbin.org/ip',
            proxies={'http': proxy, 'https': proxy},
            timeout=timeout
        )
        return response.status_code == 200
    except:
        return False

def check_proxies_from_file(input_file: str, output_file: str, max_workers: int = 10):
    """Check proxies from input file and save working ones to output file"""
    
    with open(input_file, 'r') as f:
        proxies = [line.strip() for line in f if line.strip()]
    
    working_proxies = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_proxy = {
            executor.submit(check_proxy, proxy): proxy for proxy in proxies
        }
        
        for future in concurrent.futures.as_completed(future_to_proxy):
            proxy = future_to_proxy[future]
            try:
                if future.result():
                    working_proxies.append(proxy)
                    print(f"✅ Working: {proxy}")
                else:
                    print(f"❌ Failed: {proxy}")
            except Exception as e:
                print(f"❌ Error with {proxy}: {e}")
    
    # Save working proxies
    with open(output_file, 'w') as f:
        for proxy in working_proxies:
            f.write(proxy + '\n')
    
    print(f"\n🎉 Found {len(working_proxies)} working proxies out of {len(proxies)}")
    print(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    check_proxies_from_file('proxies/input.txt', 'proxies/working.txt')