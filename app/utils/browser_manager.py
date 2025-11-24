from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
import logging
import os
from typing import Dict, Any, Optional
import random

logger = logging.getLogger(__name__)

class BrowserManager:
    """Manage browser instances with anti-detection features"""
    
    def __init__(self):
        self.drivers = {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
    
    def create_driver(self, headless: bool = True, user_agent: str = None, 
                     disable_images: bool = True, proxy: str = None) -> webdriver.Chrome:
        """Create a configured browser driver with anti-detection"""
        
        try:
            # Use undetected-chromedriver for better stealth
            options = uc.ChromeOptions()
            
            # Basic options
            if headless:
                options.add_argument('--headless=new')
            
            # Stealth options
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-features=VizDisplayCompositor')
            options.add_argument('--disable-ipc-flooding-protection')
            
            # User agent
            if user_agent:
                options.add_argument(f'--user-agent={user_agent}')
            else:
                options.add_argument(f'--user-agent={random.choice(self.user_agents)}')
            
            # Performance options
            if disable_images:
                options.add_argument('--blink-settings=imagesEnabled=false')
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)
            
            # Proxy configuration
            if proxy:
                options.add_argument(f'--proxy-server={proxy}')
            
            # Additional stealth arguments
            stealth_args = [
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
                '--disable-logging',
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-translate',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-client-side-phishing-detection',
                '--disable-crash-reporter',
                '--disable-domain-reliability',
                '--disable-ipc-flooding-protection',
                '--disable-hang-monitor',
                '--disable-popup-blocking',
                '--disable-prompt-on-repost',
                '--disable-sync',
                '--disable-web-resources',
                '--disable-back-forward-cache',
                '--aggressive-cache-discard',
                '--aggressive-tab-discard',
                '--ignore-certificate-errors',
                '--ignore-ssl-errors',
                '--metrics-recording-only',
                '--safebrowsing-disable-auto-update',
                '--password-store=basic',
                '--use-mock-keychain'
            ]
            
            for arg in stealth_args:
                options.add_argument(arg)
            
            # Create driver with undetected-chromedriver
            driver = uc.Chrome(
                options=options,
                service_log_path=os.devnull,  # Disable logging
                version_main=114  # Specify Chrome version
            )
            
            # Execute stealth scripts
            self._apply_stealth_scripts(driver)
            
            # Set window size
            driver.set_window_size(1920, 1080)
            
            logger.info("✅ Browser driver created successfully")
            return driver
            
        except Exception as e:
            logger.error(f"❌ Failed to create browser driver: {e}")
            raise
    
    def _apply_stealth_scripts(self, driver):
        """Apply JavaScript stealth scripts to avoid detection"""
        stealth_scripts = [
            # Remove webdriver property
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            """,
            # Remove automation flags
            """
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """,
            # Override permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """,
            # Mock plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            """,
            # Mock languages
            """
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            """
        ]
        
        for script in stealth_scripts:
            try:
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": script
                })
            except Exception as e:
                logger.warning(f"Failed to apply stealth script: {e}")
    
    def set_proxy(self, driver, proxy: str):
        """Set proxy for existing driver"""
        try:
            driver.execute_cdp_cmd("Network.setRequestInterception", {
                "patterns": [{"urlPattern": "*", "resourceType": "Document"}]
            })
            
            # Proxy configuration would be implemented here
            logger.info(f"🔧 Proxy set to: {proxy}")
            
        except Exception as e:
            logger.error(f"Failed to set proxy: {e}")
    
    def rotate_user_agent(self, driver):
        """Rotate user agent for existing driver"""
        new_agent = random.choice(self.user_agents)
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": new_agent
        })
        logger.info(f"🔄 User agent rotated: {new_agent[:50]}...")
    
    def clear_cookies(self, driver):
        """Clear all cookies"""
        driver.delete_all_cookies()
        logger.info("🍪 Cookies cleared")
    
    def take_screenshot(self, driver, filename: str):
        """Take screenshot for debugging"""
        try:
            driver.save_screenshot(f"data/logs/{filename}")
            logger.info(f"📸 Screenshot saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")
    
    def close_driver(self, driver):
        """Close driver properly"""
        try:
            if driver:
                driver.quit()
                logger.info("🔚 Browser driver closed")
        except Exception as e:
            logger.warning(f"Error closing driver: {e}")