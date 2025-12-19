import requests
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def __init__(self):
        # ... existing code ...
        
        # Gemini API configuration
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.gemini_available = bool(self.gemini_api_key)
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        if self.gemini_available:
            logger.info("✅ Gemini API is available for text summarization")
        else:
            logger.warning("⚠️ Gemini API key not found. Text summarization will use fallback methods.")
        
        # ... rest of existing __init__ ...