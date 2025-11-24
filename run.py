#!/usr/bin/env python3
"""
Reddit Sentiment Analyzer Pro - Main Entry Point
Advanced web scraping and ML analysis platform
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import streamlit
        import selenium
        import transformers
        import torch
        import spacy
        import plotly
        
        logger.info("✅ All core dependencies are available")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def setup_environment():
    """Setup application environment"""
    # Create necessary directories
    os.makedirs('data/database', exist_ok=True)
    os.makedirs('data/scraped_data/raw', exist_ok=True)
    os.makedirs('data/scraped_data/processed', exist_ok=True)
    os.makedirs('data/scraped_data/archived', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('proxies', exist_ok=True)
    
    logger.info("✅ Environment setup complete")

def main():
    """Main application entry point"""
    print("🚀 Starting Reddit Sentiment Analyzer Pro...")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies from requirements.txt")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Import and run the application
    try:
        from app.main import RedditSentimentApp
        
        print("✅ Application starting...")
        print("📊 Open your browser and go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        
        # This would typically run: streamlit run app/main.py
        # For now, we'll show instructions
        print("\n🔧 To run the application, use:")
        print("   streamlit run app/main.py")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()