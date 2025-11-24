import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict, Counter
import asyncio

from app.core.database import db_manager

logger = logging.getLogger(__name__)

class TrendAnalysisService:
    """Service for analyzing trends across multiple posts and time periods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze_subreddit_trends(self, subreddit: str, days: int = 7) -> Dict[str, Any]:
        """Analyze trends in a subreddit over time"""
        try:
            # Get posts from the specified time period
            start_date = datetime.now() - timedelta(days=days)
            
            # This would typically query the database for posts in the time range
            # For now, we'll return a mock analysis
            trends = {
                'subreddit': subreddit,
                'analysis_period': f"Last {days} days",
                'total_posts_analyzed': 150,
                'total_comments_analyzed': 12500,
                'sentiment_trends': self._generate_sentiment_trends(days),
                'top_topics': self._generate_top_topics(),
                'engagement_metrics': self._generate_engagement_metrics(),
                'peak_hours': self._generate_peak_hours(),
                'recommendations': self._generate_recommendations()
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing subreddit trends: {e}")
            return {}
    
    def _generate_sentiment_trends(self, days: int) -> List[Dict[str, Any]]:
        """Generate sentiment trends data"""
        trends = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            trends.append({
                'date': date.strftime('%Y-%m-%d'),
                'positive': np.random.uniform(0.3, 0.6),
                'negative': np.random.uniform(0.1, 0.3),
                'neutral': np.random.uniform(0.2, 0.4),
                'total_posts': np.random.randint(10, 50)
            })
        
        return trends
    
    def _generate_top_topics(self) -> List[Dict[str, Any]]:
        """Generate top topics data"""
        topics = [
            {'topic': 'Technology', 'mentions': 45, 'sentiment': 'positive', 'growth': 12.5},
            {'topic': 'Gaming', 'mentions': 38, 'sentiment': 'positive', 'growth': 8.2},
            {'topic': 'Politics', 'mentions': 32, 'sentiment': 'negative', 'growth': -5.3},
            {'topic': 'Sports', 'mentions': 28, 'sentiment': 'positive', 'growth': 15.7},
            {'topic': 'Entertainment', 'mentions': 24, 'sentiment': 'neutral', 'growth': 3.1}
        ]
        return topics
    
    def _generate_engagement_metrics(self) -> Dict[str, Any]:
        """Generate engagement metrics"""
        return {
            'avg_comments_per_post': 83.2,
            'avg_upvotes_per_post': 245.7,
            'engagement_rate': 12.8,
            'viral_posts_count': 15,
            'controversial_posts_count': 8
        }
    
    def _generate_peak_hours(self) -> List[Dict[str, Any]]:
        """Generate peak activity hours"""
        hours = []
        for hour in range(24):
            activity = np.random.normal(50, 15)
            hours.append({
                'hour': f"{hour:02d}:00",
                'activity_level': max(0, activity),
                'posts_count': int(max(0, activity / 10))
            })
        return sorted(hours, key=lambda x: x['activity_level'], reverse=True)[:6]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on trends"""
        return [
            "Post during peak hours (18:00-21:00) for maximum engagement",
            "Technology-related content shows positive sentiment trends",
            "Consider avoiding political topics due to negative sentiment",
            "Gaming content has consistent positive engagement",
            "Video content tends to perform better on weekends"
        ]
    
    async def compare_subreddits(self, subreddits: List[str], days: int = 7) -> Dict[str, Any]:
        """Compare trends across multiple subreddits"""
        comparison = {
            'time_period': f"Last {days} days",
            'subreddits_compared': subreddits,
            'metrics_comparison': {}
        }
        
        for subreddit in subreddits:
            trends = await self.analyze_subreddit_trends(subreddit, days)
            comparison['metrics_comparison'][subreddit] = {
                'avg_sentiment': self._calculate_avg_sentiment(trends),
                'total_engagement': trends.get('engagement_metrics', {}).get('total_comments_analyzed', 0),
                'engagement_rate': trends.get('engagement_metrics', {}).get('engagement_rate', 0),
                'post_frequency': trends.get('total_posts_analyzed', 0) / days
            }
        
        return comparison
    
    def _calculate_avg_sentiment(self, trends: Dict[str, Any]) -> float:
        """Calculate average sentiment from trends"""
        sentiment_trends = trends.get('sentiment_trends', [])
        if not sentiment_trends:
            return 0.0
        
        total_positive = sum(day['positive'] for day in sentiment_trends)
        total_negative = sum(day['negative'] for day in sentiment_trends)
        
        return (total_positive - total_negative) / len(sentiment_trends)
    
    async def predict_trends(self, subreddit: str, forecast_days: int = 3) -> Dict[str, Any]:
        """Predict future trends based on historical data"""
        # This would use time series forecasting in a real implementation
        current_trends = await self.analyze_subreddit_trends(subreddit, 7)
        
        prediction = {
            'subreddit': subreddit,
            'forecast_period': f"Next {forecast_days} days",
            'predicted_sentiment': 'positive',
            'confidence': 0.75,
            'expected_engagement_change': 12.5,
            'recommended_actions': [
                "Continue current content strategy",
                "Focus on technology and gaming topics",
                "Schedule posts for evening hours"
            ],
            'risk_factors': [
                "Potential political discussions may affect sentiment",
                "Weekend engagement typically drops by 15%"
            ]
        }
        
        return prediction