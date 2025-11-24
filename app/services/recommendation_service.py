from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from app.core.user_database import UserGroup

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db

    def get_group_recommendations(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get mock recommendations for user's interest groups"""
        user_groups = self.db.query(UserGroup).filter(
            UserGroup.user_id == user_id,
            UserGroup.is_active == True
        ).all()
        
        recommendations = []
        
        for group in user_groups:
            # Mock recommendations based on subreddit
            mock_trends = self._get_mock_trends(group.subreddit)
            
            for trend in mock_trends:
                recommendations.append({
                    "group_name": group.group_name,
                    "subreddit": group.subreddit,
                    "trend_topic": trend["topic"],
                    "sentiment_score": trend["sentiment_score"],
                    "trend_score": trend["trend_score"],
                    "post_count": trend["post_count"],
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Sort by trend score and limit results
        recommendations.sort(key=lambda x: x["trend_score"], reverse=True)
        return recommendations[:limit]

    def _get_mock_trends(self, subreddit: str) -> List[Dict[str, Any]]:
        """Generate mock trend data for demonstration"""
        trends_data = {
            "technology": [
                {"topic": "AI Breakthrough", "sentiment_score": 0.8, "trend_score": 0.9, "post_count": 45},
                {"topic": "New Smartphone", "sentiment_score": 0.6, "trend_score": 0.7, "post_count": 32},
            ],
            "programming": [
                {"topic": "New Framework", "sentiment_score": 0.7, "trend_score": 0.8, "post_count": 28},
                {"topic": "Debugging Tips", "sentiment_score": 0.5, "trend_score": 0.6, "post_count": 21},
            ],
            "datascience": [
                {"topic": "Machine Learning", "sentiment_score": 0.9, "trend_score": 0.85, "post_count": 38},
                {"topic": "Data Visualization", "sentiment_score": 0.6, "trend_score": 0.7, "post_count": 25},
            ],
            "python": [
                {"topic": "New Python Features", "sentiment_score": 0.8, "trend_score": 0.9, "post_count": 52},
                {"topic": "Best Practices", "sentiment_score": 0.7, "trend_score": 0.8, "post_count": 41},
            ]
        }
        
        return trends_data.get(subreddit.lower(), [
            {"topic": f"Trending in {subreddit}", "sentiment_score": 0.5, "trend_score": 0.6, "post_count": 15}
        ])

    def get_suggested_groups(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get suggested groups based on user's current interests"""
        user_groups = self.db.query(UserGroup).filter(
            UserGroup.user_id == user_id,
            UserGroup.is_active == True
        ).all()
        
        # Popular subreddits that might interest the user
        popular_subreddits = [
            {"name": "Technology News", "subreddit": "technology"},
            {"name": "Programming", "subreddit": "programming"},
            {"name": "Data Science", "subreddit": "datascience"},
            {"name": "Machine Learning", "subreddit": "MachineLearning"},
            {"name": "Artificial Intelligence", "subreddit": "artificial"},
            {"name": "Web Development", "subreddit": "webdev"},
            {"name": "Startups", "subreddit": "startups"},
            {"name": "Python", "subreddit": "python"},
        ]
        
        # Filter out subreddits user already follows
        user_subreddits = [group.subreddit for group in user_groups]
        suggestions = [sub for sub in popular_subreddits if sub["subreddit"] not in user_subreddits]
        
        return suggestions[:limit]