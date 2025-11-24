from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class EmotionLabel(str, Enum):
    """Emotion classification labels"""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    LOVE = "love"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    
    label: SentimentLabel
    score: float
    confidence: float
    model_used: str
    raw_output: Optional[Dict[str, Any]] = None
    analyzed_at: datetime = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentResult':
        """Create from dictionary"""
        return cls(**data)
    
    def is_positive(self) -> bool:
        """Check if sentiment is positive"""
        return self.label == SentimentLabel.POSITIVE
    
    def is_negative(self) -> bool:
        """Check if sentiment is negative"""
        return self.label == SentimentLabel.NEGATIVE
    
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral"""
        return self.label == SentimentLabel.NEUTRAL
    
    def get_sentiment_strength(self) -> float:
        """Get absolute sentiment strength"""
        return abs(self.score)
    
    def get_confidence_level(self) -> str:
        """Get confidence level as string"""
        if self.confidence >= 0.9:
            return "very_high"
        elif self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "low"

@dataclass
class EmotionResult:
    """Data class for emotion analysis results"""
    
    dominant_emotion: EmotionLabel
    emotion_scores: Dict[str, float]
    model_used: str
    analyzed_at: datetime = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionResult':
        """Create from dictionary"""
        return cls(**data)
    
    def get_top_emotions(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top K emotions by score"""
        sorted_emotions = sorted(
            self.emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {'emotion': emotion, 'score': score}
            for emotion, score in sorted_emotions
        ]
    
    def get_emotion_score(self, emotion: str) -> float:
        """Get score for specific emotion"""
        return self.emotion_scores.get(emotion, 0.0)
    
    def has_strong_emotion(self, threshold: float = 0.7) -> bool:
        """Check if any emotion exceeds threshold"""
        return any(score >= threshold for score in self.emotion_scores.values())
    
    def get_emotion_diversity(self) -> float:
        """Calculate emotion diversity (entropy-like measure)"""
        total = sum(self.emotion_scores.values())
        if total == 0:
            return 0.0
        
        # Normalize scores
        normalized_scores = [score / total for score in self.emotion_scores.values()]
        
        # Calculate diversity (1 - max score)
        return 1 - max(normalized_scores)

@dataclass
class AspectResult:
    """Data class for aspect-based analysis results"""
    
    aspect: str
    sentiment: SentimentResult
    confidence: float
    context: Optional[str] = None
    keywords: List[str] = None
    analyzed_at: datetime = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
        if self.keywords is None:
            self.keywords = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AspectResult':
        """Create from dictionary"""
        return cls(**data)
    
    def is_positive_aspect(self) -> bool:
        """Check if aspect sentiment is positive"""
        return self.sentiment.is_positive()
    
    def is_negative_aspect(self) -> bool:
        """Check if aspect sentiment is negative"""
        return self.sentiment.is_negative()
    
    def get_aspect_importance(self) -> float:
        """Calculate aspect importance score"""
        # Combine sentiment strength and confidence
        sentiment_strength = self.sentiment.get_sentiment_strength()
        return sentiment_strength * self.confidence

@dataclass
class AnalysisSummary:
    """Data class for comprehensive analysis summary"""
    
    analysis_id: str
    post_id: str
    total_comments: int
    analyzed_comments: int
    sentiment_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    aspect_summary: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    overall_sentiment: SentimentResult
    dominant_emotion: EmotionLabel
    analysis_duration: float
    analyzed_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if not self.analysis_id:
            self.analysis_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSummary':
        """Create from dictionary"""
        return cls(**data)
    
    def get_sentiment_percentages(self) -> Dict[str, float]:
        """Calculate sentiment percentages"""
        total = sum(self.sentiment_distribution.values())
        if total == 0:
            return {}
        
        return {
            sentiment: (count / total) * 100
            for sentiment, count in self.sentiment_distribution.items()
        }
    
    def get_emotion_percentages(self) -> Dict[str, float]:
        """Calculate emotion percentages"""
        total = sum(self.emotion_distribution.values())
        if total == 0:
            return {}
        
        return {
            emotion: (count / total) * 100
            for emotion, count in self.emotion_distribution.items()
        }
    
    def get_success_rate(self) -> float:
        """Get analysis success rate"""
        if self.total_comments == 0:
            return 0.0
        return (self.analyzed_comments / self.total_comments) * 100
    
    def get_engagement_score(self) -> float:
        """Calculate overall engagement score"""
        metrics = self.engagement_metrics
        avg_score = metrics.get('average_score', 0)
        total_engagement = metrics.get('total_engagement', 0)
        comment_ratio = metrics.get('comments_per_hour', 0)
        
        # Normalize and weight factors
        score_component = min(avg_score / 100, 1.0)  # Normalize score
        engagement_component = min(total_engagement / 1000, 1.0)  # Normalize engagement
        activity_component = min(comment_ratio / 10, 1.0)  # Normalize activity
        
        # Weighted average
        return (score_component * 0.4 + engagement_component * 0.4 + activity_component * 0.2) * 100
    
    def get_key_insights(self) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Sentiment insights
        sentiment_pct = self.get_sentiment_percentages()
        if sentiment_pct.get('positive', 0) > 60:
            insights.append("Overwhelmingly positive community response")
        elif sentiment_pct.get('negative', 0) > 60:
            insights.append("Significant negative sentiment detected")
        elif sentiment_pct.get('neutral', 0) > 70:
            insights.append("Mostly neutral or mixed responses")
        
        # Emotion insights
        emotion_pct = self.get_emotion_percentages()
        if emotion_pct.get('joy', 0) > 40:
            insights.append("Community expresses strong positive emotions")
        if emotion_pct.get('anger', 0) > 30:
            insights.append("Notable frustration or anger in discussions")
        
        # Aspect insights
        if self.aspect_summary:
            top_aspect = max(
                self.aspect_summary.items(),
                key=lambda x: x[1].get('total_mentions', 0)
            ) if self.aspect_summary else None
            
            if top_aspect:
                aspect_name, aspect_data = top_aspect
                dominant_sentiment = aspect_data.get('dominant_sentiment', 'neutral')
                mentions = aspect_data.get('total_mentions', 0)
                
                insights.append(
                    f"Most discussed topic: '{aspect_name}' ({mentions} mentions) "
                    f"with {dominant_sentiment} sentiment"
                )
        
        # Engagement insights
        engagement_score = self.get_engagement_score()
        if engagement_score > 80:
            insights.append("Very high community engagement")
        elif engagement_score < 30:
            insights.append("Low community engagement")
        
        return insights
    
    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        sentiment_pct = self.get_sentiment_percentages()
        engagement_score = self.get_engagement_score()
        
        if sentiment_pct.get('negative', 0) > 40:
            recommendations.append("Consider addressing concerns raised in negative comments")
        
        if engagement_score < 50:
            recommendations.append("Explore ways to increase community engagement")
        
        if self.dominant_emotion == EmotionLabel.ANGER:
            recommendations.append("Monitor discussions for potential conflicts")
        
        if len(self.aspect_summary) > 5:
            recommendations.append("Multiple topics discussed - consider focused follow-up")
        
        return recommendations

@dataclass
class CommentAnalysis:
    """Data class for individual comment analysis"""
    
    comment_id: str
    sentiment: SentimentResult
    emotion: EmotionResult
    aspects: List[AspectResult]
    text_metrics: Dict[str, Any]
    analyzed_at: datetime = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentAnalysis':
        """Create from dictionary"""
        return cls(**data)
    
    def get_overall_confidence(self) -> float:
        """Calculate overall analysis confidence"""
        sentiment_conf = self.sentiment.confidence
        emotion_conf = max(self.emotion.emotion_scores.values()) if self.emotion.emotion_scores else 0
        
        if self.aspects:
            aspect_conf = sum(aspect.confidence for aspect in self.aspects) / len(self.aspects)
        else:
            aspect_conf = 0
        
        # Weighted average
        return (sentiment_conf * 0.5 + emotion_conf * 0.3 + aspect_conf * 0.2)
    
    def has_strong_sentiment(self, threshold: float = 0.7) -> bool:
        """Check if comment has strong sentiment"""
        return self.sentiment.get_sentiment_strength() >= threshold
    
    def get_key_aspects(self, top_k: int = 3) -> List[AspectResult]:
        """Get top aspects by importance"""
        sorted_aspects = sorted(
            self.aspects,
            key=lambda x: x.get_aspect_importance(),
            reverse=True
        )[:top_k]
        return sorted_aspects
    
    def is_controversial(self, sentiment_threshold: float = 0.3, 
                        emotion_threshold: float = 0.4) -> bool:
        """Check if comment shows controversial characteristics"""
        # Mixed sentiment and strong emotions
        sentiment_strength = self.sentiment.get_sentiment_strength()
        has_strong_emotion = self.emotion.has_strong_emotion(emotion_threshold)
        
        return (sentiment_strength < sentiment_threshold and has_strong_emotion)