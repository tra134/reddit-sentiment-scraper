import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Advanced emotion analysis using transformer models"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the emotion analysis model"""
        try:
            logger.info(f"🔄 Loading emotion model: {self.model_name}")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,  # Get all emotions
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
                function_to_apply="softmax"  # Ensure probabilities sum to 1
            )
            
            # Also load model and tokenizer separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("✅ Using GPU for emotion analysis")
            else:
                logger.info("✅ Using CPU for emotion analysis")
            
            logger.info("✅ Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load emotion model: {e}")
            # Try alternative model
            try:
                logger.info("🔄 Trying alternative emotion model...")
                self.model_name = "bhadresh-savani/bert-base-uncased-emotion"
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    top_k=None,
                    truncation=True,
                    max_length=512
                )
                logger.info("✅ Alternative emotion model loaded successfully")
            except Exception as alt_e:
                logger.error(f"❌ Failed to load alternative emotion model: {alt_e}")
                raise
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze emotions in text asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_sync,
                text
            )
            return result
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return self._get_default_result()
    
    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous emotion analysis"""
        try:
            cleaned_text = self._clean_text(text)
            if not cleaned_text or len(cleaned_text.strip()) < 3:
                return self._get_default_result()
            
            # Get emotion predictions
            results = self.pipeline(cleaned_text)
            
            if results and isinstance(results, list) and len(results) > 0:
                emotions = results[0]  # First item contains all emotions
                
                # Convert to dictionary and normalize
                emotion_scores = {}
                total_score = 0
                
                for emotion in emotions:
                    label = emotion['label'].lower()
                    score = emotion['score']
                    emotion_scores[label] = score
                    total_score += score
                
                # Normalize scores to ensure they sum to 1
                if total_score > 0:
                    emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
                
                # Find dominant emotion
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                
                # Calculate emotion intensity
                emotion_intensity = self._calculate_emotion_intensity(emotion_scores)
                
                # Get emotion categories
                emotion_categories = self._categorize_emotions(emotion_scores)
                
                return {
                    'scores': emotion_scores,
                    'dominant_emotion': dominant_emotion[0],
                    'dominant_score': dominant_emotion[1],
                    'emotion_intensity': emotion_intensity,
                    'emotion_categories': emotion_categories,
                    'model': self.model_name.split('/')[-1],
                    'analyzed_at': datetime.now().isoformat(),
                    'raw_results': emotions,
                    'confidence': dominant_emotion[1]  # Confidence in dominant emotion
                }
            else:
                return self._get_default_result()
                
        except Exception as e:
            logger.error(f"Sync emotion analysis error: {e}")
            return self._get_default_result()
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze emotions for multiple texts in batch"""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._analyze_batch_sync,
                texts
            )
            return results
        except Exception as e:
            logger.error(f"Batch emotion analysis error: {e}")
            return [self._get_default_result() for _ in texts]
    
    def _analyze_batch_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Synchronous batch emotion analysis"""
        try:
            cleaned_texts = [self._clean_text(text) for text in texts]
            valid_texts = []
            original_indices = []
            
            for i, text in enumerate(cleaned_texts):
                if text and len(text.strip()) >= 3:
                    valid_texts.append(text)
                    original_indices.append(i)
            
            if not valid_texts:
                return [self._get_default_result() for _ in texts]
            
            # Batch prediction
            results = self.pipeline(valid_texts)
            processed_results = []
            
            for i, result in enumerate(results):
                if result and len(result) > 0:
                    emotions = result
                    emotion_scores = {}
                    total_score = 0
                    
                    for emotion in emotions:
                        label = emotion['label'].lower()
                        score = emotion['score']
                        emotion_scores[label] = score
                        total_score += score
                    
                    # Normalize scores
                    if total_score > 0:
                        emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
                    
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    emotion_intensity = self._calculate_emotion_intensity(emotion_scores)
                    emotion_categories = self._categorize_emotions(emotion_scores)
                    
                    processed_results.append({
                        'scores': emotion_scores,
                        'dominant_emotion': dominant_emotion[0],
                        'dominant_score': dominant_emotion[1],
                        'emotion_intensity': emotion_intensity,
                        'emotion_categories': emotion_categories,
                        'model': self.model_name.split('/')[-1],
                        'analyzed_at': datetime.now().isoformat(),
                        'confidence': dominant_emotion[1]
                    })
                else:
                    processed_results.append(self._get_default_result())
            
            # Reconstruct full results list
            full_results = []
            result_idx = 0
            
            for i in range(len(texts)):
                if i in original_indices:
                    full_results.append(processed_results[result_idx])
                    result_idx += 1
                else:
                    full_results.append(self._get_default_result())
            
            return full_results
            
        except Exception as e:
            logger.error(f"Batch sync emotion analysis error: {e}")
            return [self._get_default_result() for _ in texts]
    
    def _clean_text(self, text: str) -> str:
        """Clean text for emotion analysis"""
        if not text:
            return ""
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        # Remove Reddit markdown and special characters
        text = re.sub(r'[^\w\s\.\!\?,]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _calculate_emotion_intensity(self, emotion_scores: Dict[str, float]) -> str:
        """Calculate overall emotion intensity"""
        max_score = max(emotion_scores.values()) if emotion_scores else 0
        
        if max_score > 0.8:
            return "very_strong"
        elif max_score > 0.6:
            return "strong"
        elif max_score > 0.4:
            return "moderate"
        elif max_score > 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _categorize_emotions(self, emotion_scores: Dict[str, float]) -> List[str]:
        """Categorize emotions into groups"""
        categories = []
        
        # Positive emotions
        positive_emotions = ['joy', 'surprise']
        positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
        
        # Negative emotions
        negative_emotions = ['anger', 'disgust', 'fear', 'sadness']
        negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
        
        # Neutral
        neutral_score = emotion_scores.get('neutral', 0)
        
        if positive_score > 0.5:
            categories.append('positive')
        elif negative_score > 0.5:
            categories.append('negative')
        elif neutral_score > 0.7:
            categories.append('neutral')
        else:
            categories.append('mixed')
        
        # Add intensity category
        max_score = max(emotion_scores.values()) if emotion_scores else 0
        if max_score > 0.7:
            categories.append('focused')  # One dominant emotion
        else:
            categories.append('diffuse')  # Multiple emotions
        
        return categories
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Get default emotion analysis result"""
        return {
            'scores': {'neutral': 1.0},
            'dominant_emotion': 'neutral',
            'dominant_score': 1.0,
            'emotion_intensity': 'very_weak',
            'emotion_categories': ['neutral', 'diffuse'],
            'model': self.model_name.split('/')[-1],
            'analyzed_at': datetime.now().isoformat(),
            'confidence': 1.0
        }
    
    def get_emotion_statistics(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistics for emotion scores"""
        if not emotion_scores:
            return {}
        
        scores = list(emotion_scores.values())
        
        return {
            'max_score': max(scores),
            'min_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'entropy': self._calculate_entropy(scores),
            'dominance_ratio': max(scores) / sum(scores) if sum(scores) > 0 else 0
        }
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of emotion distribution"""
        scores = np.array(scores)
        scores = scores[scores > 0]  # Remove zeros
        if len(scores) == 0:
            return 0
        
        scores = scores / np.sum(scores)  # Normalize
        return -np.sum(scores * np.log(scores))
    
    def detect_emotion_shifts(self, previous_scores: Dict[str, float], 
                            current_scores: Dict[str, float]) -> Dict[str, Any]:
        """Detect significant emotion shifts between analyses"""
        if not previous_scores or not current_scores:
            return {'significant_shift': False}
        
        # Calculate differences
        all_emotions = set(previous_scores.keys()) | set(current_scores.keys())
        differences = {}
        
        for emotion in all_emotions:
            prev = previous_scores.get(emotion, 0)
            curr = current_scores.get(emotion, 0)
            differences[emotion] = curr - prev
        
        # Find largest changes
        largest_increase = max(differences.items(), key=lambda x: x[1]) if differences else (None, 0)
        largest_decrease = min(differences.items(), key=lambda x: x[1]) if differences else (None, 0)
        
        # Check if shift is significant
        max_change = max(abs(largest_increase[1]), abs(largest_decrease[1]))
        significant_shift = max_change > 0.3  # 30% threshold
        
        return {
            'significant_shift': significant_shift,
            'largest_increase': {
                'emotion': largest_increase[0],
                'change': largest_increase[1]
            },
            'largest_decrease': {
                'emotion': largest_decrease[0],
                'change': largest_decrease[1]
            },
            'all_differences': differences
        }
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotion labels"""
        return self.emotion_labels
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get emotion model information"""
        return {
            'model_name': self.model_name,
            'emotions': self.emotion_labels,
            'model_type': 'transformer',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': 512,
            'version': '1.0'
        }
    
    def validate_emotion_scores(self, emotion_scores: Dict[str, float]) -> bool:
        """Validate that emotion scores are reasonable"""
        if not emotion_scores:
            return False
        
        # Check if scores sum to approximately 1
        total = sum(emotion_scores.values())
        if abs(total - 1.0) > 0.1:  # Allow 10% tolerance
            logger.warning(f"Emotion scores don't sum to 1: {total}")
            return False
        
        # Check for valid emotion labels
        for emotion in emotion_scores.keys():
            if emotion not in self.emotion_labels:
                logger.warning(f"Invalid emotion label: {emotion}")
                return False
        
        return True
    
    async def analyze_emotion_trends(self, emotion_data_series: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze emotion trends over time"""
        if len(emotion_data_series) < 2:
            return {'trend_available': False}
        
        trends = {}
        
        for emotion in self.emotion_labels:
            scores = [data.get(emotion, 0) for data in emotion_data_series]
            if len(scores) > 1:
                # Simple trend calculation
                first_half = np.mean(scores[:len(scores)//2])
                second_half = np.mean(scores[len(scores)//2:])
                trend = 'increasing' if second_half > first_half else 'decreasing' if second_half < first_half else 'stable'
                
                trends[emotion] = {
                    'trend': trend,
                    'change': second_half - first_half,
                    'volatility': np.std(scores)
                }
        
        return {
            'trend_available': True,
            'emotion_trends': trends,
            'most_increasing': max(trends.items(), key=lambda x: x[1]['change']) if trends else None,
            'most_decreasing': min(trends.items(), key=lambda x: x[1]['change']) if trends else None
        }

# Global emotion analyzer instance
emotion_analyzer = EmotionAnalyzer()