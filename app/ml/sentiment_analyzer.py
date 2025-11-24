import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analyzer with multiple model support"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            logger.info(f"🔄 Loading sentiment model: {self.model_name}")
            
            # Sử dụng model đơn giản hơn, ổn định hơn
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
                device=-1  # Luôn dùng CPU để tránh lỗi
            )
            
            logger.info("✅ Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            logger.info("🔄 Falling back to rule-based analysis")
            self.pipeline = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text - SYNCHRONOUS VERSION"""
        try:
            # Debug info
            logger.debug(f"Analyzing text: {text[:100]}...")
            
            # If model failed to load or text is too short, use rule-based
            if self.pipeline is None or not text or len(text.strip()) < 3:
                return self._rule_based_analysis(text)
            
            # Clean text
            cleaned_text = self._clean_text(text)
            if not cleaned_text or len(cleaned_text.strip()) < 3:
                return self._rule_based_analysis(text)
            
            # Use pipeline for prediction - SYNCHRONOUS
            # Giới hạn độ dài văn bản để tránh lỗi
            if len(cleaned_text) > 1000:
                cleaned_text = cleaned_text[:1000]
                
            results = self.pipeline(cleaned_text)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
            else:
                result = results
            
            # Map labels to standard format
            label = str(result['label']).lower()
            score = float(result['score'])
            
            # Standardize labels
            if 'positive' in label:
                sentiment = 'positive'
            elif 'negative' in label:
                sentiment = 'negative'
            elif 'neutral' in label:
                sentiment = 'neutral'
            else:
                # Mặc định dựa trên score
                sentiment = 'positive' if score > 0.6 else 'negative' if score < 0.4 else 'neutral'
            
            logger.debug(f"Analysis result: {sentiment} (score: {score})")
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': score,
                'model': 'transformer'
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis error: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_analysis(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch - SYNCHRONOUS VERSION"""
        try:
            if self.pipeline is None:
                return [self._rule_based_analysis(text) for text in texts]
            
            # Clean and filter texts
            cleaned_texts = []
            original_indices = []
            
            for i, text in enumerate(texts):
                cleaned = self._clean_text(text)
                if cleaned and len(cleaned.strip()) >= 3:
                    if len(cleaned) > 1000:
                        cleaned = cleaned[:1000]
                    cleaned_texts.append(cleaned)
                    original_indices.append(i)
            
            if not cleaned_texts:
                return [self._rule_based_analysis(text) for text in texts]
            
            # Batch prediction với xử lý lỗi
            try:
                results = self.pipeline(cleaned_texts)
            except Exception as e:
                logger.error(f"❌ Batch prediction failed: {e}")
                return [self._rule_based_analysis(text) for text in texts]
            
            # Reconstruct full results list
            full_results = []
            result_idx = 0
            
            for i in range(len(texts)):
                if i in original_indices:
                    if result_idx < len(results):
                        result = results[result_idx]
                        result_idx += 1
                        
                        label = str(result['label']).lower()
                        score = float(result['score'])
                        
                        if 'positive' in label:
                            sentiment = 'positive'
                        elif 'negative' in label:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        
                        full_results.append({
                            'sentiment': sentiment,
                            'score': score,
                            'confidence': score,
                            'model': 'transformer'
                        })
                    else:
                        full_results.append(self._rule_based_analysis(texts[i]))
                else:
                    full_results.append(self._rule_based_analysis(texts[i]))
            
            return full_results
            
        except Exception as e:
            logger.error(f"❌ Batch sentiment analysis error: {e}")
            return [self._rule_based_analysis(text) for text in texts]
    
    def _rule_based_analysis(self, text: str) -> Dict[str, Any]:
        """Rule-based sentiment analysis as fallback"""
        if not text or len(text.strip()) < 3:
            return self._get_default_result()
        
        try:
            # Enhanced sentiment words với trọng số
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'love', 'nice', 'best', 'happy', 
                'awesome', 'fantastic', 'wonderful', 'perfect', 'brilliant', 'outstanding',
                'superb', 'terrific', 'favorite', 'beautiful', 'impressive', 'recommend',
                'outstanding', 'exceptional', 'marvelous', 'pleasant', 'delightful',
                'enjoy', 'enjoyable', 'satisfied', 'pleased', 'outstanding'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry',
                'disappointing', 'disappointed', 'poor', 'rubbish', 'garbage', 'trash',
                'useless', 'waste', 'boring', 'annoying', 'frustrating', 'broken',
                'hated', 'dislike', 'terrible', 'awful', 'horrible', 'disgusting',
                'frustrated', 'angry', 'mad', 'upset', 'displeased'
            }
            
            # Strong sentiment modifiers
            strong_positive = {'love', 'amazing', 'awesome', 'fantastic', 'perfect', 'brilliant'}
            strong_negative = {'hate', 'terrible', 'horrible', 'disgusting', 'awful'}
            
            text_lower = text.lower()
            words = text_lower.split()
            
            positive_score = 0
            negative_score = 0
            
            for word in words:
                if word in strong_positive:
                    positive_score += 2
                elif word in positive_words:
                    positive_score += 1
                elif word in strong_negative:
                    negative_score += 2
                elif word in negative_words:
                    negative_score += 1
            
            # Check for negations
            negations = {'not', "don't", 'never', 'no', 'cannot', "won't", "can't"}
            for i, word in enumerate(words):
                if word in negations and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in positive_words:
                        negative_score += 1
                        positive_score -= 1
                    elif next_word in negative_words:
                        positive_score += 1
                        negative_score -= 1
            
            # Calculate final sentiment
            total_score = positive_score + negative_score
            if total_score == 0:
                sentiment = 'neutral'
                confidence = 0.5
            elif positive_score > negative_score:
                sentiment = 'positive'
                confidence = min(0.95, 0.5 + (positive_score / 10))
            else:
                sentiment = 'negative'
                confidence = min(0.95, 0.5 + (negative_score / 10))
            
            # Adjust score for return format
            score = confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0.0
            
            return {
                'sentiment': sentiment,
                'score': float(score),
                'confidence': float(confidence),
                'model': 'rule-based'
            }
            
        except Exception as e:
            logger.error(f"❌ Rule-based analysis error: {e}")
            return self._get_default_result()
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        if not text:
            return ""
        
        try:
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\!\?,]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
        except Exception as e:
            logger.error(f"❌ Text cleaning error: {e}")
            return text if text else ""
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Get default result when analysis fails"""
        return {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.5,
            'model': 'default'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_type': 'transformer' if self.pipeline else 'rule-based',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': 512,
            'status': 'loaded' if self.pipeline else 'rule-based'
        }
    
    def __del__(self):
        """Cleanup executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Test function
def test_sentiment_analyzer():
    """Test the sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible and awful.",
        "It's okay, nothing special.",
        "The weather is nice today."
    ]
    
    print("Testing Sentiment Analyzer:")
    print("=" * 50)
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Result: {result}")
        print("-" * 30)

if __name__ == "__main__":
    test_sentiment_analyzer()