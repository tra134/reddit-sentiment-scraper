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
        self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            logger.info(f"🔄 Loading sentiment model: {self.model_name}")
            
            # Sử dụng model mới hơn, tốt hơn cho social media text
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
                device=-1,  # Luôn dùng CPU để tránh lỗi
                top_k=None  # Trả về tất cả scores
            )
            
            logger.info("✅ Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            logger.info("🔄 Trying fallback model: distilbert-base-uncased-finetuned-sst-2-english")
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                logger.info("✅ Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"❌ Fallback model also failed: {e2}")
                logger.info("🔄 Using rule-based analysis only")
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
            
            # Xử lý kết quả từ pipeline
            sentiment, confidence = self._process_pipeline_results(results)
            
            logger.debug(f"Analysis result: {sentiment} (confidence: {confidence})")
            
            return {
                'sentiment': sentiment,
                'score': confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0.0,
                'confidence': confidence,
                'model': 'transformer',
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis error: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_analysis(text)
    
    def _process_pipeline_results(self, results):
        """Process pipeline results and return standardized sentiment and confidence"""
        try:
            if isinstance(results, list) and len(results) > 0:
                # Nếu là batch results
                if isinstance(results[0], list):
                    # Multiple results per text (top_k)
                    result_list = results[0]
                else:
                    # Single result
                    result_list = results
            else:
                result_list = [results]
            
            # Tìm sentiment có confidence cao nhất
            best_sentiment = 'neutral'
            best_confidence = 0.5
            
            for result in result_list:
                label = str(result['label']).lower()
                score = float(result['score'])
                
                # Map labels to standard format
                current_sentiment = 'neutral'
                if any(pos in label for pos in ['positive', 'pos', 'labour']):
                    current_sentiment = 'positive'
                elif any(neg in label for neg in ['negative', 'neg', 'conservative']):
                    current_sentiment = 'negative'
                elif 'neutral' in label:
                    current_sentiment = 'neutral'
                else:
                    # Dựa trên score nếu không nhận dạng được label
                    current_sentiment = 'positive' if score > 0.6 else 'negative' if score < 0.4 else 'neutral'
                
                # Chọn sentiment với confidence cao nhất
                if score > best_confidence:
                    best_sentiment = current_sentiment
                    best_confidence = score
            
            return best_sentiment, best_confidence
            
        except Exception as e:
            logger.error(f"❌ Error processing pipeline results: {e}")
            return 'neutral', 0.5
    
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
                else:
                    # Thêm placeholder để giữ nguyên index
                    cleaned_texts.append("")
                    original_indices.append(i)
            
            # Batch prediction với xử lý lỗi
            try:
                if len(cleaned_texts) == 1:
                    # Xử lý trường hợp chỉ có 1 text
                    results = [self.pipeline(cleaned_texts[0])]
                else:
                    results = self.pipeline(cleaned_texts)
            except Exception as e:
                logger.error(f"❌ Batch prediction failed: {e}")
                return [self._rule_based_analysis(text) for text in texts]
            
            # Reconstruct full results list
            full_results = []
            
            for i, text in enumerate(texts):
                if i in original_indices and cleaned_texts[i]:  # Chỉ xử lý nếu text hợp lệ
                    try:
                        if i < len(results):
                            result = results[i] if len(cleaned_texts) > 1 else results[0]
                            sentiment, confidence = self._process_pipeline_results(result)
                            
                            full_results.append({
                                'sentiment': sentiment,
                                'score': confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0.0,
                                'confidence': confidence,
                                'model': 'transformer',
                                'model_name': self.model_name
                            })
                        else:
                            full_results.append(self._rule_based_analysis(text))
                    except Exception as e:
                        logger.error(f"❌ Error processing result for text {i}: {e}")
                        full_results.append(self._rule_based_analysis(text))
                else:
                    full_results.append(self._rule_based_analysis(text))
            
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
                'enjoy', 'enjoyable', 'satisfied', 'pleased', 'outstanding', 'thanks',
                'thank', 'cool', 'sweet', 'nice', 'decent', 'solid', 'helpful'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry',
                'disappointing', 'disappointed', 'poor', 'rubbish', 'garbage', 'trash',
                'useless', 'waste', 'boring', 'annoying', 'frustrating', 'broken',
                'hated', 'dislike', 'terrible', 'awful', 'horrible', 'disgusting',
                'frustrated', 'angry', 'mad', 'upset', 'displeased', 'sucks', 'crap'
            }
            
            # Strong sentiment modifiers
            strong_positive = {'love', 'amazing', 'awesome', 'fantastic', 'perfect', 'brilliant', 'outstanding'}
            strong_negative = {'hate', 'terrible', 'horrible', 'disgusting', 'awful', 'worst'}
            
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)  # Dùng regex để tách từ tốt hơn
            
            positive_score = 0
            negative_score = 0
            
            for word in words:
                if word in strong_positive:
                    positive_score += 3
                elif word in positive_words:
                    positive_score += 1
                elif word in strong_negative:
                    negative_score += 3
                elif word in negative_words:
                    negative_score += 1
            
            # Check for negations
            negations = {'not', "don't", 'never', 'no', 'cannot', "won't", "can't", 'isnt', "isn't", 'wasnt', "wasn't"}
            for i, word in enumerate(words):
                if word in negations and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in positive_words or next_word in strong_positive:
                        negative_score += 2
                        positive_score = max(0, positive_score - 1)
                    elif next_word in negative_words or next_word in strong_negative:
                        positive_score += 2
                        negative_score = max(0, negative_score - 1)
            
            # Check for intensifiers
            intensifiers = {'very', 'really', 'extremely', 'absolutely', 'completely'}
            for i, word in enumerate(words):
                if word in intensifiers and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in positive_words:
                        positive_score += 1
                    elif next_word in negative_words:
                        negative_score += 1
            
            # Calculate final sentiment
            total_words = len(words)
            if total_words == 0:
                return self._get_default_result()
            
            # Normalize scores
            positive_norm = positive_score / total_words
            negative_norm = negative_score / total_words
            
            sentiment_threshold = 0.05
            
            if positive_norm > negative_norm + sentiment_threshold:
                sentiment = 'positive'
                confidence = min(0.95, 0.5 + positive_norm * 2)
            elif negative_norm > positive_norm + sentiment_threshold:
                sentiment = 'negative'
                confidence = min(0.95, 0.5 + negative_norm * 2)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            # Adjust score for return format
            score = confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0.0
            
            return {
                'sentiment': sentiment,
                'score': float(score),
                'confidence': float(confidence),
                'model': 'rule-based',
                'positive_words': positive_score,
                'negative_words': negative_score
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
            # Remove user mentions
            text = re.sub(r'@\w+', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove excessive punctuation
            text = re.sub(r'\!{2,}', '!', text)
            text = re.sub(r'\?{2,}', '?', text)
            
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            'model_name': self.model_name,
            'model_type': 'transformer' if self.pipeline else 'rule-based',
            'device': device,
            'max_length': 512,
            'status': 'loaded' if self.pipeline else 'rule-based',
            'version': '2.0'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the analyzer is working properly"""
        test_text = "This is a great product!"
        try:
            result = self.analyze(test_text)
            return {
                'status': 'healthy',
                'model_loaded': self.pipeline is not None,
                'test_result': result
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e)
            }
    
    def __del__(self):
        """Cleanup executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Test function
def test_sentiment_analyzer():
    """Test the sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    
    # Health check
    health = analyzer.health_check()
    print(f"Health Check: {health}")
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible and awful.",
        "It's okay, nothing special.",
        "The weather is nice today.",
        "This doesn't work at all, very disappointing.",
        "Not bad, but could be better.",
        "",
        "Hi"
    ]
    
    print("\nTesting Sentiment Analyzer:")
    print("=" * 50)
    
    # Test single analysis
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: '{text}'")
        print(f"Result: {result}")
        print("-" * 40)
    
    # Test batch analysis
    print("\nBatch Analysis Results:")
    print("=" * 50)
    batch_results = analyzer.analyze_batch(test_texts)
    for i, (text, result) in enumerate(zip(test_texts, batch_results)):
        print(f"{i+1}. '{text}' -> {result['sentiment']} (conf: {result['confidence']:.2f})")

if __name__ == "__main__":
    test_sentiment_analyzer()