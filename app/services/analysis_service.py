import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import re
import sys
import os
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import Config
from app.ml.sentiment_analyzer import SentimentAnalyzer
from app.ml.aspect_analyzer import AspectAnalyzer
from app.ml.emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)

class AnalysisService:
    """Advanced analysis service with multiple ML models"""
    
    def __init__(self):
        self.config = Config.ANALYSIS
        self.ml_config = Config.ML_MODELS
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_analyzer = AspectAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Initialize models
        self._initialize_models()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("✅ Analysis service initialized")
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Load spaCy for NLP
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()
            
            logger.info("✅ ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise
    
    async def analyze_comments_batch(self, comments: List[Dict], 
                                   analysis_types: List[str] = None) -> List[Dict]:
        """Analyze comments in batches with multiple analysis types"""
        
        if analysis_types is None:
            analysis_types = ['sentiment', 'emotion', 'aspects']
        
        results = []
        batch_size = self.config['batch_size']
        
        # Process in batches
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            batch_results = await self._process_batch(batch, analysis_types)
            results.extend(batch_results)
            
            logger.info(f"📊 Processed batch {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
        
        return results
    
    async def _process_batch(self, batch: List[Dict], analysis_types: List[str]) -> List[Dict]:
        """Process a single batch of comments"""
        tasks = []
        
        for comment in batch:
            task = self._analyze_single_comment(comment, analysis_types)
            tasks.append(task)
        
        # Run analyses in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Analysis error: {result}")
                continue
            processed_results.append(result)
        
        return processed_results
    
    async def _analyze_single_comment(self, comment: Dict, analysis_types: List[str]) -> Dict:
        """Analyze a single comment with specified analysis types"""
        
        text = comment.get('body', '')
        if not text or len(text.strip()) < self.config['min_comment_length']:
            return self._create_empty_analysis(comment)
        
        # Limit text length
        if len(text) > self.config['max_text_length']:
            text = text[:self.config['max_text_length']]
        
        analysis_result = {
            'comment_id': comment.get('comment_id'),
            'original_comment': comment,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Run analyses based on requested types
        if 'sentiment' in analysis_types:
            sentiment_result = await self._analyze_sentiment(text)
            analysis_result.update(sentiment_result)
        
        if 'emotion' in analysis_types:
            emotion_result = await self._analyze_emotion(text)
            analysis_result.update(emotion_result)
        
        if 'aspects' in analysis_types:
            aspect_result = await self._analyze_aspects(text)
            analysis_result.update(aspect_result)
        
        if 'metadata' in analysis_types:
            metadata_result = self._analyze_metadata(comment)
            analysis_result.update(metadata_result)
        
        return analysis_result
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple models"""
        try:
            # Use multiple models for consensus
            vader_result = self._vader_analysis(text)
            textblob_result = self._textblob_analysis(text)
            roberta_result = await self.sentiment_analyzer.analyze(text)
            
            # Combine results
            combined_sentiment = self._combine_sentiment_results(
                vader_result, textblob_result, roberta_result
            )
            
            return {
                'sentiment_analysis': {
                    'vader': vader_result,
                    'textblob': textblob_result,
                    'roberta': roberta_result,
                    'combined': combined_sentiment
                },
                'sentiment_label': combined_sentiment['label'],
                'sentiment_score': combined_sentiment['score'],
                'confidence': combined_sentiment['confidence']
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._create_default_sentiment()
    
    def _vader_analysis(self, text: str) -> Dict[str, Any]:
        """VADER sentiment analysis"""
        scores = self.vader.polarity_scores(text)
        
        # Determine label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'label': label,
            'score': compound,
            'scores': scores,
            'model': 'vader'
        }
    
    def _textblob_analysis(self, text: str) -> Dict[str, Any]:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine label
        if polarity > 0:
            label = 'positive'
        elif polarity < 0:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'label': label,
            'score': polarity,
            'subjectivity': subjectivity,
            'model': 'textblob'
        }
    
    def _combine_sentiment_results(self, vader: Dict, textblob: Dict, roberta: Dict) -> Dict[str, Any]:
        """Combine results from multiple sentiment models"""
        models = [vader, textblob, roberta]
        
        # Count labels
        label_counts = Counter(model['label'] for model in models)
        most_common_label = label_counts.most_common(1)[0][0]
        
        # Average scores
        scores = [model.get('score', 0) for model in models]
        avg_score = sum(scores) / len(scores)
        
        # Confidence based on agreement
        confidence = label_counts[most_common_label] / len(models)
        
        return {
            'label': most_common_label,
            'score': avg_score,
            'confidence': confidence,
            'agreement': dict(label_counts)
        }
    
    async def _analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotions in text"""
        try:
            emotion_result = await self.emotion_analyzer.analyze(text)
            
            return {
                'emotion_analysis': emotion_result,
                'emotion_label': emotion_result.get('dominant_emotion'),
                'emotion_scores': emotion_result.get('scores', {})
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {
                'emotion_analysis': {},
                'emotion_label': 'neutral',
                'emotion_scores': {}
            }
    
    async def _analyze_aspects(self, text: str) -> Dict[str, Any]:
        """Extract and analyze aspects"""
        try:
            aspect_result = await self.aspect_analyzer.analyze(text)
            
            return {
                'aspect_analysis': aspect_result,
                'aspects': aspect_result.get('aspects', [])
            }
            
        except Exception as e:
            logger.error(f"Aspect analysis error: {e}")
            return {
                'aspect_analysis': {},
                'aspects': []
            }
    
    def _analyze_metadata(self, comment: Dict) -> Dict[str, Any]:
        """Analyze comment metadata"""
        text = comment.get('body', '')
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Readability metrics (simplified)
        avg_word_length = char_count / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Engagement metrics
        score = comment.get('score', 0)
        created_time = comment.get('created_utc')
        
        return {
            'metadata': {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'avg_word_length': round(avg_word_length, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'score': score,
                'has_author': comment.get('author') not in ['[deleted]', None]
            }
        }
    
    def _create_empty_analysis(self, comment: Dict) -> Dict:
        """Create empty analysis for invalid comments"""
        return {
            'comment_id': comment.get('comment_id'),
            'original_comment': comment,
            'analysis_timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {},
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'emotion_analysis': {},
            'emotion_label': 'neutral',
            'emotion_scores': {},
            'aspect_analysis': {},
            'aspects': [],
            'metadata': {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'score': comment.get('score', 0),
                'has_author': False
            }
        }
    
    def _create_default_sentiment(self) -> Dict[str, Any]:
        """Create default sentiment analysis result"""
        return {
            'sentiment_analysis': {},
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0
        }
    
    def generate_post_summary(self, post_data: Dict, analyzed_comments: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive post analysis summary"""
        
        if not analyzed_comments:
            return self._create_empty_summary(post_data)
        
        # Sentiment distribution
        sentiment_counts = Counter(
            comment.get('sentiment_label', 'neutral') 
            for comment in analyzed_comments
        )
        
        total_comments = len(analyzed_comments)
        sentiment_distribution = {
            sentiment: {
                'count': count,
                'percentage': round(count / total_comments * 100, 2)
            }
            for sentiment, count in sentiment_counts.items()
        }
        
        # Emotion distribution
        emotion_counts = Counter(
            comment.get('emotion_label', 'neutral')
            for comment in analyzed_comments
        )
        
        # Aspect summary
        all_aspects = []
        for comment in analyzed_comments:
            all_aspects.extend(comment.get('aspects', []))
        
        aspect_summary = self._summarize_aspects(all_aspects)
        
        # Engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(analyzed_comments)
        
        # Top comments by sentiment
        top_positive = self._get_top_comments_by_sentiment(analyzed_comments, 'positive')
        top_negative = self._get_top_comments_by_sentiment(analyzed_comments, 'negative')
        
        return {
            'post_id': post_data.get('post_id'),
            'summary_timestamp': datetime.now().isoformat(),
            'total_comments_analyzed': total_comments,
            'sentiment_summary': {
                'distribution': sentiment_distribution,
                'overall_sentiment': self._calculate_overall_sentiment(analyzed_comments),
                'average_sentiment_score': self._calculate_average_sentiment_score(analyzed_comments)
            },
            'emotion_summary': {
                'distribution': dict(emotion_counts),
                'dominant_emotion': emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'
            },
            'aspect_summary': aspect_summary,
            'engagement_metrics': engagement_metrics,
            'top_comments': {
                'positive': top_positive[:5],
                'negative': top_negative[:5]
            },
            'analysis_metadata': {
                'analysis_duration': None,  # Would be set by caller
                'models_used': list(self.ml_config.keys())
            }
        }
    
    def _summarize_aspects(self, aspects: List[Dict]) -> Dict[str, Any]:
        """Summarize aspect analysis results"""
        aspect_sentiments = defaultdict(list)
        
        for aspect in aspects:
            aspect_name = aspect.get('aspect')
            sentiment = aspect.get('sentiment', {}).get('label', 'neutral')
            aspect_sentiments[aspect_name].append(sentiment)
        
        summary = {}
        for aspect, sentiments in aspect_sentiments.items():
            sentiment_counts = Counter(sentiments)
            total = len(sentiments)
            
            summary[aspect] = {
                'total_mentions': total,
                'sentiment_distribution': {
                    sentiment: {
                        'count': count,
                        'percentage': round(count / total * 100, 2)
                    }
                    for sentiment, count in sentiment_counts.items()
                },
                'dominant_sentiment': sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutral'
            }
        
        return summary
    
    def _calculate_engagement_metrics(self, comments: List[Dict]) -> Dict[str, Any]:
        """Calculate engagement metrics"""
        scores = [comment.get('original_comment', {}).get('score', 0) for comment in comments]
        word_counts = [comment.get('metadata', {}).get('word_count', 0) for comment in comments]
        
        return {
            'average_score': round(np.mean(scores), 2) if scores else 0,
            'total_engagement': sum(scores),
            'average_word_count': round(np.mean(word_counts), 2) if word_counts else 0,
            'most_engaged_comment': max(scores) if scores else 0
        }
    
    def _get_top_comments_by_sentiment(self, comments: List[Dict], sentiment: str, limit: int = 5) -> List[Dict]:
        """Get top comments by sentiment intensity"""
        filtered_comments = [
            comment for comment in comments 
            if comment.get('sentiment_label') == sentiment
        ]
        
        # Sort by sentiment score (absolute value for negative)
        if sentiment == 'negative':
            filtered_comments.sort(key=lambda x: abs(x.get('sentiment_score', 0)), reverse=True)
        else:
            filtered_comments.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
        
        return filtered_comments[:limit]
    
    def _calculate_overall_sentiment(self, comments: List[Dict]) -> str:
        """Calculate overall sentiment for the post"""
        sentiment_counts = Counter(
            comment.get('sentiment_label', 'neutral') 
            for comment in comments
        )
        
        if not sentiment_counts:
            return 'neutral'
        
        return sentiment_counts.most_common(1)[0][0]
    
    def _calculate_average_sentiment_score(self, comments: List[Dict]) -> float:
        """Calculate average sentiment score"""
        scores = [comment.get('sentiment_score', 0) for comment in comments]
        return round(np.mean(scores), 4) if scores else 0.0
    
    def _create_empty_summary(self, post_data: Dict) -> Dict[str, Any]:
        """Create empty summary for posts with no comments"""
        return {
            'post_id': post_data.get('post_id'),
            'summary_timestamp': datetime.now().isoformat(),
            'total_comments_analyzed': 0,
            'sentiment_summary': {
                'distribution': {},
                'overall_sentiment': 'neutral',
                'average_sentiment_score': 0.0
            },
            'emotion_summary': {
                'distribution': {},
                'dominant_emotion': 'neutral'
            },
            'aspect_summary': {},
            'engagement_metrics': {
                'average_score': 0,
                'total_engagement': 0,
                'average_word_count': 0,
                'most_engaged_comment': 0
            },
            'top_comments': {
                'positive': [],
                'negative': []
            },
            'analysis_metadata': {
                'analysis_duration': 0,
                'models_used': []
            }
        }
        
    
    def summarize_with_gemini(self, text: str, language: str = "vi", max_length: int = 200) -> Dict[str, Any]:
        """
        Summarize text using Gemini API
        
        Args:
            text: Text to summarize
            language: Output language ('vi' for Vietnamese, 'en' for English)
            max_length: Maximum summary length in characters
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not self.gemini_available:
            return self._fallback_summarize(text, max_length)
        
        if not text or len(text.strip()) < 50:
            return {
                "summary": text[:max_length] if text else "Không có nội dung để tóm tắt",
                "method": "no_content",
                "confidence": 0.0
            }
        
        try:
            # Prepare prompt based on language
            if language == "vi":
                prompt = f"""
                Hãy tóm tắt đoạn văn bản sau đây bằng tiếng Việt:
                
                {text}
                
                Yêu cầu:
                1. Tóm tắt trong vòng {max_length} ký tự
                2. Giữ lại ý chính và thông tin quan trọng
                3. Viết bằng văn phong tự nhiên, dễ hiểu
                4. Nếu có số liệu hoặc thông tin cụ thể, hãy giữ lại
                5. Tập trung vào thông điệp chính
                
                Tóm tắt:
                """
            else:
                prompt = f"""
                Please summarize the following text in English:
                
                {text}
                
                Requirements:
                1. Summarize within {max_length} characters
                2. Keep main ideas and important information
                3. Write in natural, easy-to-understand language
                4. Keep specific data or important details
                5. Focus on the main message
                
                Summary:
                """
            
            # Prepare request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 1,
                    "topP": 0.95,
                    "maxOutputTokens": 500,
                }
            }
            
            headers = {"Content-Type": "application/json"}
            
            # Make API request
            response = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract summary from response
                if "candidates" in result and len(result["candidates"]) > 0:
                    summary = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    summary = self._clean_summary(summary, max_length)
                    
                    return {
                        "summary": summary,
                        "method": "gemini",
                        "confidence": 0.9,
                        "tokens_used": result.get("usageMetadata", {}).get("totalTokenCount", 0)
                    }
                else:
                    logger.error("No candidates in Gemini response")
                    return self._fallback_summarize(text, max_length)
            
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return self._fallback_summarize(text, max_length)
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return self._fallback_summarize(text, max_length)

    def summarize_post_comments(self, post_title: str, comments: List[str], max_comments: int = 50) -> Dict[str, Any]:
        """Summarize a Reddit post and its comments"""
        comments_to_summarize = comments[:max_comments]
        
        all_text = f"Tiêu đề bài viết: {post_title}\n\nCác bình luận:\n"
        for i, comment in enumerate(comments_to_summarize, 1):
            all_text += f"{i}. {comment}\n"
        
        if self.gemini_available:
            return self.summarize_with_gemini(all_text, language="vi", max_length=300)
        else:
            return self._summarize_post_fallback(post_title, comments_to_summarize)

    def _fallback_summarize(self, text: str, max_length: int = 200) -> Dict[str, Any]:
        """Fallback text summarization when Gemini is not available"""
        if not text:
            return {"summary": "Không có nội dung để tóm tắt", "method": "fallback", "confidence": 0.0}
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            summary = text[:max_length]
        elif len(sentences) <= 3:
            summary = " ".join(sentences)[:max_length]
        else:
            middle_idx = len(sentences) // 2
            selected_sentences = [sentences[0], sentences[middle_idx], sentences[-1]]
            summary = " ".join(selected_sentences)[:max_length]
        
        if len(summary) < len(text):
            summary += "..."
            
        return {"summary": summary, "method": "fallback", "confidence": 0.5}

    def _summarize_post_fallback(self, post_title: str, comments: List[str]) -> Dict[str, Any]:
        """Fallback summarization for Reddit posts"""
        from collections import Counter
        sentiments = [self.analyze_text(c)["sentiment"] for c in comments]
        sentiment_counts = Counter(sentiments)
        total_comments = len(comments)
        
        if total_comments == 0:
            overall_sentiment = "Trung lập"
        else:
            pos = sum(sentiment_counts.get(s, 0) for s in ["Tích cực", "Rất tích cực"])
            neg = sum(sentiment_counts.get(s, 0) for s in ["Tiêu cực", "Rất tiêu cực"])
            overall_sentiment = "Tích cực" if pos > neg else ("Tiêu cực" if neg > pos else "Trung lập")
        
        summary = f"Bài viết '{post_title}' có {total_comments} bình luận. "
        if total_comments > 0:
            summary += f"Phản hồi chung: {overall_sentiment.lower()}. "
            if sentiment_counts:
                top_s, count = sentiment_counts.most_common(1)[0]
                summary += f"{(count/total_comments)*100:.0f}% bình luận mang cảm xúc {top_s.lower()}. "
        
        summary += "Nội dung tập trung vào các chủ đề thảo luận chính."
        return {"summary": summary, "method": "fallback_post", "confidence": 0.6}

    def _clean_summary(self, summary: str, max_length: int) -> str:
        """Clean and format the summary"""
        summary = re.sub(r'[\*\#\_]', '', summary)
        summary = re.sub(r'^\d+\.\s*', '', summary, flags=re.MULTILINE)
        
        if len(summary) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            truncated = ""
            for s in sentences:
                if len(truncated + s) < max_length - 3:
                    truncated += s + " "
                else:
                    break
            summary = truncated.strip() or summary[:max_length - 3]
            if not summary.endswith('.'): summary += "..."
            
        return summary.strip()