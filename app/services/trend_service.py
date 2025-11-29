"""
TrendAnalysisService - Improved Version
- Prophet for time-series forecasting
- BERTopic for topic modeling
- KeyBERT for keyword extraction

Improvements:
1. Better error handling and model management
2. More efficient data preprocessing
3. Improved topic modeling with Vietnamese support
4. Better forecasting with validation
5. Memory optimization
6. Configurable parameters

Usage:
    pip install prophet bertopic keybert sentence-transformers scikit-learn pandas numpy
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class TrendAnalysisService:

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        bertopic_kwargs: Optional[Dict[str, Any]] = None,
        prophet_kwargs: Optional[Dict[str, Any]] = None,
        keybert_kwargs: Optional[Dict[str, Any]] = None,
        min_posts_for_analysis: int = 5
    ):
        self.embedding_model = embedding_model
        self.min_posts_for_analysis = min_posts_for_analysis

        self._bertopic_kwargs = {
            'language': 'multilingual',
            'calculate_probabilities': False,
            'verbose': False,
            **(bertopic_kwargs or {})
        }

        self._prophet_kwargs = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            **(prophet_kwargs or {})
        }

        self._keybert_kwargs = {
            'diversity': 0.7,
            'use_mmr': True,
            'use_maxsum': False,
            **(keybert_kwargs or {})
        }

        self._topic_model = None
        self._kw_model = None
        self._prophet = None
        self._embedding_model = None
        self.stopwords = self._get_stopwords()

    def _get_stopwords(self) -> set:
        english_stopwords = {
            'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'is', 'are',
            'and', 'or', 'with', 'i', 'my', 'you', 'it', 'this', 'that', 'was',
            'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        vietnamese_stopwords = {
            'của', 'và', 'là', 'những', 'các', 'trong', 'khi', 'với', 'có', 'được',
            'cho', 'này', 'nào', 'đó', 'nên', 'theo', 'như', 'một', 'về', 'cũng',
            'vẫn', 'đã', 'sẽ', 'rất', 'vào', 'ra', 'lại', 'năm', 'tháng', 'ngày',
            'giờ', 'phút', 'giây', 'người', 'người ta', 'mình', 'tôi', 'ta', 'chúng ta',
            'bạn', 'các bạn', 'anh', 'chị', 'em', 'ông', 'bà', 'nó', 'họ', 'chúng nó'
        }
        return english_stopwords.union(vietnamese_stopwords)

    def _ensure_models(self):
        if self._topic_model is None or self._kw_model is None:
            try:
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer
                from keybert import KeyBERT
            except ImportError as e:
                logger.error("Required packages not installed: %s", e)
                raise ImportError("Please install bertopic, sentence-transformers, and keybert") from e

            try:
                self._embedding_model = SentenceTransformer(self.embedding_model)
                self._topic_model = BERTopic(
                    embedding_model=self._embedding_model,
                    **self._bertopic_kwargs
                )
                self._kw_model = KeyBERT(model=self._embedding_model)
            except Exception as e:
                logger.error("Model initialization failed: %s", e)
                raise

        if self._prophet is None:
            try:
                from prophet import Prophet
                self._prophet = Prophet(**self._prophet_kwargs)
            except Exception as e:
                logger.error("Prophet initialization failed: %s", e)
                raise

    def clear_models(self):
        self._topic_model = None
        self._kw_model = None
        self._prophet = None
        self._embedding_model = None

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text

    def _preprocess(self, posts_data: List[Dict[str, Any]]) -> pd.DataFrame:
        if not posts_data:
            return pd.DataFrame()
        df = pd.DataFrame(posts_data)
        if 'created_utc' in df.columns:
            df['datetime'] = self._to_datetime(df['created_utc'])
        elif 'timestamp' in df.columns:
            df['datetime'] = self._to_datetime(df['timestamp'])
        else:
            df['datetime'] = pd.Timestamp.now()
        if 'title' not in df.columns:
            df['title'] = df.get('selftext', '').apply(str)
        df['title_clean'] = df['title'].apply(self._clean_text)
        df['text_length'] = df['title_clean'].str.len()
        df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0).astype(int)
        df['comments_count'] = pd.to_numeric(
            df.get('comments_count', df.get('num_comments', 0)),
            errors='coerce'
        ).fillna(0).astype(int)
        df['engagement'] = df['score'] + df['comments_count'] * 2
        df = df.dropna(subset=['datetime']).reset_index(drop=True)
        df = df[df['text_length'] > 5].reset_index(drop=True)
        return df

    @staticmethod
    def _to_datetime(series: pd.Series) -> pd.Series:
        if np.issubdtype(series.dtype, np.number):
            return pd.to_datetime(series, unit='s', utc=True).dt.tz_convert(None)
        else:
            return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)

    def extract_topics(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        if len(df) < self.min_posts_for_analysis:
            return []
        self._ensure_models()
        titles = df['title_clean'].tolist()
        try:
            topics, probabilities = self._topic_model.fit_transform(titles)
            topic_info = self._topic_model.get_topic_info()
            topics_out = []
            for _, row in topic_info.head(top_n + 1).iterrows():
                topic_id = int(row['Topic'])
                if topic_id == -1:
                    continue
                topic_terms = self._topic_model.get_topic(topic_id)
                if topic_terms:
                    representative_terms = [term for term, score in topic_terms[:5]]
                    name = ', '.join(representative_terms)
                else:
                    representative_terms = []
                    name = f"Topic_{topic_id}"
                topics_out.append({
                    'topic_id': topic_id,
                    'name': name,
                    'frequency': int(row['Count']),
                    'representation': representative_terms,
                    'percentage': float(row['Count']) / len(df) * 100
                })
            return topics_out
        except Exception as e:
            logger.error("Topic extraction failed: %s", e)
            return []

    def extract_keywords(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        if len(df) < 3:
            return []
        self._ensure_models()
        valid_texts = [text for text in df['title_clean'].tolist() if len(text) > 10]
        if not valid_texts:
            return []
        text_blob = ". ".join(valid_texts)
        try:
            keywords = self._kw_model.extract_keywords(
                text_blob,
                top_n=top_n * 2,
                **self._keybert_kwargs
            )
            filtered_keywords = []
            for kw, score in keywords:
                kw_lower = kw.lower()
                if len(kw) > 2 and kw_lower not in self.stopwords and not kw_lower.isdigit():
                    filtered_keywords.append((kw, score))
            return [{'keyword': kw, 'score': float(score)} for kw, score in filtered_keywords[:top_n]]
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    def forecast_scores(self, df: pd.DataFrame, days: int = 7,
                        confidence_interval: float = 0.8) -> Dict[str, Any]:
        if len(df) < 3:
            return {
                'error': 'insufficient_data',
                'message': f'Cần ít nhất 3 data points, có {len(df)}',
                'required': 3,
                'found': len(df)
            }
        try:
            forecast_engine = RobustForecastEngine()
            result = forecast_engine.forecast_engagement(df, days)
            if result and result.get('forecast'):
                return {
                    'forecast': result['forecast'],
                    'trend_direction': result['trend_direction'],
                    'trend_slope': 0.0,
                    'last_actual_date': df['datetime'].max().strftime('%Y-%m-%d') if 'datetime' in df.columns else datetime.now().strftime('%Y-%m-%d'),
                    'last_actual_value': float(df['engagement'].iloc[-1]) if 'engagement' in df.columns else 0,
                    'data_points': {
                        'total': len(df),
                        'forecast_period': days
                    },
                    'confidence_interval': result.get('confidence', 'medium'),
                    'method_used': result.get('method', 'unknown')
                }
            else:
                return {
                    'error': 'forecast_failed',
                    'message': 'Tất cả methods forecasting đều thất bại',
                    'fallback_data': forecast_engine._fallback_forecast(df, days)
                }
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return {
                'error': 'forecast_failed',
                'reason': str(e),
                'fallback': 'Sử dụng giá trị trung bình làm ước lượng'
            }

    # ---------------------------- Full Analysis ----------------------------
    def analyze_subreddit(self, 
                         subreddit: str, 
                         posts_data: List[Dict[str, Any]], 
                         days: int = 7,
                         topic_top_n: int = 5, 
                         kw_top_n: int = 10) -> Dict[str, Any]:
        """Improved full analysis pipeline with better error handling."""
        
        start_time = datetime.now()
        logger.info("Starting analysis for r/%s with %d posts", subreddit, len(posts_data))

        df = self._preprocess(posts_data)
        if df.empty:
            return {
                'error': 'no_valid_data',
                'message': 'No valid posts data after preprocessing',
                'subreddit': subreddit
            }

        # Filter by period
        since = pd.Timestamp.now() - pd.Timedelta(days=days)
        df_period = df[df['datetime'] >= since]

        if len(df_period) < self.min_posts_for_analysis:
            return {
                'error': 'insufficient_recent_data',
                'message': f'Not enough recent posts for analysis',
                'subreddit': subreddit,
                'required': self.min_posts_for_analysis,
                'found': len(df_period),
                'period_days': days
            }

        # Basic metrics
        analysis = {
            'subreddit': subreddit,
            'analysis_period_days': days,
            'data_summary': {
                'total_posts_analyzed': len(df_period),
                'total_score': int(df_period['score'].sum()),
                'total_engagement': int(df_period['engagement'].sum()),
                'total_comments': int(df_period['comments_count'].sum()),
                'avg_score_per_post': float(df_period['score'].mean()),
                'avg_comments_per_post': float(df_period['comments_count'].mean()),
                'avg_engagement_per_post': float(df_period['engagement'].mean()),
            },
            'analysis_timestamp': datetime.now().isoformat()
        }

        # Parallelizable analyses
        try:
            analysis['top_topics'] = self.extract_topics(df_period, top_n=topic_top_n)
        except Exception as e:
            logger.error("Topic analysis failed: %s", e)
            analysis['top_topics'] = []

        try:
            analysis['top_keywords'] = self.extract_keywords(df_period, top_n=kw_top_n)
        except Exception as e:
            logger.error("Keyword analysis failed: %s", e)
            analysis['top_keywords'] = []

        # Temporal analysis
        try:
            df_period['hour'] = df_period['datetime'].dt.hour
            hourly_activity = df_period.groupby('hour').agg({
                'engagement': 'sum',
                'score': 'count'
            }).reset_index()
            hourly_activity.columns = ['hour', 'total_engagement', 'post_count']
            
            analysis['peak_hours'] = [
                {
                    'hour': int(row['hour']),
                    'total_engagement': int(row['total_engagement']),
                    'post_count': int(row['post_count'])
                }
                for _, row in hourly_activity.iterrows()
            ]
        except Exception as e:
            logger.error("Peak hours analysis failed: %s", e)
            analysis['peak_hours'] = []

        # Forecasting
        try:
            analysis['forecast'] = self.forecast_scores(df_period, days=min(7, days))
        except Exception as e:
            logger.error("Forecasting failed: %s", e)
            analysis['forecast'] = {'error': 'forecast_failed', 'reason': str(e)}

        # Calculate analysis duration
        analysis['analysis_duration_seconds'] = (datetime.now() - start_time).total_seconds()
        logger.info("Analysis completed in %.2f seconds", analysis['analysis_duration_seconds'])

        return analysis


# ---------------------------- Demo ----------------------------
if __name__ == '__main__':
    # Improved demo with realistic data
    sample_data = []
    base_time = datetime.now()
    
    # Generate sample data with trends
    for i in range(30):
        post_time = base_time - timedelta(days=29-i)
        # Simulate growing trend
        engagement_base = 10 + i
        sample_data.append({
            'created_utc': int(post_time.timestamp()),
            'title': f'Cách học Python hiệu quả cho người mới bắt đầu {i}',
            'score': engagement_base + np.random.randint(0, 10),
            'comments_count': engagement_base // 2 + np.random.randint(0, 5)
        })
    
    # Initialize service
    service = TrendAnalysisService()
    
    # Run analysis
    try:
        results = service.analyze_subreddit('python', sample_data, days=30)
        print("Analysis Results:")
        print(f"Subreddit: {results['subreddit']}")
        print(f"Posts analyzed: {results['data_summary']['total_posts_analyzed']}")
        print(f"Total engagement: {results['data_summary']['total_engagement']}")
        
        if results['top_topics']:
            print("\nTop Topics:")
            for topic in results['top_topics'][:3]:
                print(f"  - {topic['name']} (freq: {topic['frequency']})")
        
        if results['top_keywords']:
            print("\nTop Keywords:")
            for kw in results['top_keywords'][:5]:
                print(f"  - {kw['keyword']} (score: {kw['score']:.3f})")
        
        if 'forecast' in results and 'error' not in results['forecast']:
            print(f"\nTrend: {results['forecast']['trend_direction']}")
            print("Forecast:")
            for fc in results['forecast']['forecast']:
                print(f"  {fc['date']}: {fc['predicted_engagement']:.1f}")
    
    except Exception as e:
        print(f"Demo failed: {e}")
    
    finally:
        service.clear_models()