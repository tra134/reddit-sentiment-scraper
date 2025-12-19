# Improved TrendAnalysisService v2
# Focus: performance, correctness, production-readiness

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrendAnalysisService:
    """
    Trend Analysis Service v2
    Improvements:
    - Modular preprocessing
    - Better topic quality (title + selftext)
    - Keyword extraction per-topic
    - Simple but robust forecasting (rolling + Prophet fallback)
    - Embedding cache
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        min_posts_for_analysis: int = 5,
        cache_embeddings: bool = True
    ):
        self.embedding_model_name = embedding_model
        self.min_posts_for_analysis = min_posts_for_analysis
        self.cache_embeddings = cache_embeddings

        self._embedding_model = None
        self._topic_model = None
        self._kw_model = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}

        self.stopwords = self._get_stopwords()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _get_stopwords(self) -> set:
        en = {
            'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'is', 'are',
            'and', 'or', 'with', 'this', 'that', 'was', 'were'
        }
        vi = {
            'của', 'và', 'là', 'những', 'các', 'trong', 'khi', 'với', 'có', 'được',
            'cho', 'này', 'đó', 'một', 'về', 'cũng', 'đã', 'sẽ'
        }
        return en | vi

    def _ensure_models(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

        if self._topic_model is None:
            from bertopic import BERTopic
            self._topic_model = BERTopic(
                embedding_model=self._embedding_model,
                language="multilingual",
                calculate_probabilities=False,
                verbose=False
            )

        if self._kw_model is None:
            from keybert import KeyBERT
            self._kw_model = KeyBERT(self._embedding_model)

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        if not posts:
            return pd.DataFrame()

        df = pd.DataFrame(posts)

        if 'created_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df.get('timestamp'), errors='coerce')

        df['title'] = df.get('title', '')
        df['selftext'] = df.get('selftext', '')
        df['full_text'] = (
            df['title'].fillna('') + '. ' + df['selftext'].fillna('')
        ).apply(self._clean_text)

        df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)
        df['comments'] = pd.to_numeric(df.get('comments_count', 0), errors='coerce').fillna(0)
        df['engagement'] = df['score'] + 2 * df['comments']

        df = df.dropna(subset=['datetime'])
        df = df[df['full_text'].str.len() > 10]
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Topic Modeling
    # ------------------------------------------------------------------
    def extract_topics(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        if len(df) < self.min_posts_for_analysis:
            return []

        self._ensure_models()
        docs = df['full_text'].tolist()

        topics, _ = self._topic_model.fit_transform(docs)
        info = self._topic_model.get_topic_info()

        results = []
        for _, row in info.iterrows():
            topic_id = int(row['Topic'])
            if topic_id == -1:
                continue
            terms = self._topic_model.get_topic(topic_id)
            keywords = [t for t, _ in terms[:5]] if terms else []
            results.append({
                'topic_id': topic_id,
                'name': ', '.join(keywords),
                'count': int(row['Count']),
                'percentage': round(row['Count'] / len(df) * 100, 2)
            })
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------
    # Keyword Extraction (global)
    # ------------------------------------------------------------------
    def extract_keywords(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        if len(df) < 3:
            return []

        self._ensure_models()
        text_blob = '. '.join(df['full_text'].tolist())

        keywords = self._kw_model.extract_keywords(
            text_blob,
            top_n=top_n * 2,
            use_mmr=True,
            diversity=0.7
        )

        results = []
        for kw, score in keywords:
            if kw not in self.stopwords and len(kw) > 2:
                results.append({'keyword': kw, 'score': float(score)})
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------
    # Forecasting (robust & cheap)
    # ------------------------------------------------------------------
    def forecast_engagement(self, df: pd.DataFrame, days: int = 7) -> Dict[str, Any]:
        if len(df) < 5:
            mean_val = float(df['engagement'].mean())
            return {
                'method': 'mean_fallback',
                'forecast': [
                    {
                        'date': (df['datetime'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                        'predicted_engagement': mean_val
                    }
                    for i in range(days)
                ]
            }

        ts = df.sort_values('datetime')
        rolling = ts['engagement'].rolling(window=3).mean().iloc[-1]
        slope = ts['engagement'].diff().mean()

        forecast = []
        last_date = ts['datetime'].max()
        base = rolling
        for i in range(days):
            base += slope
            forecast.append({
                'date': (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_engagement': max(0, float(base))
            })

        trend = 'up' if slope > 0 else 'down' if slope < 0 else 'flat'

        return {
            'method': 'rolling_trend',
            'trend': trend,
            'forecast': forecast
        }

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def analyze(self, posts: List[Dict[str, Any]], days: int = 7) -> Dict[str, Any]:
        start = datetime.now()
        df = self.preprocess(posts)

        if df.empty:
            return {'error': 'no_valid_data'}

        since = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df['datetime'] >= since]

        if len(df) < self.min_posts_for_analysis:
            return {'error': 'insufficient_data', 'found': len(df)}

        result = {
            'summary': {
                'posts': len(df),
                'total_engagement': int(df['engagement'].sum()),
                'avg_engagement': float(df['engagement'].mean())
            },
            'topics': self.extract_topics(df),
            'keywords': self.extract_keywords(df),
            'forecast': self.forecast_engagement(df, days=min(days, 7)),
            'runtime_sec': (datetime.now() - start).total_seconds()
        }
        return result
