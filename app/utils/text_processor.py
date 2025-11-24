import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any
import logging
import string

logger = logging.getLogger(__name__)

class TextProcessor:
    """Advanced text processing and cleaning utilities"""
    
    def __init__(self):
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            logger.warning("NLTK downloads failed, some features may be limited")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.setup_special_patterns()
    
    def setup_special_patterns(self):
        """Setup regex patterns for special text processing"""
        # URL pattern
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Reddit-specific patterns
        self.reddit_mention_pattern = re.compile(r'/?u/[A-Za-z0-9_-]+')
        self.subreddit_pattern = re.compile(r'/r/[A-Za-z0-9_-]+')
        
        # Emoji patterns
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        # Special characters
        self.special_char_pattern = re.compile(r'[^\w\s\.\!\?,]')
    
    def clean_text(self, text: str, remove_stopwords: bool = False, 
                  lemmatize: bool = False, min_length: int = 3) -> str:
        """Clean and preprocess text for analysis"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove Reddit mentions and subreddits
        text = self.reddit_mention_pattern.sub('', text)
        text = self.subreddit_pattern.sub('', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub('', text)
        
        # Remove special characters but keep basic punctuation
        text = self.special_char_pattern.sub('', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= min_length]
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'mentions': [],
            'subreddits': [],
            'urls': [],
            'hashtags': []
        }
        
        # Extract Reddit mentions
        entities['mentions'] = self.reddit_mention_pattern.findall(text)
        
        # Extract subreddits
        entities['subreddits'] = self.subreddit_pattern.findall(text)
        
        # Extract URLs
        entities['urls'] = self.url_pattern.findall(text)
        
        # Extract hashtags (common in some subreddits)
        hashtag_pattern = re.compile(r'#\w+')
        entities['hashtags'] = hashtag_pattern.findall(text)
        
        return entities
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for text"""
        if not text:
            return {}
        
        # Basic metrics
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        num_sentences = len([s for s in sentences if s.strip()])
        num_words = len(words)
        num_chars = len(text)
        
        if num_sentences == 0 or num_words == 0:
            return {}
        
        # Average sentence length
        avg_sentence_length = num_words / num_sentences
        
        # Average word length
        avg_word_length = num_chars / num_words
        
        # Flesch Reading Ease (simplified)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / num_words))
        
        # Word complexity (percentage of long words)
        long_words = [word for word in words if len(word) > 6]
        word_complexity = len(long_words) / num_words
        
        return {
            'word_count': num_words,
            'sentence_count': num_sentences,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'flesch_score': round(flesch_score, 2),
            'word_complexity': round(word_complexity, 4),
            'readability_level': self._get_readability_level(flesch_score)
        }
    
    def _get_readability_level(self, flesch_score: float) -> str:
        """Get readability level from Flesch score"""
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def detect_language_features(self, text: str) -> Dict[str, Any]:
        """Detect language features in text"""
        features = {
            'has_questions': False,
            'has_exclamations': False,
            'has_emphasis': False,
            'tone_indicators': [],
            'sentiment_indicators': []
        }
        
        # Check for questions
        features['has_questions'] = '?' in text
        
        # Check for exclamations
        features['has_exclamations'] = '!' in text
        
        # Check for emphasis (multiple punctuation, caps)
        if '!!' in text or '??' in text or text.isupper():
            features['has_emphasis'] = True
        
        # Common tone indicators
        tone_indicators = ['/s', '/j', '/srs', '/hj', '/lh', '/gen', '/pos', '/neg']
        features['tone_indicators'] = [indicator for indicator in tone_indicators if indicator in text.lower()]
        
        # Sentiment indicators
        positive_words = ['love', 'great', 'awesome', 'amazing', 'perfect', 'excellent', 'good', 'nice']
        negative_words = ['hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointing']
        
        text_lower = text.lower()
        features['sentiment_indicators'] = {
            'positive': [word for word in positive_words if word in text_lower],
            'negative': [word for word in negative_words if word in text_lower]
        }
        
        return features
    
    def batch_process(self, texts: List[str], **kwargs) -> List[str]:
        """Process multiple texts in batch"""
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        cleaned_text = self.clean_text(text)
        entities = self.extract_entities(text)
        readability = self.calculate_readability_metrics(cleaned_text)
        features = self.detect_language_features(text)
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'entities': entities,
            'readability': readability,
            'language_features': features,
            'processing_time': None  # Could be set by caller
        }

# Global text processor instance
text_processor = TextProcessor()