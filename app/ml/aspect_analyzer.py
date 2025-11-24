import spacy
import re
from typing import Dict, Any, List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logger = logging.getLogger(__name__)

class AspectAnalyzer:
    """Aspect-based sentiment analysis"""
    
    def __init__(self):
        self.nlp = None
        self.aspect_keywords = {
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money', 'worth'],
            'quality': ['quality', 'durable', 'reliable', 'solid', 'premium', 'excellent', 'poor', 'bad'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient', 'powerful', 'lag', 'smooth'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly', 'sleek', 'modern'],
            'battery': ['battery', 'charge', 'power', 'life', 'endurance', 'drain', 'lasting'],
            'screen': ['screen', 'display', 'resolution', 'brightness', 'colors', 'size', 'inch'],
            'camera': ['camera', 'photo', 'picture', 'video', 'resolution', 'lens', 'zoom', 'quality'],
            'service': ['service', 'support', 'customer', 'help', 'warranty', 'return', 'refund'],
            'delivery': ['delivery', 'shipping', 'arrived', 'package', 'fast', 'slow', 'time'],
            'features': ['feature', 'function', 'option', 'setting', 'customization', 'ability']
        }
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model"""
        try:
            logger.info("🔄 Loading spaCy model for aspect analysis...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model: {e}")
            # Try to download if not available
            try:
                import os
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model downloaded and loaded")
            except:
                logger.error("❌ Could not load spaCy model, aspect analysis will be limited")
                self.nlp = None
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Extract aspects and their sentiments from text"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=2),
                self._analyze_sync,
                text
            )
            return result
        except Exception as e:
            logger.error(f"Aspect analysis error: {e}")
            return self._get_default_result()
    
    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous aspect analysis"""
        try:
            if not text or len(text.strip()) < 5:
                return self._get_default_result()
            
            aspects_found = []
            
            # Method 1: Keyword-based aspect extraction
            keyword_aspects = self._extract_aspects_by_keywords(text)
            aspects_found.extend(keyword_aspects)
            
            # Method 2: NLP-based aspect extraction (if spaCy is available)
            if self.nlp:
                nlp_aspects = self._extract_aspects_with_nlp(text)
                aspects_found.extend(nlp_aspects)
            
            # Remove duplicates and merge similar aspects
            unique_aspects = self._deduplicate_aspects(aspects_found)
            
            # Analyze sentiment for each aspect
            for aspect in unique_aspects:
                aspect['sentiment'] = self._analyze_aspect_sentiment(text, aspect['aspect'])
            
            return {
                'aspects': unique_aspects,
                'aspect_count': len(unique_aspects),
                'methods_used': ['keywords', 'nlp'] if self.nlp else ['keywords']
            }
            
        except Exception as e:
            logger.error(f"Sync aspect analysis error: {e}")
            return self._get_default_result()
    
    def _extract_aspects_by_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract aspects using keyword matching"""
        aspects = []
        text_lower = text.lower()
        
        for aspect_name, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the context around the keyword
                    context = self._get_keyword_context(text, keyword)
                    
                    aspects.append({
                        'aspect': aspect_name,
                        'keyword': keyword,
                        'context': context,
                        'method': 'keyword',
                        'confidence': 0.7
                    })
                    break  # Only need one keyword match per aspect
        
        return aspects
    
    def _extract_aspects_with_nlp(self, text: str) -> List[Dict[str, Any]]:
        """Extract aspects using spaCy NLP"""
        aspects = []
        
        try:
            doc = self.nlp(text)
            
            # Look for nouns and noun phrases that might be aspects
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                
                # Check if this noun phrase relates to any known aspect
                for aspect_name, keywords in self.aspect_keywords.items():
                    for keyword in keywords:
                        if keyword in chunk_text:
                            aspects.append({
                                'aspect': aspect_name,
                                'entity': chunk_text,
                                'context': chunk.sent.text if chunk.sent else text,
                                'method': 'nlp',
                                'confidence': 0.8
                            })
                            break
            
            # Also look for adjectives that might modify aspects
            for token in doc:
                if token.pos_ == 'ADJ':
                    # Check what this adjective might be modifying
                    for child in token.children:
                        if child.pos_ in ['NOUN', 'PROPN']:
                            child_text = child.text.lower()
                            for aspect_name, keywords in self.aspect_keywords.items():
                                for keyword in keywords:
                                    if keyword in child_text:
                                        aspects.append({
                                            'aspect': aspect_name,
                                            'entity': child_text,
                                            'adjective': token.text,
                                            'context': token.sent.text if token.sent else text,
                                            'method': 'nlp_adjective',
                                            'confidence': 0.75
                                        })
                                        break
            
        except Exception as e:
            logger.warning(f"NLP aspect extraction error: {e}")
        
        return aspects
    
    def _get_keyword_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Get context around a keyword"""
        try:
            index = text.lower().find(keyword)
            if index == -1:
                return text[:100]  # Return beginning if keyword not found
            
            start = max(0, index - window)
            end = min(len(text), index + len(keyword) + window)
            
            context = text[start:end]
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            return context
        except:
            return text[:100]
    
    def _analyze_aspect_sentiment(self, text: str, aspect: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific aspect"""
        # Simple rule-based sentiment for the aspect context
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'horrible', 'disappointing', 'worst']
        
        text_lower = text.lower()
        aspect_keywords = self.aspect_keywords.get(aspect, [])
        
        # Find sentences mentioning the aspect
        aspect_sentences = []
        for sentence in re.split(r'[.!?]+', text):
            if any(keyword in sentence.lower() for keyword in aspect_keywords):
                aspect_sentences.append(sentence)
        
        if not aspect_sentences:
            return {'label': 'neutral', 'score': 0.0}
        
        # Analyze sentiment in aspect-related sentences
        positive_count = sum(1 for word in positive_words if any(word in sent for sent in aspect_sentences))
        negative_count = sum(1 for word in negative_words if any(word in sent for sent in aspect_sentences))
        
        if positive_count > negative_count:
            return {'label': 'positive', 'score': min(positive_count / 3, 1.0)}
        elif negative_count > positive_count:
            return {'label': 'negative', 'score': -min(negative_count / 3, 1.0)}
        else:
            return {'label': 'neutral', 'score': 0.0}
    
    def _deduplicate_aspects(self, aspects: List[Dict]) -> List[Dict]:
        """Remove duplicate aspects"""
        seen = set()
        unique_aspects = []
        
        for aspect in aspects:
            aspect_key = (aspect['aspect'], aspect.get('keyword', ''), aspect.get('entity', ''))
            if aspect_key not in seen:
                seen.add(aspect_key)
                unique_aspects.append(aspect)
        
        return unique_aspects
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Get default aspect analysis result"""
        return {
            'aspects': [],
            'aspect_count': 0,
            'methods_used': []
        }
    
    def add_custom_aspect(self, aspect_name: str, keywords: List[str]):
        """Add custom aspect for analysis"""
        self.aspect_keywords[aspect_name] = keywords
        logger.info(f"➕ Added custom aspect: {aspect_name}")
    
    def get_available_aspects(self) -> List[str]:
        """Get list of available aspects for analysis"""
        return list(self.aspect_keywords.keys())