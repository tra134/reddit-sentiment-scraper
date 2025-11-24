import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional
import logging
import os
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)

class ModelManager:
    """Manager for ML models with caching and version control"""
    
    def __init__(self, cache_dir: str = "data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.model_configs = {}
        self.setup_default_models()
    
    def setup_default_models(self):
        """Setup default model configurations"""
        self.model_configs = {
            'sentiment_roberta': {
                'name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'type': 'sentiment',
                'task': 'sentiment-analysis',
                'max_length': 512,
                'cache_key': 'sentiment_v1'
            },
            'emotion_roberta': {
                'name': 'j-hartmann/emotion-english-distilroberta-base',
                'type': 'emotion',
                'task': 'text-classification',
                'max_length': 512,
                'cache_key': 'emotion_v1'
            },
            'vader': {
                'name': 'vader',
                'type': 'sentiment',
                'task': 'rule-based',
                'cache_key': 'vader'
            }
        }
    
    def get_model(self, model_key: str, force_reload: bool = False) -> Any:
        """Get a model instance, loading if necessary"""
        if model_key in self.loaded_models and not force_reload:
            return self.loaded_models[model_key]
        
        model_config = self.model_configs.get(model_key)
        if not model_config:
            raise ValueError(f"Unknown model key: {model_key}")
        
        # Check cache first
        cached_model = self._load_from_cache(model_config['cache_key'])
        if cached_model and not force_reload:
            self.loaded_models[model_key] = cached_model
            return cached_model
        
        # Load new model
        model = self._load_model(model_config)
        
        # Cache the model
        self._save_to_cache(model_config['cache_key'], model)
        self.loaded_models[model_key] = model
        
        logger.info(f"✅ Loaded model: {model_key}")
        return model
    
    def _load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model based on configuration"""
        model_name = model_config['name']
        model_type = model_config['type']
        
        try:
            if model_name == 'vader':
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                return SentimentIntensityAnalyzer()
            
            elif model_type == 'sentiment':
                return pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    max_length=model_config.get('max_length', 512),
                    device=0 if torch.cuda.is_available() else -1
                )
            
            elif model_type == 'emotion':
                return pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    top_k=None,
                    max_length=model_config.get('max_length', 512),
                    device=0 if torch.cuda.is_available() else -1
                )
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load model from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, model: Any):
        """Save model to cache"""
        try:
            import pickle
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Failed to save model to cache: {e}")
    
    def unload_model(self, model_key: str):
        """Unload a model from memory"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"🗑️ Unloaded model: {model_key}")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.loaded_models.clear()
        logger.info("🗑️ Unloaded all models")
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_key not in self.model_configs:
            return {}
        
        config = self.model_configs[model_key]
        model = self.loaded_models.get(model_key)
        
        info = {
            'key': model_key,
            'name': config['name'],
            'type': config['type'],
            'task': config['task'],
            'loaded': model_key in self.loaded_models,
            'cache_key': config['cache_key']
        }
        
        if model:
            if hasattr(model, 'model'):
                info['model_class'] = model.model.__class__.__name__
            info['device'] = getattr(model, 'device', 'cpu')
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get information about all models"""
        models_info = {}
        for model_key in self.model_configs:
            models_info[model_key] = self.get_model_info(model_key)
        
        return {
            'total_models': len(self.model_configs),
            'loaded_models': len(self.loaded_models),
            'models': models_info
        }
    
    def add_custom_model(self, model_key: str, model_config: Dict[str, Any]):
        """Add a custom model configuration"""
        if model_key in self.model_configs:
            logger.warning(f"Model key {model_key} already exists, overwriting")
        
        self.model_configs[model_key] = model_config
        logger.info(f"➕ Added custom model: {model_key}")
    
    def clear_cache(self):
        """Clear model cache"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            cache_file.unlink()
        
        logger.info(f"🧹 Cleared {len(cache_files)} cached models")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cached_models': len(cache_files),
            'total_cache_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.cache_dir)
        }

# Global model manager instance
model_manager = ModelManager()