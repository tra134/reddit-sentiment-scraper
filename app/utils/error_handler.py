import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    pass

class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass

class ErrorHandler:
    """Comprehensive error handling and logging utility"""
    
    def __init__(self):
        self.error_counts = {}
        self.last_error_time = {}
        self.setup_error_categories()
    
    def setup_error_categories(self):
        """Setup error categories and their handling strategies"""
        self.error_categories = {
            'network': {
                'max_retries': 3,
                'retry_delay': 5,
                'severity': 'medium'
            },
            'parsing': {
                'max_retries': 2,
                'retry_delay': 2,
                'severity': 'low'
            },
            'authentication': {
                'max_retries': 1,
                'retry_delay': 10,
                'severity': 'high'
            },
            'rate_limit': {
                'max_retries': 1,
                'retry_delay': 60,
                'severity': 'medium'
            },
            'resource': {
                'max_retries': 2,
                'retry_delay': 30,
                'severity': 'high'
            },
            'unknown': {
                'max_retries': 1,
                'retry_delay': 10,
                'severity': 'high'
            }
        }
    
    def handle_error(self, error: Exception, context: str = "", 
                    category: str = "unknown") -> Dict[str, Any]:
        """Handle an error and return recovery information"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self._update_error_stats(error_type, category)
        
        # Get error category configuration
        category_config = self.error_categories.get(category, self.error_categories['unknown'])
        
        # Log the error
        self._log_error(error, context, category)
        
        # Determine recovery action
        recovery_info = self._get_recovery_action(error_type, category, category_config)
        
        return {
            'error_type': error_type,
            'error_message': error_message,
            'category': category,
            'context': context,
            'recovery_action': recovery_info['action'],
            'retry_after': recovery_info['retry_after'],
            'can_retry': recovery_info['can_retry'],
            'should_escalate': recovery_info['should_escalate'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_error_stats(self, error_type: str, category: str):
        """Update error statistics"""
        key = f"{category}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.last_error_time[key] = datetime.now()
    
    def _log_error(self, error: Exception, context: str, category: str):
        """Log error with appropriate level"""
        error_type = type(error).__name__
        
        log_message = f"{category.upper()} error in {context}: {error_type} - {str(error)}"
        
        if category in ['authentication', 'resource']:
            logger.error(log_message)
            logger.debug(f"Stack trace: {traceback.format_exc()}")
        elif category in ['rate_limit', 'network']:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _get_recovery_action(self, error_type: str, category: str, 
                           category_config: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the appropriate recovery action"""
        key = f"{category}_{error_type}"
        error_count = self.error_counts.get(key, 0)
        max_retries = category_config['max_retries']
        
        if error_count <= max_retries:
            return {
                'action': 'retry',
                'retry_after': category_config['retry_delay'],
                'can_retry': True,
                'should_escalate': False
            }
        else:
            return {
                'action': 'stop',
                'retry_after': 0,
                'can_retry': False,
                'should_escalate': True
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(self.error_counts.values())
        
        # Group by category
        category_counts = {}
        for key, count in self.error_counts.items():
            category = key.split('_')[0]
            category_counts[category] = category_counts.get(category, 0) + count
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts,
            'category_counts': category_counts,
            'last_error_times': self.last_error_time
        }
    
    def reset_error_counts(self, category: str = None):
        """Reset error counts"""
        if category:
            # Reset specific category
            keys_to_remove = [key for key in self.error_counts.keys() if key.startswith(category)]
            for key in keys_to_remove:
                del self.error_counts[key]
                if key in self.last_error_time:
                    del self.last_error_time[key]
        else:
            # Reset all
            self.error_counts.clear()
            self.last_error_time.clear()
        
        logger.info(f"Error counts reset for category: {category or 'all'}")
    
    def create_error_report(self) -> Dict[str, Any]:
        """Create a comprehensive error report"""
        stats = self.get_error_stats()
        
        report = {
            'summary': {
                'total_errors': stats['total_errors'],
                'most_common_category': max(stats['category_counts'].items(), key=lambda x: x[1])[0] if stats['category_counts'] else 'none',
                'report_generated': datetime.now().isoformat()
            },
            'detailed_stats': stats,
            'recommendations': self._generate_recommendations(stats)
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        if stats['category_counts'].get('rate_limit', 0) > 5:
            recommendations.append("Consider increasing request delays or using proxies")
        
        if stats['category_counts'].get('network', 0) > 10:
            recommendations.append("Check network stability and consider implementing retry logic")
        
        if stats['category_counts'].get('authentication', 0) > 0:
            recommendations.append("Verify API credentials and authentication tokens")
        
        if stats['category_counts'].get('resource', 0) > 3:
            recommendations.append("Monitor system resources and consider optimizing memory usage")
        
        if not recommendations:
            recommendations.append("No specific recommendations - error levels are normal")
        
        return recommendations

# Global error handler instance
error_handler = ErrorHandler()