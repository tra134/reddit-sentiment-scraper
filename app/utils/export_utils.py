import pandas as pd
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ExportUtils:
    """Utilities for exporting data in various formats"""
    
    def __init__(self, export_dir: str = "data/exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """Export data to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"✅ Data exported to CSV: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            raise
    
    def export_to_json(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """Export data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.json"
        
        filepath = self.export_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"✅ Data exported to JSON: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            raise
    
    def export_to_excel(self, data: List[Dict[str, Any]], filename: str = None, 
                       sheet_name: str = "Data") -> str:
        """Export data to Excel file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.xlsx"
        
        filepath = self.export_dir / filename
        
        try:
            df = pd.DataFrame(data)
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"✅ Data exported to Excel: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"❌ Excel export failed: {e}")
            raise
    
    def export_analysis_report(self, analysis_data: Dict[str, Any], 
                             format: str = 'json') -> str:
        """Export comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"analysis_report_{timestamp}.json"
            return self.export_to_json([analysis_data], filename)
        elif format == 'csv':
            # Flatten analysis data for CSV
            flattened_data = self._flatten_analysis_data(analysis_data)
            filename = f"analysis_report_{timestamp}.csv"
            return self.export_to_csv(flattened_data, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _flatten_analysis_data(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested analysis data for CSV export"""
        flattened = []
        
        # Extract basic post info
        post_info = {
            'post_id': analysis_data.get('post_id'),
            'title': analysis_data.get('title', '')[:100],  # Truncate long titles
            'subreddit': analysis_data.get('subreddit'),
            'analysis_timestamp': analysis_data.get('analysis_timestamp')
        }
        
        # Add sentiment summary
        sentiment_summary = analysis_data.get('summary', {}).get('sentiment_summary', {})
        post_info.update({
            'total_comments': sentiment_summary.get('total_comments_analyzed', 0),
            'overall_sentiment': sentiment_summary.get('overall_sentiment', 'neutral'),
            'avg_sentiment_score': sentiment_summary.get('average_sentiment_score', 0)
        })
        
        flattened.append(post_info)
        return flattened
    
    def export_comments_with_analysis(self, comments: List[Dict[str, Any]], 
                                    format: str = 'csv') -> str:
        """Export comments with their analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for export
        export_data = []
        for comment in comments:
            original_comment = comment.get('original_comment', {})
            row = {
                'comment_id': original_comment.get('comment_id'),
                'author': original_comment.get('author'),
                'body': original_comment.get('body'),
                'score': original_comment.get('score'),
                'sentiment': comment.get('sentiment_label'),
                'sentiment_score': comment.get('sentiment_score'),
                'emotion': comment.get('emotion_label'),
                'confidence': comment.get('confidence', 0),
                'aspects': ', '.join([aspect['aspect'] for aspect in comment.get('aspects', [])]),
                'created_utc': original_comment.get('created_utc')
            }
            export_data.append(row)
        
        if format == 'csv':
            filename = f"comments_analysis_{timestamp}.csv"
            return self.export_to_csv(export_data, filename)
        elif format == 'json':
            filename = f"comments_analysis_{timestamp}.json"
            return self.export_to_json(export_data, filename)
        elif format == 'excel':
            filename = f"comments_analysis_{timestamp}.xlsx"
            return self.export_to_excel(export_data, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_readme_file(self, export_path: str, data_description: str = ""):
        """Create a README file for the exported data"""
        readme_path = Path(export_path).parent / "README.txt"
        
        readme_content = f"""
Reddit Sentiment Analysis Export
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Data Description:
{data_description}

File Information:
- File: {Path(export_path).name}
- Format: {Path(export_path).suffix.upper().replace('.', '')}
- Size: {self._get_file_size(export_path)}

Data Fields:
- comment_id: Unique identifier for each comment
- author: Reddit username of comment author
- body: The comment text content
- score: Reddit score (upvotes - downvotes)
- sentiment: Overall sentiment (positive/negative/neutral)
- sentiment_score: Numerical sentiment score (-1 to 1)
- emotion: Dominant emotion detected
- confidence: Model confidence in analysis
- aspects: Detected aspects/topics in comment
- created_utc: Comment creation timestamp

Notes:
- This data was generated using the Reddit Sentiment Analyzer Pro
- Sentiment analysis uses multiple ML models (RoBERTa, VADER, TextBlob)
- Emotion detection uses transformer-based models
- Aspect analysis identifies key topics mentioned in comments

For questions or support, please refer to the project documentation.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"📝 README file created: {readme_path}")
    
    def _get_file_size(self, filepath: str) -> str:
        """Get human-readable file size"""
        size_bytes = os.path.getsize(filepath)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} TB"
    
    def get_export_history(self) -> List[Dict[str, Any]]:
        """Get history of exported files"""
        exports = []
        
        for file_path in self.export_dir.glob("export_*.*"):
            stats = file_path.stat()
            exports.append({
                'filename': file_path.name,
                'filepath': str(file_path),
                'size': self._get_file_size(str(file_path)),
                'created': datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                'file_type': file_path.suffix.upper().replace('.', '')
            })
        
        return sorted(exports, key=lambda x: x['created'], reverse=True)
    
    def cleanup_old_exports(self, days_old: int = 30):
        """Clean up export files older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in self.export_dir.glob("export_*.*"):
            if file_path.stat().st_ctime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} export files older than {days_old} days")

# Global export utils instance
export_utils = ExportUtils()