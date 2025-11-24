import pdfkit
from jinja2 import Template
import base64
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generator for comprehensive analysis reports"""
    
    def __init__(self, templates_dir: str = "app/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.setup_default_templates()
    
    def setup_default_templates(self):
        """Setup default report templates"""
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Reddit Sentiment Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }
                .metric-card { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #667eea; }
                .positive { border-left-color: #00D26A; }
                .negative { border-left-color: #FF4757; }
                .neutral { border-left-color: #FFB800; }
                .chart { text-align: center; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .footer { margin-top: 50px; text-align: center; color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reddit Sentiment Analysis Report</h1>
                <p>Generated on {{ generation_date }}</p>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    {% for metric in summary_metrics %}
                    <div class="metric-card {{ metric.class }}">
                        <strong>{{ metric.name }}</strong><br>
                        <span style="font-size: 1.5em;">{{ metric.value }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            {% if sentiment_analysis %}
            <div class="section">
                <h2>😊 Sentiment Analysis</h2>
                <div class="chart">
                    <!-- Sentiment chart would be embedded here -->
                    <p><em>Sentiment distribution chart</em></p>
                </div>
                <table>
                    <tr>
                        <th>Sentiment</th>
                        <th>Percentage</th>
                        <th>Count</th>
                    </tr>
                    {% for sentiment, data in sentiment_analysis.distribution.items() %}
                    <tr>
                        <td>{{ sentiment.title() }}</td>
                        <td>{{ data.percentage }}%</td>
                        <td>{{ data.count }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if emotion_analysis %}
            <div class="section">
                <h2>🎭 Emotion Analysis</h2>
                <table>
                    <tr>
                        <th>Emotion</th>
                        <th>Count</th>
                        <th>Dominant</th>
                    </tr>
                    {% for emotion, count in emotion_analysis.distribution.items() %}
                    <tr>
                        <td>{{ emotion.title() }}</td>
                        <td>{{ count }}</td>
                        <td>{% if emotion == emotion_analysis.dominant_emotion %}⭐{% endif %}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if aspect_analysis %}
            <div class="section">
                <h2>🔍 Aspect Analysis</h2>
                {% for aspect, data in aspect_analysis.items() %}
                <div style="margin-bottom: 20px;">
                    <h3>{{ aspect.title() }}</h3>
                    <p><strong>Total Mentions:</strong> {{ data.total_mentions }}</p>
                    <p><strong>Dominant Sentiment:</strong> {{ data.dominant_sentiment.title() }}</p>
                    <table>
                        <tr>
                            <th>Sentiment</th>
                            <th>Percentage</th>
                            <th>Count</th>
                        </tr>
                        {% for sentiment, s_data in data.sentiment_distribution.items() %}
                        <tr>
                            <td>{{ sentiment.title() }}</td>
                            <td>{{ s_data.percentage }}%</td>
                            <td>{{ s_data.count }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if top_comments %}
            <div class="section">
                <h2>💬 Top Comments</h2>
                <h3>Positive Comments</h3>
                {% for comment in top_comments.positive %}
                <div class="metric-card positive" style="margin-bottom: 15px;">
                    <strong>Score: {{ comment.original_comment.score }}</strong><br>
                    {{ comment.original_comment.body[:200] }}{% if comment.original_comment.body|length > 200 %}...{% endif %}
                </div>
                {% endfor %}
                
                <h3>Negative Comments</h3>
                {% for comment in top_comments.negative %}
                <div class="metric-card negative" style="margin-bottom: 15px;">
                    <strong>Score: {{ comment.original_comment.score }}</strong><br>
                    {{ comment.original_comment.body[:200] }}{% if comment.original_comment.body|length > 200 %}...{% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="footer">
                <p>Report generated by Reddit Sentiment Analyzer Pro</p>
                <p>Analysis completed: {{ analysis_timestamp }}</p>
            </div>
        </body>
        </html>
        """
    
    def generate_html_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate HTML report from analysis data"""
        template = Template(self.html_template)
        
        # Prepare data for template
        report_data = self._prepare_report_data(analysis_data)
        
        # Render template
        html_content = template.render(**report_data)
        
        return html_content
    
    def generate_pdf_report(self, analysis_data: Dict[str, Any], 
                          output_path: Optional[str] = None) -> str:
        """Generate PDF report from analysis data"""
        try:
            html_content = self.generate_html_report(analysis_data)
            
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/exports/report_{timestamp}.pdf"
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Configure PDF options
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            # Generate PDF
            pdfkit.from_string(html_content, output_path, options=options)
            
            logger.info(f"✅ PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ PDF report generation failed: {e}")
            raise
    
    def generate_text_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate text summary of analysis"""
        summary = []
        
        # Header
        summary.append("REDDIT SENTIMENT ANALYSIS REPORT")
        summary.append("=" * 50)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Basic info
        summary_data = analysis_data.get('summary', {})
        summary.append("SUMMARY")
        summary.append("-" * 20)
        summary.append(f"Total Comments Analyzed: {summary_data.get('total_comments_analyzed', 0)}")
        summary.append(f"Overall Sentiment: {summary_data.get('sentiment_summary', {}).get('overall_sentiment', 'N/A')}")
        summary.append(f"Average Sentiment Score: {summary_data.get('sentiment_summary', {}).get('average_sentiment_score', 0):.3f}")
        summary.append("")
        
        # Sentiment breakdown
        sentiment_data = summary_data.get('sentiment_summary', {}).get('distribution', {})
        if sentiment_data:
            summary.append("SENTIMENT DISTRIBUTION")
            summary.append("-" * 25)
            for sentiment, data in sentiment_data.items():
                summary.append(f"{sentiment.title()}: {data.get('percentage', 0)}% ({data.get('count', 0)} comments)")
            summary.append("")
        
        # Emotion breakdown
        emotion_data = summary_data.get('emotion_summary', {}).get('distribution', {})
        if emotion_data:
            summary.append("EMOTION DISTRIBUTION")
            summary.append("-" * 25)
            for emotion, count in emotion_data.items():
                summary.append(f"{emotion.title()}: {count} comments")
            summary.append("")
        
        # Key insights
        summary.append("KEY INSIGHTS")
        summary.append("-" * 15)
        insights = self._generate_insights(analysis_data)
        for i, insight in enumerate(insights, 1):
            summary.append(f"{i}. {insight}")
        
        return "\n".join(summary)
    
    def _prepare_report_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for report generation"""
        summary_data = analysis_data.get('summary', {})
        
        # Summary metrics
        summary_metrics = [
            {'name': 'Total Comments', 'value': summary_data.get('total_comments_analyzed', 0), 'class': ''},
            {'name': 'Overall Sentiment', 'value': summary_data.get('sentiment_summary', {}).get('overall_sentiment', 'N/A'), 'class': 'positive' if summary_data.get('sentiment_summary', {}).get('overall_sentiment') == 'positive' else 'negative' if summary_data.get('sentiment_summary', {}).get('overall_sentiment') == 'negative' else 'neutral'},
            {'name': 'Avg Sentiment', 'value': f"{summary_data.get('sentiment_summary', {}).get('average_sentiment_score', 0):.3f}", 'class': ''},
            {'name': 'Dominant Emotion', 'value': summary_data.get('emotion_summary', {}).get('dominant_emotion', 'N/A').title(), 'class': ''}
        ]
        
        return {
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_timestamp': analysis_data.get('analysis_timestamp', 'N/A'),
            'summary_metrics': summary_metrics,
            'sentiment_analysis': summary_data.get('sentiment_summary'),
            'emotion_analysis': summary_data.get('emotion_summary'),
            'aspect_analysis': summary_data.get('aspect_summary', {}),
            'top_comments': summary_data.get('top_comments', {}),
            'engagement_metrics': summary_data.get('engagement_metrics', {})
        }
    
    def _generate_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis data"""
        insights = []
        summary_data = analysis_data.get('summary', {})
        
        # Sentiment insights
        sentiment_data = summary_data.get('sentiment_summary', {})
        overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
        
        if overall_sentiment == 'positive':
            insights.append("The community response is generally positive")
        elif overall_sentiment == 'negative':
            insights.append("The community response shows significant concerns")
        else:
            insights.append("The community response is mixed or neutral")
        
        # Emotion insights
        emotion_data = summary_data.get('emotion_summary', {})
        dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
        
        if dominant_emotion != 'neutral':
            insights.append(f"The dominant emotion in discussions is {dominant_emotion}")
        
        # Engagement insights
        engagement_data = summary_data.get('engagement_metrics', {})
        avg_score = engagement_data.get('average_score', 0)
        
        if avg_score > 10:
            insights.append("Comments are receiving high engagement (upvotes)")
        elif avg_score < 0:
            insights.append("Comments are receiving more downvotes than upvotes")
        
        # Aspect insights
        aspect_data = summary_data.get('aspect_summary', {})
        if aspect_data:
            top_aspect = max(aspect_data.items(), key=lambda x: x[1]['total_mentions']) if aspect_data else None
            if top_aspect:
                insights.append(f"The most discussed topic is '{top_aspect[0]}' with {top_aspect[1]['total_mentions']} mentions")
        
        return insights
    
    def generate_comprehensive_report(self, analysis_data: Dict[str, Any], 
                                    formats: List[str] = ['html', 'pdf', 'txt']) -> Dict[str, str]:
        """Generate reports in multiple formats"""
        reports = {}
        
        for format in formats:
            try:
                if format == 'html':
                    reports['html'] = self.generate_html_report(analysis_data)
                elif format == 'pdf':
                    reports['pdf'] = self.generate_pdf_report(analysis_data)
                elif format == 'txt':
                    reports['txt'] = self.generate_text_summary(analysis_data)
            except Exception as e:
                logger.error(f"Failed to generate {format} report: {e}")
                reports[format] = f"Error: {e}"
        
        return reports

# Global report generator instance
report_generator = ReportGenerator()