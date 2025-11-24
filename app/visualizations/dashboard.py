import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .charts import chart_generator

logger = logging.getLogger(__name__)

class DashboardComponents:
    """Reusable dashboard components for Streamlit"""
    
    def __init__(self):
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom CSS styles for the dashboard"""
        st.markdown("""
        <style>
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        .positive-metric {
            border-left-color: #00D26A;
        }
        .negative-metric {
            border-left-color: #FF4757;
        }
        .neutral-metric {
            border-left-color: #FFB800;
        }
        .analysis-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self, title: str, subtitle: str = ""):
        """Render dashboard header"""
        st.markdown(f"""
        <div class="dashboard-header">
            <h1 style="margin:0; font-size: 2.5rem;">{title}</h1>
            <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics_row(self, metrics: Dict[str, Any]):
        """Render a row of metric cards"""
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                self.render_metric_card(metric_name, metric_value)
    
    def render_metric_card(self, title: str, value: Any, delta: Optional[str] = None):
        """Render a single metric card"""
        # Determine card style based on value
        card_class = "metric-card"
        if isinstance(value, (int, float)):
            if value > 0:
                card_class += " positive-metric"
            elif value < 0:
                card_class += " negative-metric"
            else:
                card_class += " neutral-metric"
        
        delta_html = f"<div style='font-size: 0.9rem; color: #666;'>{delta}</div>" if delta else ""
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;'>{title}</div>
            <div style='font-size: 1.8rem; font-weight: bold; color: #333;'>{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def render_sentiment_analysis(self, analysis_data: Dict[str, Any]):
        """Render sentiment analysis section"""
        st.markdown("### 📊 Sentiment Analysis")
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sentiment distribution chart
                sentiment_data = analysis_data.get('sentiment_summary', {}).get('distribution', {})
                if sentiment_data:
                    fig = chart_generator.create_sentiment_pie_chart(sentiment_data)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Key metrics
                metrics = {
                    'Overall Sentiment': analysis_data.get('sentiment_summary', {}).get('overall_sentiment', 'N/A'),
                    'Avg Score': f"{analysis_data.get('sentiment_summary', {}).get('average_sentiment_score', 0):.3f}",
                    'Total Comments': analysis_data.get('total_comments_analyzed', 0)
                }
                
                for metric_name, metric_value in metrics.items():
                    self.render_metric_card(metric_name, metric_value)
    
    def render_emotion_analysis(self, analysis_data: Dict[str, Any]):
        """Render emotion analysis section"""
        st.markdown("### 😊 Emotion Analysis")
        
        emotion_data = analysis_data.get('emotion_summary', {}).get('distribution', {})
        if emotion_data:
            fig = chart_generator.create_emotion_bar_chart(emotion_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotion data available for this analysis")
    
    def render_aspect_analysis(self, analysis_data: Dict[str, Any]):
        """Render aspect-based analysis section"""
        st.markdown("### 🔍 Aspect Analysis")
        
        aspect_data = analysis_data.get('aspect_summary', {})
        if aspect_data:
            fig = chart_generator.create_aspect_sentiment_chart(aspect_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display aspect details
            with st.expander("View Aspect Details"):
                for aspect, details in aspect_data.items():
                    st.write(f"**{aspect.title()}**")
                    st.write(f"Total Mentions: {details['total_mentions']}")
                    st.write(f"Dominant Sentiment: {details['dominant_sentiment']}")
                    st.progress(details['sentiment_distribution'].get('positive', {}).get('percentage', 0) / 100)
                    st.write("---")
        else:
            st.info("No aspects detected in this analysis")
    
    def render_engagement_metrics(self, analysis_data: Dict[str, Any]):
        """Render engagement metrics section"""
        st.markdown("### 📈 Engagement Metrics")
        
        engagement_data = analysis_data.get('engagement_metrics', {})
        if engagement_data:
            metrics = {
                'Avg Score': engagement_data.get('average_score', 0),
                'Total Engagement': engagement_data.get('total_engagement', 0),
                'Avg Words/Comment': engagement_data.get('average_word_count', 0),
                'Most Engaged': engagement_data.get('most_engaged_comment', 0)
            }
            
            self.render_metrics_row(metrics)
        else:
            st.info("No engagement metrics available")
    
    def render_comment_explorer(self, comments: List[Dict[str, Any]]):
        """Render interactive comment explorer"""
        st.markdown("### 💬 Comment Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Negative", "Neutral"]
            )
        
        with col2:
            emotion_filter = st.selectbox(
                "Filter by Emotion",
                ["All", "Joy", "Anger", "Sadness", "Fear", "Surprise", "Love", "Neutral"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Score", "Sentiment Strength", "Length"]
            )
        
        # Apply filters
        filtered_comments = self._filter_comments(comments, sentiment_filter, emotion_filter, sort_by)
        
        # Display comments
        for i, comment in enumerate(filtered_comments[:20]):
            with st.expander(f"💬 Comment {i+1} (Score: {comment.get('original_comment', {}).get('score', 0)})"):
                self._render_comment_detail(comment)
    
    def _filter_comments(self, comments: List[Dict[str, Any]], 
                        sentiment_filter: str, emotion_filter: str, 
                        sort_by: str) -> List[Dict[str, Any]]:
        """Filter and sort comments"""
        filtered = comments
        
        # Apply sentiment filter
        if sentiment_filter != "All":
            filtered = [
                c for c in filtered 
                if c.get('sentiment_label', '').lower() == sentiment_filter.lower()
            ]
        
        # Apply emotion filter
        if emotion_filter != "All":
            filtered = [
                c for c in filtered 
                if c.get('emotion_label', '').lower() == emotion_filter.lower()
            ]
        
        # Apply sorting
        if sort_by == "Score":
            filtered.sort(key=lambda x: x.get('original_comment', {}).get('score', 0), reverse=True)
        elif sort_by == "Sentiment Strength":
            filtered.sort(key=lambda x: abs(x.get('sentiment_score', 0)), reverse=True)
        elif sort_by == "Length":
            filtered.sort(key=lambda x: len(x.get('original_comment', {}).get('body', '')), reverse=True)
        
        return filtered
    
    def _render_comment_detail(self, comment: Dict[str, Any]):
        """Render detailed comment view"""
        original_comment = comment.get('original_comment', {})
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(original_comment.get('body', ''))
        
        with col2:
            sentiment = comment.get('sentiment_label', 'neutral')
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'orange'
            }.get(sentiment, 'gray')
            
            st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment.title()}]")
            st.write(f"**Score:** {comment.get('sentiment_score', 0):.3f}")
            st.write(f"**Emotion:** {comment.get('emotion_label', 'neutral').title()}")
            st.write(f"**Confidence:** {comment.get('confidence', 0):.2f}")
    
    def render_export_section(self, analysis_data: Dict[str, Any]):
        """Render data export section"""
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                self._handle_export(analysis_data, 'csv')
        
        with col2:
            if st.button("📈 Export to Excel", use_container_width=True):
                self._handle_export(analysis_data, 'excel')
        
        with col3:
            if st.button("📋 Export to JSON", use_container_width=True):
                self._handle_export(analysis_data, 'json')
    
    def _handle_export(self, analysis_data: Dict[str, Any], format: str):
        """Handle data export"""
        # This would integrate with the export utilities
        st.success(f"Export to {format.upper()} would be implemented here")
        # In practice, this would call export_utils.export_analysis_report()

# Global dashboard components instance
dashboard_components = DashboardComponents()