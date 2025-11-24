import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generator for various visualization charts"""
    
    def __init__(self):
        self.color_scheme = {
            'positive': '#00D26A',
            'negative': '#FF4757',
            'neutral': '#FFB800',
            'joy': '#FF6B9D',
            'anger': '#FF4757',
            'sadness': '#5352ED',
            'fear': '#3742FA',
            'surprise': '#FF9F1A',
            'love': '#FF3838'
        }
    
    def create_sentiment_pie_chart(self, sentiment_data: Dict[str, Any]) -> go.Figure:
        """Create a pie chart for sentiment distribution"""
        labels = list(sentiment_data.keys())
        values = [data['percentage'] for data in sentiment_data.values()]
        
        colors = [self.color_scheme.get(label, '#CCCCCC') for label in labels]
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Sentiment Distribution",
            color=labels,
            color_discrete_map=self.color_scheme
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<br>%{percent}'
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    def create_emotion_bar_chart(self, emotion_data: Dict[str, int]) -> go.Figure:
        """Create a bar chart for emotion distribution"""
        emotions = list(emotion_data.keys())
        counts = list(emotion_data.values())
        
        colors = [self.color_scheme.get(emotion, '#CCCCCC') for emotion in emotions]
        
        fig = px.bar(
            x=emotions,
            y=counts,
            title="Emotion Distribution",
            labels={'x': 'Emotion', 'y': 'Count'},
            color=emotions,
            color_discrete_map=self.color_scheme
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=20)
        )
        
        return fig
    
    def create_sentiment_trend_chart(self, trend_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a line chart for sentiment trends over time"""
        if not trend_data:
            return self._create_empty_chart("No trend data available")
        
        df = pd.DataFrame(trend_data)
        
        fig = go.Figure()
        
        # Add lines for each sentiment
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['positive'],
            mode='lines+markers',
            name='Positive',
            line=dict(color=self.color_scheme['positive'], width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['negative'],
            mode='lines+markers',
            name='Negative',
            line=dict(color=self.color_scheme['negative'], width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['neutral'],
            mode='lines+markers',
            name='Neutral',
            line=dict(color=self.color_scheme['neutral'], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Percentage",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=50, l=50, r=20)
        )
        
        return fig
    
    def create_aspect_sentiment_chart(self, aspect_data: Dict[str, Any]) -> go.Figure:
        """Create a chart for aspect-based sentiment analysis"""
        if not aspect_data:
            return self._create_empty_chart("No aspect data available")
        
        aspects = list(aspect_data.keys())
        
        # Prepare data for stacked bar chart
        positive_data = []
        negative_data = []
        neutral_data = []
        
        for aspect in aspects:
            dist = aspect_data[aspect]['sentiment_distribution']
            positive_data.append(dist.get('positive', {}).get('percentage', 0))
            negative_data.append(dist.get('negative', {}).get('percentage', 0))
            neutral_data.append(dist.get('neutral', {}).get('percentage', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Positive',
            x=aspects,
            y=positive_data,
            marker_color=self.color_scheme['positive']
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=aspects,
            y=neutral_data,
            marker_color=self.color_scheme['neutral']
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=aspects,
            y=negative_data,
            marker_color=self.color_scheme['negative']
        ))
        
        fig.update_layout(
            title="Aspect-Based Sentiment Analysis",
            xaxis_title="Aspects",
            yaxis_title="Percentage",
            barmode='stack',
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=100, l=50, r=20)
        )
        
        return fig
    
    def create_engagement_metrics_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create a chart for engagement metrics"""
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig = px.bar(
            x=metric_names,
            y=metric_values,
            title="Engagement Metrics",
            labels={'x': 'Metric', 'y': 'Value'},
            color=metric_names,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, b=100, l=50, r=20)
        )
        
        return fig
    
    def create_word_cloud_data(self, comments: List[Dict[str, Any]], 
                             max_words: int = 100) -> Dict[str, int]:
        """Prepare data for word cloud visualization"""
        from collections import Counter
        import re
        
        all_text = ' '.join([
            comment.get('cleaned_body', comment.get('body', ''))
            for comment in comments
        ])
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by'}
        for word in stopwords:
            word_freq.pop(word, None)
        
        return dict(word_freq.most_common(max_words))
    
    def create_comparison_chart(self, comparison_data: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create a comparison chart for multiple datasets"""
        categories = list(comparison_data.keys())
        metrics = list(next(iter(comparison_data.values())).keys())
        
        fig = make_subplots(rows=1, cols=len(metrics), 
                          subplot_titles=metrics,
                          shared_yaxes=True)
        
        for i, metric in enumerate(metrics):
            values = [comparison_data[cat][metric] for cat in categories]
            
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=categories,
                    y=values,
                    marker_color=px.colors.qualitative.Set3[i],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Comparison Analysis",
            height=400,
            showlegend=False,
            margin=dict(t=80, b=50, l=50, r=20)
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def update_color_scheme(self, new_scheme: Dict[str, str]):
        """Update the color scheme for charts"""
        self.color_scheme.update(new_scheme)
        logger.info("🎨 Updated chart color scheme")

# Global chart generator instance
chart_generator = ChartGenerator()