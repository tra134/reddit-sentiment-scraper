# app/visualizations/ui.py - ENHANCED VERSION WITH CLEANER UI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. IMPROVED COLOR SCHEME FOR BETTER READABILITY ---
COLORS = {
    "bg_dark": "#0F172A",           # Darker, softer background
    "bg_card": "#1E293B",           # Card background with better contrast
    "primary": "#3B82F6",           # Softer blue for better visibility
    "accent": "#8B5CF6",            # Purple accent for variety
    "secondary": "#EF4444",         # Red for alerts
    "text_main": "#F1F5F9",         # Softer white for text
    "text_sub": "#94A3B8",          # Muted gray for subtext
    "border": "#334155",            # Softer border color
    "sidebar": "#0F172A",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "success": "#10B981",           # Softer green
    "info": "#0EA5E9",
    "chart_bg": "#1E293B"
}

SENTIMENT_COLORS = {
    "Rất tích cực": "#10B981",      # Green
    "Tích cực": "#34D399",          # Light green
    "Trung lập": "#9CA3AF",         # Gray
    "Tiêu cực": "#F59E0B",          # Amber/Orange
    "Rất tiêu cực": "#EF4444"       # Red
}

def load_css():
    """Load cleaner CSS styles for enhanced readability"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --primary: {COLORS['primary']};
            --accent: {COLORS['accent']};
            --secondary: {COLORS['secondary']};
            --bg-card: {COLORS['bg_card']};
            --text-main: {COLORS['text_main']};
            --text-sub: {COLORS['text_sub']};
            --border: {COLORS['border']};
            --danger: {COLORS['danger']};
            --warning: {COLORS['warning']};
            --success: {COLORS['success']};
            --info: {COLORS['info']};
            --sidebar: {COLORS['sidebar']};
            --chart-bg: {COLORS['chart_bg']};
        }}
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            background-color: {COLORS['bg_dark']};
            line-height: 1.6;
        }}
        
        /* Improve sidebar */
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar);
            border-right: 1px solid var(--border);
        }}
        
        /* Cleaner dashboard cards */
        .dashboard-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.2s ease;
        }}
        
        .dashboard-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        }}
        
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .card-subtitle {{
            color: var(--text-sub);
            font-size: 0.875rem;
            margin-bottom: 12px;
        }}
        
        /* Cleaner KPI cards */
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            transition: all 0.2s ease;
            height: 100%;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }}
        
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
            margin: 8px 0;
        }}
        
        .kpi-label {{
            font-size: 0.8rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .kpi-change {{
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 4px;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-block;
        }}
        
        /* Cleaner sentiment badges */
        .sentiment-badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 2px;
        }}
        
        /* Improved filter section */
        .filter-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        
        /* Cleaner comment cards */
        .comment-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            margin: 10px 0;
            transition: all 0.2s ease;
        }}
        
        .comment-card:hover {{
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.05);
        }}
        
        /* Cleaner chart containers */
        .chart-container {{
            background: var(--chart-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        /* Cleaner insight boxes */
        .insight-box {{
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
            border-left: 4px solid var(--accent);
            padding: 18px;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
        }}
        
        /* Better spacing */
        .spacer-xs {{ margin-top: 4px; }}
        .spacer-sm {{ margin-top: 8px; }}
        .spacer-md {{ margin-top: 16px; }}
        .spacer-lg {{ margin-top: 24px; }}
        .spacer-xl {{ margin-top: 32px; }}
        
        /* Improved progress bars */
        .progress-bar {{
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
            margin: 8px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }}
        
        /* Better button styling */
        .stButton button {{
            font-weight: 500 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }}
        
        /* Cleaner tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }}
        
        /* Improved select boxes and inputs */
        .stSelectbox, .stMultiselect, .stDateInput, .stSlider {{
            margin-bottom: 8px;
        }}
        
        /* Better table styling */
        .dataframe {{
            background: var(--bg-card) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }}
        
        /* Cleaner headers */
        h1 {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 16px !important;
        }}
        
        h2 {{
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
        }}
        
        h3 {{
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
        }}
        
        /* Status indicators */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
        }}
        
        .status-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
        }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. IMPROVED DASHBOARD COMPONENTS ---
def render_dashboard_header(username="Người dùng", total_data_points=0):
    """Render cleaner dashboard header"""
    # Đảm bảo username không rỗng
    display_name = username if username and username != "" else "Người dùng"
    
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <h1 style="margin-bottom: 4px;">Phân Tích Cảm Xúc Dashboard</h1>
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <div style="color: var(--text-sub); font-size: 0.95rem;">
                        Xin chào, <span style="color: var(--primary); font-weight: 600;">{display_name}</span>!
                    </div>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span>Live Analysis</span>
                    </div>
                </div>
                <div style="color: var(--text-sub); font-size: 0.85rem; margin-top: 8px;">
                    📊 {total_data_points:,} điểm dữ liệu • 🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(metrics):
    """Render cleaner KPI cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    kpi_data = [
        {
            "label": "Tổng bình luận",
            "value": metrics.get('total_comments', 0),
            "change": metrics.get('comments_change', 0),
            "color": "primary"
        },
        {
            "label": "Tỷ lệ tích cực",
            "value": f"{metrics.get('positive_rate', 0):.1f}%",
            "change": metrics.get('positive_change', 0),
            "color": "success"
        },
        {
            "label": "Tỷ lệ tiêu cực",
            "value": f"{metrics.get('negative_rate', 0):.1f}%",
            "change": metrics.get('negative_change', 0),
            "color": "danger"
        },
        {
            "label": "Điểm TB",
            "value": f"{metrics.get('avg_sentiment', 0):.2f}",
            "change": metrics.get('sentiment_change', 0),
            "color": "warning"
        }
    ]
    
    for i, kpi in enumerate(kpi_data):
        col = [col1, col2, col3, col4][i]
        with col:
            change_color = "success" if kpi["change"] >= 0 else "danger"
            change_icon = "↗️" if kpi["change"] >= 0 else "↘️"
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{kpi['label']}</div>
                <div class="kpi-value">{kpi['value']}</div>
                <div class="kpi-change" style="color: var(--{change_color}); background: rgba(var(--{change_color}-rgb), 0.1);">
                    {change_icon} {abs(kpi['change']):.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_filters(groups=None):
    """Render cleaner filters - CHỈ DÀNH CHO REDDIT"""
    if groups is None:
        groups = []
    
    with st.container():
        st.markdown('<div class="card-title">🔍 Bộ Lọc Phân Tích Reddit</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.date_input(
                    "Khoảng thời gian",
                    value=[datetime.now() - timedelta(days=7), datetime.now()],
                    key="dashboard_date_range"
                )
            
            with col2:
                st.multiselect(
                    "Loại cảm xúc",
                    options=["Rất tích cực", "Tích cực", "Trung lập", "Tiêu cực", "Rất tiêu cực"],
                    default=["Rất tích cực", "Tích cực", "Trung lập", "Tiêu cực", "Rất tiêu cực"],
                    key="dashboard_sentiment_filter"
                )
            
            with col3:
                # Tạo options từ groups của user
                subreddit_options = ["Tất cả"]
                for group in groups:
                    raw_name = group.get('name', '')
                    # Clean tên
                    clean_name = raw_name
                    if clean_name.startswith('r/'):
                        clean_name = clean_name[2:]
                    if clean_name.startswith('rj'):
                        clean_name = clean_name[2:]
                    clean_name = clean_name.rstrip('/').lower().strip()
                    
                    if clean_name and clean_name not in subreddit_options:
                        subreddit_options.append(clean_name)
                
                st.selectbox(
                    "Subreddit",
                    options=subreddit_options,
                    key="dashboard_subreddit_filter"
                )
        
        col_apply, col_reset = st.columns([1, 1])
        with col_apply:
            if st.button("✅ Áp dụng bộ lọc", use_container_width=True, type="primary"):
                st.session_state.filters_applied = True
                st.rerun()
        
        with col_reset:
            if st.button("🔄 Đặt lại", use_container_width=True):
                st.session_state.filters_applied = False
                st.rerun()
            
            

# --- 3. ENHANCED VISUALIZATIONS WITH CLEANER STYLING ---
def render_sentiment_timeline(df, title="Diễn biến cảm xúc theo thời gian"):
    """Render cleaner sentiment timeline chart"""
    if df.empty or 'timestamp' not in df.columns or 'polarity' not in df.columns:
        st.info("📊 Cần ít nhất 2 bình luận có thời gian để vẽ biểu đồ.")
        return
    
    with st.container():
        st.markdown(f'<div class="card-title">📈 {title}</div>', unsafe_allow_html=True)
        
        try:
            # Prepare data
            df_sorted = df.sort_values('timestamp').copy()
            df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
            df_sorted = df_sorted.dropna(subset=['timestamp', 'polarity'])
            
            if len(df_sorted) < 2:
                st.info("📊 Cần thêm dữ liệu để hiển thị diễn biến.")
                return
            
            # Calculate moving average
            df_sorted['moving_avg'] = df_sorted['polarity'].rolling(window=5, min_periods=1).mean()
            
            # Create cleaner figure
            fig = go.Figure()
            
            # Add scatter points with better styling
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=df_sorted['polarity'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df_sorted['polarity'],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(width=1, color='white'),
                    symbol='circle'
                ),
                name='Điểm cảm xúc',
                hovertemplate='<b>Thời gian:</b> %{x|%d/%m %H:%M}<br><b>Điểm:</b> %{y:.2f}<extra></extra>'
            ))
            
            # Add moving average line
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=df_sorted['moving_avg'],
                mode='lines',
                line=dict(color=COLORS['accent'], width=2.5),
                name='Trung bình động'
            ))
            
            # Update layout for better readability
            fig.update_layout(
                height=350,
                paper_bgcolor=COLORS['chart_bg'],
                plot_bgcolor=COLORS['chart_bg'],
                font=dict(color=COLORS['text_main'], size=12),
                xaxis=dict(
                    title="Thời gian",
                    gridcolor=COLORS['border'],
                    showgrid=True,
                    linecolor=COLORS['border'],
                    tickfont=dict(color=COLORS['text_sub'])
                ),
                yaxis=dict(
                    title="Điểm cảm xúc",
                    gridcolor=COLORS['border'],
                    showgrid=True,
                    range=[-1.1, 1.1],
                    linecolor=COLORS['border'],
                    tickfont=dict(color=COLORS['text_sub'])
                ),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text_main'])
                ),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        except Exception as e:
            st.error(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")

def render_sentiment_distribution(df, title="Phân bố cảm xúc"):
    """Render cleaner sentiment distribution chart"""
    if df.empty or 'sentiment' not in df.columns:
        st.info("📊 Chưa có dữ liệu cảm xúc để phân tích.")
        return
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cleaner donut chart
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            
            sentiment_mapping = {
                'positive': 'Tích cực',
                'negative': 'Tiêu cực', 
                'neutral': 'Trung lập',
                'very_positive': 'Rất tích cực',
                'very_negative': 'Rất tiêu cực'
            }
            
            sentiment_counts['sentiment'] = sentiment_counts['sentiment'].map(
                lambda x: sentiment_mapping.get(x, x)
            )
            
            fig = px.pie(
                sentiment_counts,
                names='sentiment',
                values='count',
                hole=0.5,
                color='sentiment',
                color_discrete_map=SENTIMENT_COLORS
            )
            
            fig.update_layout(
                height=320,
                paper_bgcolor=COLORS['chart_bg'],
                plot_bgcolor=COLORS['chart_bg'],
                font=dict(color=COLORS['text_main']),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.1,
                    font=dict(size=11)
                ),
                annotations=[dict(
                    text=f"<b>Tổng</b><br>{len(df)}",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False,
                    font_color=COLORS['text_main']
                )],
                margin=dict(l=10, r=100, t=20, b=20)
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                insidetextfont=dict(color='white', size=11)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Cleaner detailed statistics
            st.markdown('<div class="card-title">📊 Thống kê chi tiết</div>', unsafe_allow_html=True)
            
            total = len(df)
            for sentiment, color in SENTIMENT_COLORS.items():
                count = len(df[df['sentiment'] == sentiment])
                if count > 0:
                    percentage = (count / total) * 100
                    st.markdown(f"""
                    <div style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: {color}; font-weight: 500; font-size: 0.9rem;">{sentiment}</span>
                            <span style="font-weight: 600;">{percentage:.1f}%</span>
                        </div>
                        <div class="progress-bar">
                            <div style="width: {percentage}%; height: 100%; background: {color}; border-radius: 3px;"></div>
                        </div>
                        <div style="font-size: 0.75rem; color: var(--text-sub); text-align: right; margin-top: 2px;">
                            {count} bình luận
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_word_cloud_enhanced(df, title="Từ khóa cảm xúc nổi bật"):
    """Render cleaner word cloud with sentiment coloring"""
    if df.empty or 'body' not in df.columns or 'sentiment' not in df.columns:
        st.info("📊 Chưa có đủ dữ liệu để tạo word cloud.")
        return
    
    with st.container():
        st.markdown(f'<div class="card-title">☁️ {title}</div>', unsafe_allow_html=True)
        
        try:
            # Prepare text data
            positive_text = " ".join(df[df['sentiment'].isin(['Tích cực', 'Rất tích cực'])]['body'].astype(str))
            negative_text = " ".join(df[df['sentiment'].isin(['Tiêu cực', 'Rất tiêu cực'])]['body'].astype(str))
            
            if not positive_text and not negative_text:
                st.info("📝 Không có đủ văn bản để tạo word cloud.")
                return
            
            # Create cleaner word clouds
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                if positive_text:
                    st.markdown('<div style="color: var(--success); font-weight: 500; margin-bottom: 10px;">🟢 Từ khóa tích cực</div>', unsafe_allow_html=True)
                    create_single_wordcloud(positive_text, 'Greens')
                else:
                    st.info("📝 Chưa có từ khóa tích cực")
            
            with col_neg:
                if negative_text:
                    st.markdown('<div style="color: var(--danger); font-weight: 500; margin-bottom: 10px;">🔴 Từ khóa tiêu cực</div>', unsafe_allow_html=True)
                    create_single_wordcloud(negative_text, 'Reds')
                else:
                    st.info("📝 Chưa có từ khóa tiêu cực")
                    
        except Exception as e:
            st.error(f"❌ Lỗi khi tạo word cloud: {str(e)}")

def create_single_wordcloud(text, colormap='viridis'):
    """Create cleaner word cloud"""
    try:
        # Clean text
        text = ' '.join(text.split()[:300])  # Limit text length
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=400,
            height=250,
            background_color=COLORS['chart_bg'],
            colormap=colormap,
            max_words=40,
            contour_width=0,
            min_font_size=10,
            max_font_size=60,
            random_state=42
        ).generate(text)
        
        # Display
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Set background color
        fig.patch.set_facecolor(COLORS['chart_bg'])
        ax.set_facecolor(COLORS['chart_bg'])
        
        # Remove padding
        plt.tight_layout(pad=0)
        
        st.pyplot(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Lỗi tạo word cloud: {str(e)}")

def render_comparison_chart(df, title="So sánh theo thời gian"):
    """Render cleaner comparison chart"""
    if df.empty or 'timestamp' not in df.columns:
        st.info("📊 Chưa có đủ dữ liệu để so sánh.")
        return
    
    with st.container():
        st.markdown(f'<div class="card-title">📅 {title}</div>', unsafe_allow_html=True)
        
        try:
            # Prepare data
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
            df_copy['date'] = df_copy['timestamp'].dt.date
            
            # Group by date
            daily_stats = df_copy.groupby('date').agg({
                'polarity': 'mean',
                'body': 'count'
            }).reset_index()
            daily_stats.columns = ['date', 'avg_sentiment', 'comment_count']
            
            # Create cleaner figure
            fig = go.Figure()
            
            # Add cleaner bar chart
            fig.add_trace(go.Bar(
                x=daily_stats['date'],
                y=daily_stats['comment_count'],
                name='Số bình luận',
                marker_color=COLORS['accent'],
                opacity=0.8,
                yaxis='y2'
            ))
            
            # Add cleaner line chart
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['avg_sentiment'],
                mode='lines+markers',
                name='Điểm cảm xúc TB',
                line=dict(color=COLORS['primary'], width=2.5),
                marker=dict(size=8, color=COLORS['primary']),
                yaxis='y'
            ))
            
            # Cleaner layout
            fig.update_layout(
                height=350,
                paper_bgcolor=COLORS['chart_bg'],
                plot_bgcolor=COLORS['chart_bg'],
                font=dict(color=COLORS['text_main'], size=12),
                xaxis=dict(
                    title="Ngày",
                    gridcolor=COLORS['border'],
                    showgrid=True,
                    linecolor=COLORS['border'],
                    tickfont=dict(color=COLORS['text_sub'])
                ),
                yaxis=dict(
                    title="Điểm cảm xúc TB",
                    gridcolor=COLORS['border'],
                    showgrid=True,
                    range=[-1.1, 1.1],
                    linecolor=COLORS['border'],
                    tickfont=dict(color=COLORS['text_sub'])
                ),
                yaxis2=dict(
                    title="Số bình luận",
                    overlaying='y',
                    side='right',
                    gridcolor=COLORS['border'],
                    showgrid=False,
                    tickfont=dict(color=COLORS['text_sub'])
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text_main'])
                ),
                hovermode='x unified',
                margin=dict(l=10, r=10, t=30, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        except Exception as e:
            st.error(f"❌ Lỗi khi vẽ biểu đồ so sánh: {str(e)}")

def render_top_comments(df, title="Top bình luận tiêu biểu"):
    """Render cleaner top comments table"""
    if df.empty:
        st.info("📊 Chưa có bình luận để hiển thị.")
        return
    
    with st.container():
        st.markdown(f'<div class="card-title">💬 {title}</div>', unsafe_allow_html=True)
        
        # Cleaner tabs
        tab1, tab2, tab3 = st.tabs(["🟢 Tích cực", "🔴 Tiêu cực", "⚫ Trung lập"])
        
        with tab1:
            positive_comments = df[df['sentiment'].isin(['Tích cực', 'Rất tích cực'])]
            if not positive_comments.empty:
                display_comments_table(positive_comments.head(5), "Tích cực")
            else:
                st.info("📝 Chưa có bình luận tích cực")
        
        with tab2:
            negative_comments = df[df['sentiment'].isin(['Tiêu cực', 'Rất tiêu cực'])]
            if not negative_comments.empty:
                display_comments_table(negative_comments.head(5), "Tiêu cực")
            else:
                st.info("📝 Chưa có bình luận tiêu cực")
        
        with tab3:
            neutral_comments = df[df['sentiment'] == 'Trung lập']
            if not neutral_comments.empty:
                display_comments_table(neutral_comments.head(5), "Trung lập")
            else:
                st.info("📝 Chưa có bình luận trung lập")

def display_comments_table(df, sentiment_type):
    """Display cleaner comments table"""
    for idx, row in df.iterrows():
        sentiment_color = SENTIMENT_COLORS.get(sentiment_type, COLORS['text_sub'])
        
        # Format time if available
        time_str = ""
        if 'timestamp' in row:
            try:
                time_obj = pd.to_datetime(row['timestamp'])
                time_str = time_obj.strftime('%H:%M • %d/%m')
            except:
                time_str = str(row['timestamp'])[:10]
        
        st.markdown(f"""
        <div class="comment-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {sentiment_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 500;">
                        {sentiment_type}
                    </span>
                    <span style="font-size: 0.85rem; color: var(--text-sub);">
                        Điểm: {row.get('polarity', 0):.2f}
                    </span>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-sub);">
                    {time_str}
                </div>
            </div>
            <div style="color: var(--text-main); font-size: 0.9rem; line-height: 1.5;">
                {str(row.get('body', ''))[:180]}...
            </div>
            <div style="margin-top: 8px; font-size: 0.8rem; color: var(--text-sub); display: flex; align-items: center; gap: 4px;">
                👤 {row.get('author', 'Ẩn danh')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_gauge_chart(overall_score, title="Chỉ số tổng quan"):
    """Render cleaner gauge chart"""
    with st.container():
        st.markdown(f'<div class="card-title">📊 {title}</div>', unsafe_allow_html=True)
        
        try:
            # Normalize score to 0-100
            normalized_score = (overall_score + 1) * 50
            
            # Determine color
            if normalized_score >= 70:
                color = COLORS['success']
                label = "TỐT"
            elif normalized_score >= 40:
                color = COLORS['warning']
                label = "TRUNG BÌNH"
            else:
                color = COLORS['danger']
                label = "CẦN CẢI THIỆN"
            
            # Create cleaner gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=normalized_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': f"<span style='font-size:1rem'>{label}</span>",
                    'font': {'color': color, 'size': 16}
                },
                number={
                    'suffix': "%", 
                    'font': {'size': 32, 'color': COLORS['text_main']}
                },
                gauge={
                    'axis': {
                        'range': [0, 100], 
                        'tickwidth': 1,
                        'tickcolor': COLORS['text_main'],
                        'tickfont': {'color': COLORS['text_sub']}
                    },
                    'bar': {'color': color},
                    'bgcolor': COLORS['border'],
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.8,
                        'value': normalized_score
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                paper_bgcolor=COLORS['chart_bg'],
                font=dict(color=COLORS['text_main']),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        except Exception as e:
            st.error(f"❌ Lỗi khi vẽ gauge chart: {str(e)}")

# --- 4. IMPROVED ANALYSIS RESULT WITH CLEANER UI ---
def render_enhanced_analysis_result(data):
    """Render cleaner analysis result"""
    if not data or 'meta' not in data or 'df' not in data:
        st.error("❌ Dữ liệu phân tích không hợp lệ.")
        return
    
    meta = data['meta']
    df = data['df']
    summary = data.get('summary', 'Chưa có phân tích AI.')
    
    # Cleaner header
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h1 style="margin-bottom: 8px;">📊 Phân tích chi tiết</h1>
        <div style="color: var(--text-sub); font-size: 0.95rem; margin-bottom: 4px;">
            📝 {meta.get('title', 'Không có tiêu đề')[:100]}
        </div>
        <div style="color: var(--text-sub); font-size: 0.85rem;">
            👤 {meta.get('author', 'Ẩn danh')} • 
            🏷️ r/{meta.get('subreddit', 'unknown')} • 
            💬 {len(df)} bình luận • 
            📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    total_comments = len(df)
    avg_score = df['polarity'].mean() if not df.empty and 'polarity' in df.columns else 0
    pos_count = len(df[df['sentiment'].isin(['Tích cực', 'Rất tích cực'])]) if not df.empty and 'sentiment' in df.columns else 0
    neg_count = len(df[df['sentiment'].isin(['Tiêu cực', 'Rất tiêu cực'])]) if not df.empty and 'sentiment' in df.columns else 0
    
    render_kpi_cards({
        'total_comments': total_comments,
        'positive_rate': (pos_count / total_comments * 100) if total_comments > 0 else 0,
        'negative_rate': (neg_count / total_comments * 100) if total_comments > 0 else 0,
        'avg_sentiment': avg_score,
        'comments_change': 12.5,
        'positive_change': 8.3,
        'negative_change': -4.2,
        'sentiment_change': 0.15
    })
    
    # Filters
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    render_filters()
    
    # Main visualizations in cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        render_sentiment_distribution(df)
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        render_gauge_chart(avg_score)
    
    with col2:
        render_sentiment_timeline(df)
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        render_comparison_chart(df)
    
    # Word Cloud and Top Comments
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    render_word_cloud_enhanced(df)
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    render_top_comments(df)
    
    # Cleaner AI Insights section
    st.markdown('<div class="spacer-xl"></div>', unsafe_allow_html=True)
    render_ai_insights_section(summary, meta, df)

def render_ai_insights_section(summary, meta, df):
    """Render cleaner AI insights section"""
    with st.container():
        st.markdown('<div class="card-title">🤖 AI Insights & Gợi ý Hành động</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cleaner AI Summary
            st.markdown('<div style="color: var(--text-main); font-weight: 500; margin-bottom: 10px;">📋 Tóm tắt phân tích</div>', unsafe_allow_html=True)
            if summary and not summary.startswith("⚠️"):
                st.markdown(f"""
                <div class="insight-box">
                    <div style="font-size: 0.9rem; color: var(--text-main); line-height: 1.6;">
                        {summary[:400]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Chưa có phân tích AI. Vui lòng kiểm tra cấu hình API.")
        
        with col2:
            # Cleaner actionable recommendations
            st.markdown('<div style="color: var(--text-main); font-weight: 500; margin-bottom: 10px;">🎯 Gợi ý hành động</div>', unsafe_allow_html=True)
            
            recommendations = generate_recommendations(df)
            for i, rec in enumerate(recommendations[:3], 1):
                st.markdown(f"""
                <div style="background: var(--bg-card); padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid var(--primary);">
                    <div style="font-weight: 600; color: var(--text-main); font-size: 0.9rem;">{rec['title']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-sub); margin-top: 4px;">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Cleaner export options
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📥 Tải báo cáo PDF", use_container_width=True):
                st.success("✅ Báo cáo PDF đang được tạo...")
        
        with col_exp2:
            if st.button("📊 Xuất CSV", use_container_width=True):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Tải ngay",
                    data=csv,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv",
                    key="download_csv"
                )
        
        with col_exp3:
            if st.button("🔄 Phân tích lại", use_container_width=True):
                st.rerun()

# --- 5. IMPROVED SIDEBAR WITH CLEANER DESIGN ---
def render_enhanced_sidebar(username, groups, logout_callback, add_group_callback, delete_group_callback):
    """Render cleaner sidebar với quản lý subreddit groups"""
    with st.sidebar:
        # Cleaner User Profile
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 16px; 
                   background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border);">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, var(--primary), var(--accent)); 
                      border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                      font-size: 20px; color: white; font-weight: bold;">
                {username[0].upper()}
            </div>
            <div>
                <div style="font-weight: 600; color: var(--text-main);">{username}</div>
                <div style="display: flex; align-items: center; gap: 4px; margin-top: 2px;">
                    <div style="width: 8px; height: 8px; background: var(--success); border-radius: 50%;"></div>
                    <div style="font-size: 0.8rem; color: var(--success);">Online</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cleaner Navigation
        st.markdown('<div style="color: var(--text-sub); font-size: 0.9rem; margin-bottom: 8px;">ĐIỀU HƯỚNG</div>', unsafe_allow_html=True)
        
        nav_cols = st.columns(2)
        
        with nav_cols[0]:
            if st.button("🏠 Dashboard", use_container_width=True, 
                        type="primary" if st.session_state.get('page') == 'Dashboard' else "secondary"):
                st.session_state.page = "Dashboard"
                st.rerun()
        
        with nav_cols[1]:
            if st.button("📊 Phân tích URL", use_container_width=True,
                        type="primary" if st.session_state.get('page') == 'Analysis' else "secondary"):
                st.session_state.page = "Analysis"
                st.rerun()
        
        # ============ SUBREDDIT GROUP MANAGEMENT ============
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color: var(--text-sub); font-size: 0.9rem; margin-bottom: 8px;">QUẢN LÝ SUBREDDIT GROUPS</div>', unsafe_allow_html=True)
        
        # Add new subreddit form
        with st.form("add_subreddit_form", clear_on_submit=True):
            new_subreddit = st.text_input(
                "Thêm subreddit mới",
                placeholder="Nhập tên subreddit",
                help="Ví dụ: technology, programming, science",
                key="sidebar_subreddit_input"
            )
            
            col_add, _ = st.columns([1, 1])
            with col_add:
                add_submitted = st.form_submit_button("➕ Thêm vào nhóm", use_container_width=True)
            
            if add_submitted and new_subreddit:
                if add_group_callback(new_subreddit):
                    st.success(f"Đã thêm r/{new_subreddit}")
                    st.rerun()
                else:
                    st.error("Không thể thêm subreddit")
        
        # Display current groups
        if groups:
            st.markdown('<div style="color: var(--text-sub); font-size: 0.9rem; margin-top: 16px; margin-bottom: 8px;">NHÓM HIỆN TẠI</div>', unsafe_allow_html=True)
            
            for group in groups[:5]:  # Show max 5 groups
                col_group, col_del = st.columns([3, 1])
                
                with col_group:
                    # LẤY VÀ CLEAN TÊN
                    raw_name = group.get('name', '')
                    
                    # Clean tên để hiển thị
                    display_name = raw_name
                    
                    # Loại bỏ r/ nếu có
                    if display_name.startswith('r/'):
                        display_name = display_name[2:]
                    # Loại bỏ / nếu có ở cuối
                    display_name = display_name.rstrip('/')
                    # Chuyển về chữ thường
                    display_name = display_name.lower()
                    
                    # Format added date
                    added_date = group.get('added_date', '')
                    if added_date and len(str(added_date)) >= 10:
                        display_date = str(added_date)[:10]
                    else:
                        display_date = 'Chưa có'
                    
                    st.markdown(f"""
                    <div style="background: var(--bg-card); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border);">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="color: var(--primary);">🏷️</span>
                            <span style="font-size: 0.85rem; color: var(--text-main);">r/{display_name}</span>
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-sub); margin-top: 2px;">
                            Thêm: {display_date}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_del:
                    # SỬA: Tạo unique key an toàn
                    group_id = group.get('id', 'unknown')
                    unique_key = f"del_sidebar_{group_id}_{display_name}"
                    
                    if st.button("🗑️", key=unique_key, help="Xóa khỏi nhóm"):
                        if delete_group_callback(group_id):
                            st.rerun()
            
            if len(groups) > 5:
                st.caption(f"Và {len(groups) - 5} subreddit khác...")
        
        # Cleaner Data sources (CHỈ GIỮ REDDIT)
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color: var(--text-sub); font-size: 0.9rem; margin-bottom: 8px;">NGUỒN DỮ LIỆU</div>', unsafe_allow_html=True)
        
        st.info("📊 **Reddit** - Nguồn dữ liệu chính", icon="✅")
        
        # Analysis mode
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        
        
        # Quick Actions
        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color: var(--text-sub); font-size: 0.9rem; margin-bottom: 8px;">HÀNH ĐỘNG NHANH</div>', unsafe_allow_html=True)
        
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.rerun()
        
        with action_cols[1]:
            if st.button("📥 Xuất dữ liệu", use_container_width=True):
                st.success("Đang chuẩn bị dữ liệu...")
        
        # Cleaner Logout
        st.markdown('<div class="spacer-xl"></div>', unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất", use_container_width=True, type="secondary"):
            logout_callback()
        
        # Cleaner Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: var(--text-sub); font-size: 0.75rem; line-height: 1.4;">
            🔐 Reddit Sentiment Analyzer<br>
            📅 Phiên bản 2.1.0<br>
            © 2024 - Chỉ phân tích dữ liệu từ Reddit
        </div>
        """, unsafe_allow_html=True)

# --- 6. MAIN DASHBOARD VIEW WITH CLEANER UI ---
def render_main_dashboard(dashboard_data=None, username="Người dùng"):
    """Render cleaner main dashboard với dữ liệu thực"""
    
    # Kiểm tra dữ liệu
    if dashboard_data is None:
        dashboard_data = {}
    
    user_stats = dashboard_data.get('user_stats', {})
    trending_posts = dashboard_data.get('trending_posts', [])
    user_subreddits = dashboard_data.get('user_subreddits', [])
    
    # Cleaner Dashboard Header với stats thực
    render_dashboard_header(
        username=username,  # Sử dụng username được truyền vào
        total_data_points=user_stats.get('total_analyses', 0)
    )
    
    # KPI Cards với dữ liệu thực
    if trending_posts:
        total_comments = sum(post.get('comments_count', 0) for post in trending_posts[:10])
        
        # Tính sentiment từ trending posts
        total_posts = len(trending_posts[:10])
        
        # Tính điểm trung bình (normalized)
        if total_posts > 0:
            avg_score_raw = sum(post.get('score', 0) for post in trending_posts[:10]) / total_posts
            # Chuyển đổi điểm reddit (thường 0-5000) thành sentiment score (-1 đến 1)
            avg_sentiment = min(max(avg_score_raw / 2500 - 1, -1), 1)
            
            # Đếm post tích cực/tiêu cực dựa trên upvote ratio
            positive_posts = sum(1 for post in trending_posts[:10] 
                                if post.get('upvote_ratio', 0) > 0.7)
            negative_posts = sum(1 for post in trending_posts[:10] 
                                if post.get('upvote_ratio', 0) < 0.3)
            
            positive_rate = (positive_posts / total_posts * 100) if total_posts > 0 else 0
            negative_rate = (negative_posts / total_posts * 100) if total_posts > 0 else 0
        else:
            avg_sentiment = 0
            positive_rate = 0
            negative_rate = 0
            
        render_kpi_cards({
            'total_comments': total_comments,
            'positive_rate': positive_rate,
            'negative_rate': negative_rate,
            'avg_sentiment': avg_sentiment,
            'comments_change': 0,
            'positive_change': 0,
            'negative_change': 0,
            'sentiment_change': 0
        })
    else:
        # Hiển thị KPI từ user_stats
        render_kpi_cards({
            'total_comments': user_stats.get('total_analyses', 0),
            'positive_rate': 0,
            'negative_rate': 0,
            'avg_sentiment': user_stats.get('avg_sentiment', 0),
            'comments_change': 0,
            'positive_change': 0,
            'negative_change': 0,
            'sentiment_change': 0
        })
    
    # Filters chỉ cho Reddit - truyền groups
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    render_filters(user_subreddits)
    
    # Hiển thị trending posts nếu có
    if trending_posts:
        st.markdown('<div class="card-title">🔥 Bài viết Trending từ Subreddits của bạn</div>', unsafe_allow_html=True)
        
        for i, post in enumerate(trending_posts[:5]):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Clean subreddit name for display
                    subreddit_name = post.get('subreddit', 'unknown')
                    if subreddit_name.startswith('r/'):
                        subreddit_name = subreddit_name[2:]
                    if subreddit_name.startswith('rj'):
                        subreddit_name = subreddit_name[2:]
                    
                    st.markdown(f"**{post.get('title', 'Không có tiêu đề')}**")
                    st.caption(f"r/{subreddit_name} • 👤 {post.get('author', 'unknown')} • {post.get('time_str', '')}")
                    
                    if post.get('selftext'):
                        st.write(post['selftext'][:150] + "...")
                
                with col2:
                    st.metric("▲ Điểm", post.get('score', 0))
                    st.metric("💬 Bình luận", post.get('comments_count', 0))
                    
                    # Tạo unique key cho button
                    unique_key = f"dashboard_analyze_{i}_{post.get('id', 'unknown')}"
                    if st.button("Phân tích", key=unique_key, use_container_width=True):
                        st.session_state.analyze_url = post.get('url')
                        st.session_state.page = "Analysis"
                        st.rerun()
    
    # Chia layout cho charts
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Timeline từ dữ liệu thực nếu có
        if 'timeline_data' in dashboard_data and not dashboard_data['timeline_data'].empty:
            render_sentiment_timeline(dashboard_data['timeline_data'])
        else:
            st.info("📈 Chưa có dữ liệu timeline")
        
        # Top comments từ dữ liệu thực nếu có
        if 'comments_data' in dashboard_data and not dashboard_data['comments_data'].empty:
            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            render_top_comments(dashboard_data['comments_data'])
        else:
            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            st.info("💬 Chưa có dữ liệu bình luận")
    
    with col2:
        # Sentiment distribution từ dữ liệu thực nếu có
        if 'sentiment_data' in dashboard_data and not dashboard_data['sentiment_data'].empty:
            render_sentiment_distribution(dashboard_data['sentiment_data'])
        else:
            st.info("📊 Chưa có dữ liệu phân bố cảm xúc")
        
        # Gauge chart từ dữ liệu thực
        avg_sentiment = user_stats.get('avg_sentiment', 0)
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        render_gauge_chart(avg_sentiment)
    
    # AI Insights từ dữ liệu thực
    st.markdown('<div class="spacer-xl"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-title">📊 Thống kê & Phân tích</div>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.markdown('<div style="color: var(--text-main); font-weight: 500; margin-bottom: 10px;">📈 Thống kê người dùng</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-box">
                <div style="font-size: 0.9rem; color: var(--text-main); line-height: 1.6;">
                    <strong>📊 Tổng phân tích:</strong> {user_stats.get('total_analyses', 0)} bài<br>
                    <strong>🏷️ Subreddit theo dõi:</strong> {len(user_subreddits)}<br>
                    <strong>📅 Điểm TB:</strong> {user_stats.get('avg_sentiment', 0):.2f}<br>
                    <strong>🔥 Trending posts:</strong> {len(trending_posts)} bài
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats2:
            st.markdown('<div style="color: var(--text-main); font-weight: 500; margin-bottom: 10px;">🎯 Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="insight-box">
                <div style="font-size: 0.9rem; color: var(--text-main); line-height: 1.6;">
                    1. <strong>Thêm subreddit</strong> vào nhóm theo dõi<br>
                    2. <strong>Phân tích URL</strong> Reddit bất kỳ<br>
                    3. <strong>Theo dõi trending</strong> từ subreddits<br>
                    4. <strong>Xuất báo cáo</strong> phân tích chi tiết
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 7. UTILITY FUNCTIONS WITH CLEANER UI ---
def show_loading_animation(message="Đang tải dữ liệu..."):
    """Show cleaner loading animation"""
    with st.container():
        st.markdown(f"""
        <div style="text-align: center; padding: 40px;">
            <div style="font-size: 3rem; margin-bottom: 16px; color: var(--primary);">⏳</div>
            <div style="color: var(--text-main); font-size: 1rem; margin-bottom: 20px;">{message}</div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_error_message(error, title="Đã xảy ra lỗi"):
    """Show cleaner error message"""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border: 1px solid var(--danger);">
        <div style="font-size: 3rem; margin-bottom: 16px; color: var(--danger);">❌</div>
        <div style="color: var(--danger); font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">{title}</div>
        <div style="color: var(--text-main); font-size: 0.9rem; margin-bottom: 20px;">{error}</div>
        <button onclick="window.location.reload()" style="background: var(--danger); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 500;">
            🔄 Thử lại
        </button>
    </div>
    """, unsafe_allow_html=True)

def show_success_message(message, title="Thành công!"):
    """Show cleaner success message"""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px; background: rgba(16, 185, 129, 0.1); border-radius: 12px; border: 1px solid var(--success);">
        <div style="font-size: 3rem; margin-bottom: 16px; color: var(--success);">✅</div>
        <div style="color: var(--success); font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">{title}</div>
        <div style="color: var(--text-main); font-size: 0.9rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 8. INITIALIZATION ---
def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    if 'use_advanced_analysis' not in st.session_state:
        st.session_state.use_advanced_analysis = False
    if 'real_time_analysis' not in st.session_state:
        st.session_state.real_time_analysis = True
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = ['Reddit', 'Twitter', 'Facebook']
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False

# Keep the existing helper functions (they're fine)
def generate_sample_data():
    """Generate sample data chỉ với dữ liệu Reddit"""
    return {
        'timeline_data': pd.DataFrame(),  # Empty để tránh lỗi
        'sentiment_data': pd.DataFrame(),
        'comparison_data': pd.DataFrame(),
        'comments_data': pd.DataFrame(),
        'wordcloud_data': pd.DataFrame(),
        'trending_posts': [],
        'user_stats': {'total_analyses': 0, 'avg_sentiment': 0, 'subreddit_count': 0}
    }

def generate_recommendations(df):
    """Generate actionable recommendations based on data"""
    # Same as before...
    recommendations = []
    
    if df.empty:
        return [
            {"title": "Thu thập thêm dữ liệu", "description": "Cần thêm bình luận để phân tích chính xác hơn"},
            {"title": "Kiểm tra nguồn dữ liệu", "description": "Đảm bảo kết nối API hoạt động ổn định"}
        ]
    
    total = len(df)
    positive_rate = len(df[df['sentiment'].isin(['Tích cực', 'Rất tích cực'])]) / total * 100
    negative_rate = len(df[df['sentiment'].isin(['Tiêu cực', 'Rất tiêu cực'])]) / total * 100
    avg_score = df['polarity'].mean()
    
    if negative_rate > 30:
        recommendations.append({
            "title": "Giảm phản hồi tiêu cực",
            "description": f"Tỷ lệ tiêu cực cao ({negative_rate:.1f}%). Xem xét cải thiện sản phẩm/dịch vụ"
        })
    
    if positive_rate < 40:
        recommendations.append({
            "title": "Tăng tương tác tích cực",
            "description": f"Tỷ lệ tích cực thấp ({positive_rate:.1f}%). Khuyến khích phản hồi tích cực"
        })
    
    if avg_score < 0:
        recommendations.append({
            "title": "Cải thiện điểm cảm xúc",
            "description": f"Điểm trung bình âm ({avg_score:.2f}). Tập trung vào giải quyết vấn đề người dùng"
        })
    
    recommendations.extend([
        {
            "title": "Theo dõi xu hướng",
            "description": "Phân tích diễn biến theo thời gian để phát hiện thay đổi sớm"
        },
        {
            "title": "Phân tích từ khóa",
            "description": "Xác định từ khóa chính ảnh hưởng đến cảm xúc người dùng"
        },
        {
            "title": "So sánh đối thủ",
            "description": "Phân tích cảm xúc so với đối thủ cạnh tranh"
        }
    ])
    
    return recommendations[:5]

# Main initialization
load_css()
init_session_state()

# --- EXPORTED FUNCTIONS FOR APP ---
__all__ = [
    'load_css',
    'render_enhanced_sidebar',
    'render_main_dashboard',  # Đã sửa signature
    'render_enhanced_analysis_result',
    'render_sentiment_timeline',
    'render_sentiment_distribution',
    'render_word_cloud_enhanced',
    'render_comparison_chart',
    'render_top_comments',
    'render_gauge_chart',
    'render_kpi_cards',
    'render_filters',
    'show_loading_animation',
    'show_error_message',
    'show_success_message',
    'COLORS',
    'SENTIMENT_COLORS',
    'generate_sample_data',
    'generate_recommendations'
]