# app/visualizations/ui.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
from datetime import datetime, timedelta

# --- 1. BẢNG MÀU & CSS ---
COLORS = {
    "bg_dark": "#0E1117",
    "bg_card": "#161B22",
    "primary": "#00C095",       # Xanh lá (Chủ đạo)
    "accent": "#4FACFE",        # Xanh dương (Dự báo/Link)
    "text_main": "#E6EAF1",
    "text_sub": "#9CA3AF",
    "border": "#30363D",
    "sidebar": "#0D1117",
    "danger": "#FF4B4B",        # Đỏ (Tiêu cực/Xóa)
    "warning": "#FFD166"
}

def load_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        :root {{
            --primary: {COLORS['primary']};
            --accent: {COLORS['accent']};
            --bg-card: {COLORS['bg_card']};
            --text-main: {COLORS['text_main']};
            --text-sub: {COLORS['text_sub']};
            --border: {COLORS['border']};
        }}
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            background-color: {COLORS['bg_dark']};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['sidebar']};
            border-right: 1px solid var(--border);
        }}
        
        /* Glassmorphism Cards */
        .insider-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .insider-card:hover {{
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 192, 149, 0.15);
        }}

        /* Typography */
        .card-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 8px; color: var(--text-main); display: flex; align-items: center; gap: 10px; }}
        .card-desc {{ color: var(--text-sub); font-size: 0.9rem; line-height: 1.5; }}
        
        /* Buttons */
        .stButton button {{
            background-color: var(--bg-card); color: var(--text-main);
            border: 1px solid var(--border); border-radius: 8px;
            font-weight: 600; padding: 0.5rem 1rem; transition: all 0.2s;
        }}
        .stButton button:hover {{ border-color: var(--primary); color: var(--primary); }}
        
        /* Primary Button */
        .stButton button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, #00A37E 100%);
            color: #000; border: none; box-shadow: 0 4px 12px rgba(0, 192, 149, 0.3);
        }}
        .stButton button[kind="primary"]:hover {{ transform: scale(1.02); color: #000; }}

        /* AI Insight Box */
        .ai-insight-box {{
            background-color: #13161C; border-left: 4px solid var(--accent);
            padding: 20px; border-radius: 0 12px 12px 0; margin-top: 15px;
            line-height: 1.6; color: #E0E0E0; font-size: 1rem;
        }}

        /* Metrics Stats Box */
        .stat-box {{
            text-align: center; padding: 15px; background: var(--bg-card);
            border-radius: 12px; border: 1px solid var(--border);
            transition: border-color 0.3s;
        }}
        .stat-box:hover {{ border-color: var(--primary); }}
        .stat-val {{ font-size: 1.8rem; font-weight: 800; color: var(--primary); }}
        .stat-lbl {{ font-size: 0.85rem; color: var(--text-sub'); text-transform: uppercase; letter-spacing: 0.5px; }}

        /* Group/Topic List Items */
        .list-item {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 14px; background: var(--bg-card); border-radius: 8px;
            border: 1px solid var(--border); margin-bottom: 8px; font-size: 0.9rem;
        }}
        
        /* Progress Bar Customization */
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, var(--primary), var(--accent));
        }}
        
        /* Trend Indicator */
        .trend-up {{ color: #00C095; font-weight: bold; }}
        .trend-down {{ color: #FF4B4B; font-weight: bold; }}
        .trend-neutral {{ color: #FFD166; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. AUTH SCREENS ---
def render_login_screen():
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-top: 60px; margin-bottom: 40px;">
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 10px;">💎</div>
            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(to right, {COLORS['primary']}, {COLORS['accent']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Reddit Insider AI</h1>
            <p style="color: {COLORS['text_sub']}; font-size: 1.1rem; margin-top: 10px;">Nền tảng phân tích dữ liệu xã hội thông minh</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
def render_sidebar_logged_in(username, groups, logout_callback, add_group_callback, delete_group_callback):
    with st.sidebar:
        # User Profile
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px; padding: 15px; background: {COLORS['bg_card']}; border-radius: 12px; border: 1px solid {COLORS['border']};">
            <div style="width: 42px; height: 42px; background: linear-gradient(135deg, {COLORS['primary']}, #008F6E); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; color: black; font-weight: bold;">
                {username[0].upper()}
            </div>
            <div style="overflow: hidden;">
                <div style="font-weight: 700; color: white; font-size: 1rem;">{username}</div>
                <div style="font-size: 0.75rem; color: {COLORS['primary']};">• Online</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Menu
        st.markdown("### 🧭 Điều Hướng")
        if st.button("🏠 Dashboard", use_container_width=True, type="secondary"):
            st.session_state.page = "Dashboard"
            st.rerun()
        if st.button("📊 Phân Tích Xu Hướng", use_container_width=True, type="secondary"):
            st.session_state.page = "Trending"
            st.rerun()
        if st.button("🔗 Phân Tích Link", use_container_width=True, type="secondary"):
            st.session_state.page = "Analysis"
            st.rerun()

        st.divider()
        
        # Group Manager
        st.markdown("### 🎯 Nhóm Theo Dõi")
        with st.expander("➕ Thêm nhóm mới"):
            with st.form("add_group"):
                new_sub = st.text_input("Tên Subreddit", label_visibility="collapsed", placeholder="vd: python")
                if st.form_submit_button("Thêm", use_container_width=True, type="primary"):
                    if new_sub: add_group_callback(new_sub)
        
        if groups:
            st.markdown(f"<div style='margin: 10px 0 10px 0; color:{COLORS['text_sub']}; font-size:0.8rem;'>Đang theo dõi {len(groups)} cộng đồng</div>", unsafe_allow_html=True)
            for group in groups:
                c1, c2 = st.columns([4, 1])
                with c1: st.markdown(f"<div class='list-item'>r/{group['subreddit']}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_g_{group['id']}", type="secondary", help="Xóa nhóm"):
                        delete_group_callback(group['id'])
                        st.rerun()
        else:
            st.info("Chưa có nhóm nào.")

        st.markdown("---")
        if st.button("Đăng xuất", use_container_width=True): logout_callback()

# --- 4. DASHBOARD COMPONENTS ---

def render_dashboard_header(username):
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="font-size: 2.2rem; margin-bottom: 10px;">Xin chào, {username}! 👋</h1>
        <p style="color: {COLORS['text_sub']}; font-size: 1.1rem;">
            Hệ thống đã sẵn sàng. Cập nhật xu hướng mới nhất ngay bây giờ.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_feature_card(icon, title, desc, btn_key, btn_label, on_click_action):
    st.markdown(f"""
    <div class="insider-card">
        <div class="card-title"><span style="font-size: 1.5rem;">{icon}</span> {title}</div>
        <p class="card-desc" style="margin-bottom: 20px;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button(btn_label, key=btn_key, use_container_width=True, type="primary"):
        on_click_action()

def render_history_list(history, delete_callback):
    st.markdown("### 🕒 Hoạt động gần đây")
    if not history:
        st.info("Chưa có lịch sử phân tích nào.")
        return

    for item in history:
        with st.container():
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(f"""
                <div class="insider-card" style="padding: 15px; margin-bottom: 10px; min-height: auto; border-left: 3px solid {COLORS['accent']};">
                    <div style="font-weight: 600; color: {COLORS['text_main']}; font-size: 1rem;">{item['title']}</div>
                    <div style="font-size: 0.85rem; color: {COLORS['text_sub']}; margin-top: 6px; display: flex; gap: 15px;">
                        <span>🕒 {item['timestamp']}</span>
                        <a href="{item['url']}" target="_blank" style="color: {COLORS['primary']}; text-decoration: none;">Xem bài viết gốc ↗️</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.write("") 
                if st.button("🗑️", key=f"del_h_{item['id']}", type="secondary", help="Xóa"):
                    delete_callback(item['id'])
                    st.rerun()

# --- 5. TREND ANALYSIS COMPONENTS (UPDATED) ---

def render_trend_analysis(trend_data):
    """Hiển thị phân tích xu hướng từ TrendAnalysisService"""
    
    if 'error' in trend_data:
        st.error(f"Lỗi phân tích: {trend_data['error']}")
        if 'message' in trend_data:
            st.info(trend_data['message'])
        return

    st.markdown(f"## 📊 Phân Tích Xu Hướng: r/{trend_data['subreddit']}")
    
    # Header với thông tin cơ bản
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Bài viết", trend_data['data_summary']['total_posts_analyzed'])
    with col2:
        st.metric("💎 Engagement", f"{trend_data['data_summary']['total_engagement']:,}")
    with col3:
        st.metric("💬 Bình luận", f"{trend_data['data_summary']['total_comments']:,}")
    with col4:
        st.metric("📅 Phân tích", f"{trend_data['analysis_period_days']} ngày")

    st.divider()

    # Layout chính: 2 cột
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Dự báo xu hướng
        render_forecast_section(trend_data.get('forecast', {}))
        
        # Phân tích giờ cao điểm
        render_peak_hours(trend_data.get('peak_hours', []))

    with col_right:
        # Chủ đề nổi bật
        render_top_topics(trend_data.get('top_topics', []))
        
        # Từ khóa quan trọng
        render_keywords(trend_data.get('top_keywords', []))

def render_forecast_section(forecast_data):
    """Hiển thị phần dự báo xu hướng"""
    
    st.markdown("### 🔮 Dự Báo Xu Hướng")
    
    if 'error' in forecast_data:
        st.warning(f"Không thể dự báo: {forecast_data.get('message', forecast_data['error'])}")
        return

    if 'forecast' not in forecast_data:
        st.info("Đang tính toán dự báo...")
        return

    # Hiển thị xu hướng hiện tại
    trend_dir = forecast_data.get('trend_direction', 'Không xác định')
    trend_class = "trend-up" if "Tăng" in trend_dir else "trend-down" if "Giảm" in trend_dir else "trend-neutral"
    
    st.markdown(f"""
    <div class="insider-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.9rem; color: {COLORS['text_sub']};">Xu hướng hiện tại</div>
                <div class="{trend_class}" style="font-size: 1.2rem;">{trend_dir}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: {COLORS['text_sub']};">Độ dốc</div>
                <div style="font-size: 1.1rem; font-weight: bold;">{forecast_data.get('trend_slope', 0):.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Biểu đồ dự báo
    if forecast_data['forecast']:
        forecast_df = pd.DataFrame(forecast_data['forecast'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        fig = go.Figure()
        
        # Vùng confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df['predicted_upper'], forecast_df['predicted_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(79, 172, 254, 0.2)',
            line_color='rgba(255,255,255,0)',
            name=f"{forecast_data.get('confidence_interval', '80%')} CI"
        ))
        
        # Đường dự báo chính
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_engagement'],
            mode='lines+markers',
            line=dict(color=COLORS['accent'], width=3),
            marker=dict(size=6),
            name='Dự báo Engagement'
        ))
        
        fig.update_layout(
            title="Dự báo Engagement 7 ngày tới",
            xaxis_title="Ngày",
            yaxis_title="Engagement dự báo",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_main']),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Bảng dự báo chi tiết
        with st.expander("📋 Chi tiết dự báo"):
            display_df = forecast_df.copy()
            display_df['Engagement'] = display_df['predicted_engagement'].round(1)
            display_df['Khoảng tin cậy'] = display_df.apply(
                lambda x: f"{x['predicted_lower']:.1f} - {x['predicted_upper']:.1f}", axis=1
            )
            st.dataframe(
                display_df[['date', 'Engagement', 'Khoảng tin cậy']],
                hide_index=True,
                use_container_width=True
            )

def render_peak_hours(peak_hours):
    """Hiển thị phân tích giờ cao điểm"""
    if not peak_hours:
        return

    st.markdown("### 🕐 Giờ Hoạt Động Cao Điểm")
    
    peak_df = pd.DataFrame(peak_hours)
    peak_df = peak_df.sort_values('hour')
    
    fig = px.bar(
        peak_df, 
        x='hour', 
        y='total_engagement',
        title="Engagement theo giờ trong ngày",
        color='total_engagement',
        color_continuous_scale=['#1f77b4', '#00C095']
    )
    
    fig.update_layout(
        xaxis_title="Giờ trong ngày",
        yaxis_title="Tổng Engagement",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_main']),
        height=300,
        showlegend=False
    )
    fig.update_xaxes(tickvals=list(range(0, 24, 3)))
    
    st.plotly_chart(fig, use_container_width=True)

    # Top 3 giờ cao điểm
    top_hours = sorted(peak_hours, key=lambda x: x['total_engagement'], reverse=True)[:3]
    for i, hour_data in enumerate(top_hours, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background: {COLORS['bg_card']}; border-radius: 8px; margin: 5px 0;">
            <span>{emoji} {hour_data['hour']:02d}:00</span>
            <span style="font-weight: bold; color: {COLORS['primary']};">{hour_data['total_engagement']} engagement</span>
        </div>
        """, unsafe_allow_html=True)

def render_top_topics(topics):
    """Hiển thị các chủ đề nổi bật"""
    if not topics:
        return

    st.markdown("### 🎯 Chủ Đề Nổi Bật")
    
    for topic in topics[:5]:  # Hiển thị top 5
        percentage = topic.get('percentage', 0)
        st.markdown(f"""
        <div class="insider-card" style="padding: 15px; margin-bottom: 10px;">
            <div style="font-weight: 600; margin-bottom: 8px;">{topic['name']}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem; color: {COLORS['text_sub']};">
                    {topic['frequency']} bài viết
                </span>
                <span style="font-size: 0.8rem; color: {COLORS['primary']}; font-weight: bold;">
                    {percentage:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_keywords(keywords):
    """Hiển thị từ khóa quan trọng"""
    if not keywords:
        return

    st.markdown("### 🔑 Từ Khóa Quan Trọng")
    
    # Tạo word cloud
    word_freq = {kw['keyword']: kw['score'] * 100 for kw in keywords}
    
    if word_freq:
        # Tạo word cloud với nền trong suốt
        wc = WordCloud(
            width=400, 
            height=200,
            background_color=None,
            mode='RGBA',
            colormap='viridis',
            relative_scaling=0.5,
            max_words=20
        ).generate_from_frequencies(word_freq)
        
        # Hiển thị word cloud
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        st.pyplot(fig, use_container_width=True)

    # Danh sách từ khóa
    for i, kw in enumerate(keywords[:8], 1):
        score_width = min(kw['score'] * 100, 100)
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>#{i} {kw['keyword']}</span>
                <span style="font-size: 0.8rem; color: {COLORS['text_sub']};">{kw['score']:.3f}</span>
            </div>
            <div style="width: 100%; height: 4px; background: {COLORS['border']}; border-radius: 2px;">
                <div style="width: {score_width}%; height: 100%; background: {COLORS['primary']}; border-radius: 2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_trending_card(post, analyze_callback):
    """Hiển thị card bài viết trending"""
    title = post.get('title', 'No Title')
    sub = post.get('subreddit', 'reddit')
    author = post.get('author', 'unknown')
    time_str = post.get('time_str', '')
    thumbnail = post.get('thumbnail')
    link = post.get('url')
    post_id = post.get('id', str(hash(title)))

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            if thumbnail and thumbnail.startswith('http'):
                st.image(thumbnail, use_container_width=True)
            else:
                st.markdown(f"<div style='height:80px; background:{COLORS['bg_card']}; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:2rem; border:1px solid {COLORS['border']}'>📝</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"#### {title}")
            st.caption(f"r/{sub} • {author} • {time_str}")
            
            b1, b2 = st.columns([1.5, 2.5])
            with b1:
                if st.button("⚡ Phân tích", key=f"btn_{post_id}", type="primary"): 
                    analyze_callback(link)
            with b2:
                st.link_button("Xem gốc ↗️", link)
        st.divider()

# --- 6. ANALYSIS RESULT COMPONENTS ---

def render_analysis_result_full(data):
    """Hiển thị kết quả phân tích bài viết (giữ nguyên)"""
    meta = data['meta']
    df = data['df']
    summary = data['summary']

    st.markdown(f"## 📄 {meta['title']}")
    st.caption(f"Tác giả: {meta['author']} • Subreddit: r/{meta['subreddit']} • {meta['count']} bình luận")

    # 1. Metrics Overview (KPIs)
    c1, c2, c3, c4 = st.columns(4)
    total_comments = len(df)
    avg_score = df['polarity'].mean() if not df.empty else 0
    pos_count = len(df[df['sentiment'] == 'Tích cực'])
    neg_count = len(df[df['sentiment'] == 'Tiêu cực'])
    
    with c1: 
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{total_comments}</div><div class='stat-lbl'>Bình luận</div></div>", unsafe_allow_html=True)
    with c2: 
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{avg_score:.2f}</div><div class='stat-lbl'>Điểm cảm xúc</div></div>", unsafe_allow_html=True)
    with c3: 
        st.markdown(f"<div class='stat-box'><div class='stat-val' style='color:{COLORS['primary']}'>{pos_count}</div><div class='stat-lbl'>Tích cực</div></div>", unsafe_allow_html=True)
    with c4: 
        st.markdown(f"<div class='stat-box'><div class='stat-val' style='color:{COLORS['danger']}'>{neg_count}</div><div class='stat-lbl'>Tiêu cực</div></div>", unsafe_allow_html=True)

    st.write("")

    # 2. Tabs Layout
    t1, t2, t3 = st.tabs(["🤖 AI Insight", "📊 Biểu Đồ", "📋 Dữ Liệu Chi Tiết"])

    with t1: 
        render_ai_summary_box(summary)
    with t2: 
        render_charts(df)
    with t3: 
        render_data_table(df)

def render_charts(df):
    """Hiển thị biểu đồ phân tích (giữ nguyên)"""
    chart_colors = [COLORS['primary'], COLORS['danger'], COLORS['warning'], COLORS['accent']]
    if df.empty:
        st.info("Chưa đủ dữ liệu.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🎭 Phân bố Cảm Xúc")
        fig = px.pie(df, names='sentiment', hole=0.6, color_discrete_sequence=chart_colors)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color=COLORS['text_main'], 
            showlegend=True, 
            margin=dict(t=20, b=20, l=20, r=20), 
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### 🌊 Diễn biến theo thời gian")
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            fig2 = px.scatter(
                df_sorted, 
                x='timestamp', 
                y='polarity', 
                color='sentiment', 
                color_discrete_sequence=chart_colors
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font_color=COLORS['text_main'], 
                xaxis=dict(showgrid=False), 
                yaxis=dict(showgrid=True, gridcolor='#333')
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Cần ít nhất 2 bình luận có thời gian để vẽ biểu đồ.")

def render_data_table(df):
    """Hiển thị bảng dữ liệu (giữ nguyên)"""
    st.markdown("##### 📥 Dữ liệu chi tiết")
    
    # Nút Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    c1, c2 = st.columns([3, 1])
    with c2:
        st.download_button(
            label="Tải xuống CSV",
            data=csv,
            file_name='reddit_data.csv',
            mime='text/csv',
            key='download-csv',
            use_container_width=True,
            type="primary"
        )
    
    # Bảng dữ liệu Interactive
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Thời gian", format="D MMM, HH:mm"),
            "polarity": st.column_config.ProgressColumn("Điểm số", min_value=-1, max_value=1, format="%.2f"),
            "body": st.column_config.TextColumn("Nội dung bình luận", width="large"),
            "sentiment": st.column_config.TextColumn("Cảm xúc"),
            "emotion": st.column_config.TextColumn("Chi tiết"),
            "author": st.column_config.TextColumn("Người dùng")
        },
        height=400,
        hide_index=True
    )

def render_ai_summary_box(summary_text):
    """Hiển thị AI Insight (giữ nguyên)"""
    st.markdown("### 🤖 AI Insight")
    # Làm sạch text, xóa các thẻ HTML thừa nếu AI sinh ra
    clean_text = summary_text.replace("```html", "").replace("```", "").replace("</div>", "")
    
    st.markdown(f"""
    <div class="ai-insight-box">
        {clean_text}
    </div>
    """, unsafe_allow_html=True)