# app/visualizations/ui.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import hashlib

# --- 1. BẢNG MÀU & CSS CẢI TIẾN ---
COLORS = {
    "bg_dark": "#0E1117",
    "bg_card": "#161B22",
    "primary": "#00C095",
    "accent": "#4FACFE",
    "text_main": "#E6EAF1",
    "text_sub": "#9CA3AF",
    "border": "#30363D",
    "sidebar": "#0D1117",
    "danger": "#FF4B4B",
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
            --danger: {COLORS['danger']};
            --warning: {COLORS['warning']};
            --sidebar: {COLORS['sidebar']};
        }}
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            background-color: {COLORS['bg_dark']};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar);
            border-right: 1px solid var(--border);
        }}
        
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

        .card-title {{ 
            font-size: 1.1rem; 
            font-weight: 700; 
            margin-bottom: 8px; 
            color: var(--text-main); 
            display: flex; 
            align-items: center; 
            gap: 10px; 
        }}
        .card-desc {{ 
            color: var(--text-sub); 
            font-size: 0.9rem; 
            line-height: 1.5; 
        }}
        
        .stButton button {{
            background-color: var(--bg-card); 
            color: var(--text-main);
            border: 1px solid var(--border); 
            border-radius: 8px;
            font-weight: 600; 
            padding: 0.5rem 1rem; 
            transition: all 0.2s;
        }}
        .stButton button:hover {{ 
            border-color: var(--primary); 
            color: var(--primary); 
        }}
        
        .stButton button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, #00A37E 100%);
            color: #000; 
            border: none; 
            box-shadow: 0 4px 12px rgba(0, 192, 149, 0.3);
        }}
        .stButton button[kind="primary"]:hover {{ 
            transform: scale(1.02); 
            color: #000; 
        }}

        .ai-insight-box {{
            background-color: #13161C; 
            border-left: 4px solid var(--accent);
            padding: 20px; 
            border-radius: 0 12px 12px 0; 
            margin-top: 15px;
            line-height: 1.6; 
            color: #E0E0E0; 
            font-size: 1rem;
        }}

        /* CẢI TIẾN: Stat box với hover effect tốt hơn */
        .stat-box {{
            text-align: center; 
            padding: 15px; 
            background: var(--bg-card);
            border-radius: 12px; 
            border: 1px solid var(--border);
            transition: transform 0.2s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
            display: flex; 
            flex-direction: column; 
            justify-content: center;
        }}
        .stat-box:hover {{ 
            border-color: var(--primary); 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 192, 149, 0.1);
        }}
        .stat-val {{ 
            font-size: 1.8rem; 
            font-weight: 800; 
            color: var(--primary); 
        }}
        .stat-lbl {{ 
            font-size: 0.85rem; 
            color: var(--text-sub); 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
        }}

        .list-item {{
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            padding: 10px 14px; 
            background: var(--bg-card); 
            border-radius: 8px;
            border: 1px solid var(--border); 
            margin-bottom: 8px; 
            font-size: 0.9rem;
        }}
        
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, var(--primary), var(--accent));
        }}
        
        .trend-up {{ color: var(--primary); font-weight: bold; }}
        .trend-down {{ color: var(--danger); font-weight: bold; }}
        .trend-neutral {{ color: var(--warning); font-weight: bold; }}
        
        /* CẢI TIẾN: Trending post card */
        .trending-post-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            transition: all 0.3s ease;
        }}
        .trending-post-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 4px 12px rgba(0, 192, 149, 0.1);
        }}
        .post-thumbnail-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 80px;
        }}
        .post-content h4 {{
            margin: 0 0 8px 0;
            color: var(--text-main);
        }}
        .post-content h4 a {{
            text-decoration: none;
            color: var(--text-main);
        }}
        .post-content h4 a:hover {{
            color: var(--primary);
        }}
        .post-meta {{
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-bottom: 12px;
        }}
        .post-actions {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        /* CẢI TIẾN: Tag cloud đẹp hơn */
        .tag-cloud {{
            text-align: center; 
            margin: 15px 0;
            line-height: 2.5;
        }}
        .tag {{
            display: inline-block;
            margin: 3px;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
        }}
        .tag:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        /* CẢI TIẾN: Form trong sidebar */
        .sidebar-form {{
            background: var(--bg-card);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        /* Spacing utilities */
        .spacer-sm {{ margin-top: 8px; }}
        .spacer-md {{ margin-top: 16px; }}
        .spacer-lg {{ margin-top: 24px; }}
        
        /* Link styling */
        .custom-link {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }}
        .custom-link:hover {{
            text-decoration: underline;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. AUTH SCREENS ---
def render_login_screen():
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-top: 60px; margin-bottom: 40px;">
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 10px;">💎</div>
            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(to right, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Reddit Insider AI</h1>
            <p style="color: var(--text-sub); font-size: 1.1rem; margin-top: 10px;">Nền tảng phân tích dữ liệu xã hội thông minh</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR CẢI TIẾN ---
def render_sidebar_logged_in(username, groups, logout_callback, add_group_callback, delete_group_callback):
    with st.sidebar:
        # User Profile
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px; padding: 15px; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border);">
            <div style="width: 42px; height: 42px; background: linear-gradient(135deg, var(--primary), #008F6E); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; color: black; font-weight: bold;">
                {username[0].upper()}
            </div>
            <div style="overflow: hidden;">
                <div style="font-weight: 700; color: white; font-size: 1rem;">{username}</div>
                <div style="font-size: 0.75rem; color: var(--primary);">• Online</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Menu với tên đồng bộ
        st.markdown("### 🧭 Điều Hướng")
        if st.button("🏠 Dashboard", use_container_width=True, type="secondary"):
            st.session_state.page = "Dashboard"
            st.rerun()
        if st.button("📊 Phân Tích Xu Hướng", use_container_width=True, type="secondary"):
            st.session_state.page = "Trending"
            st.rerun()
        if st.button("🔗 Phân Tích Bài Viết", use_container_width=True, type="secondary"):
            st.session_state.page = "Analysis"
            st.rerun()

        st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
        
        # CẢI TIẾN: Group Manager với form đẹp hơn
        st.markdown("### 🎯 Nhóm Theo Dõi")
        with st.expander("➕ Thêm nhóm mới", expanded=False):
            st.markdown("<div class='sidebar-form'>", unsafe_allow_html=True)
            with st.form("add_group_form", clear_on_submit=True):
                new_sub = st.text_input(
                    "Tên Subreddit", 
                    placeholder="vd: python, programming, machinelearning",
                    help="Nhập tên subreddit (không cần r/)"
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    submitted = st.form_submit_button(
                        "🎯 Thêm nhóm", 
                        use_container_width=True, 
                        type="primary"
                    )
                with col2:
                    st.form_submit_button(
                        "🔄 Làm mới", 
                        use_container_width=True,
                        type="secondary"
                    )
                
                if submitted and new_sub:
                    if add_group_callback(new_sub):
                        st.success(f"✅ Đã thêm r/{new_sub}")
                    else:
                        st.error("❌ Không thể thêm nhóm này")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if groups:
            st.markdown(f"<div style='margin: 10px 0 10px 0; color:var(--text-sub); font-size:0.8rem;'>Đang theo dõi {len(groups)} cộng đồng</div>", unsafe_allow_html=True)
            
            # CẢI TIẾN: Thêm bộ lọc và sắp xếp
            search_term = st.text_input(
                "🔍 Tìm kiếm nhóm...", 
                placeholder="Tìm theo tên subreddit",
                key="group_search"
            )
            
            filtered_groups = [
                group for group in groups 
                if search_term.lower() in group['subreddit'].lower()
            ] if search_term else groups
            
            for group in filtered_groups:
                c1, c2 = st.columns([4, 1])
                with c1: 
                    st.markdown(f"<div class='list-item'>r/{group['subreddit']}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_g_{group['id']}", type="secondary", help="Xóa nhóm"):
                        delete_group_callback(group['id'])
                        st.rerun()
        else:
            st.info("📝 Chưa có nhóm nào. Hãy thêm nhóm đầu tiên!")

        st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất", use_container_width=True): 
            logout_callback()

# --- 4. DASHBOARD COMPONENTS ---
def render_dashboard_header(username):
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="font-size: 2.2rem; margin-bottom: 10px;">Xin chào, {username}! 👋</h1>
        <p style="color: var(--text-sub); font-size: 1.1rem;">
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
        st.info("📊 Chưa có lịch sử phân tích nào. Hãy bắt đầu phân tích bài viết đầu tiên!")
        return

    # CẢI TIẾN: Thêm bộ lọc lịch sử
    col1, col2 = st.columns([2, 1])
    with col1:
        search_history = st.text_input(
            "🔍 Tìm kiếm lịch sử...",
            placeholder="Tìm theo tiêu đề hoặc URL",
            key="history_search"
        )
    
    filtered_history = [
        item for item in history 
        if search_history.lower() in item['title'].lower() or 
           search_history.lower() in item['url'].lower()
    ] if search_history else history

    if not filtered_history:
        st.info("🔍 Không tìm thấy kết quả phù hợp với từ khóa tìm kiếm.")
        return

    for item in filtered_history:
        with st.container():
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(f"""
                <div class="insider-card" style="padding: 15px; margin-bottom: 10px; min-height: auto; border-left: 3px solid var(--accent);">
                    <div style="font-weight: 600; color: var(--text-main); font-size: 1rem;">{item['title']}</div>
                    <div style="font-size: 0.85rem; color: var(--text-sub); margin-top: 6px; display: flex; gap: 15px;">
                        <span>🕒 {item['timestamp']}</span>
                        <a href="{item['url']}" target="_blank" class="custom-link">Xem bài viết gốc ↗️</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("🗑️", key=f"del_h_{item['id']}", type="secondary", help="Xóa khỏi lịch sử"):
                    delete_callback(item['id'])
                    st.rerun()

# --- 5. TREND ANALYSIS COMPONENTS CẢI TIẾN ---
def _safe_hash_data(data):
    """Hàm helper để hash data an toàn"""
    try:
        if isinstance(data, (dict, list)):
            return hashlib.md5(str(data).encode()).hexdigest()[:8]
        else:
            return hashlib.md5(str(data).encode()).hexdigest()[:8]
    except:
        return "default_key"

def render_trend_analysis(trend_data):
    """Hiển thị phân tích xu hướng với xử lý lỗi tốt hơn"""
    
    if 'error' in trend_data:
        st.error(f"❌ Lỗi phân tích: {trend_data['error']}")
        if 'message' in trend_data:
            st.info(f"💡 {trend_data['message']}")
        return

    # Kiểm tra dữ liệu bắt buộc
    if 'subreddit' not in trend_data or 'data_summary' not in trend_data:
        st.error("⚠️ Dữ liệu phân tích không đầy đủ. Vui lòng thử lại.")
        return

    st.markdown(f"## 📊 Phân Tích Xu Hướng: r/{trend_data['subreddit']}")
    
    # Header metrics với stat-box cải tiến
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{trend_data['data_summary']['total_posts_analyzed']}</div><div class='stat-lbl'>Bài viết</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{trend_data['data_summary']['total_engagement']:,}</div><div class='stat-lbl'>Engagement</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{trend_data['data_summary']['total_comments']:,}</div><div class='stat-lbl'>Bình luận</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{trend_data['analysis_period_days']}</div><div class='stat-lbl'>Ngày phân tích</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # Main layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_forecast_section(trend_data.get('forecast', {}), trend_data['subreddit'])
        render_peak_hours(trend_data.get('peak_hours', []), trend_data['subreddit'])

    with col_right:
        render_top_topics(trend_data.get('top_topics', []), trend_data['subreddit'])
        render_keywords(trend_data.get('top_keywords', []), trend_data['subreddit'])

def render_forecast_section(forecast_data, subreddit):
    """Hiển thị phần dự báo xu hướng với xử lý lỗi"""
    
    st.markdown("### 🔮 Dự Báo Xu Hướng")
    
    if 'error' in forecast_data:
        st.warning(f"📊 {forecast_data.get('message', forecast_data['error'])}")
        return

    if 'forecast' not in forecast_data or not forecast_data['forecast']:
        st.info("⏳ Đang tính toán dự báo...")
        return

    # Trend indicator
    trend_dir = forecast_data.get('trend_direction', 'Không xác định')
    trend_class = "trend-up" if "Tăng" in trend_dir else "trend-down" if "Giảm" in trend_dir else "trend-neutral"
    
    st.markdown(f"""
    <div class="insider-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.9rem; color: var(--text-sub);">Xu hướng hiện tại</div>
                <div class="{trend_class}" style="font-size: 1.2rem;">{trend_dir}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: var(--text-sub);">Độ dốc</div>
                <div style="font-size: 1.1rem; font-weight: bold;">{forecast_data.get('trend_slope', 0):.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast chart với xử lý lỗi
    try:
        forecast_df = pd.DataFrame(forecast_data['forecast'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        fig = go.Figure()
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df['predicted_upper'], forecast_df['predicted_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(79, 172, 254, 0.2)',
            line_color='rgba(255,255,255,0)',
            name=f"{forecast_data.get('confidence_interval', '80%')} CI"
        ))
        
        # Main forecast line
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
        
        # Sử dụng hàm hash an toàn
        chart_key = f"forecast_chart_{subreddit}_{_safe_hash_data(forecast_data)}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Forecast details table
        with st.expander("📋 Chi tiết dự báo"):
            display_df = forecast_df.copy()
            display_df['Engagement'] = display_df['predicted_engagement'].round(1)
            display_df['Khoảng tin cậy'] = display_df.apply(
                lambda x: f"{x['predicted_lower']:.1f} - {x['predicted_upper']:.1f}", axis=1
            )
            table_key = f"forecast_table_{subreddit}_{_safe_hash_data(forecast_data)}"
            st.dataframe(
                display_df[['date', 'Engagement', 'Khoảng tin cậy']],
                hide_index=True,
                use_container_width=True,
                key=table_key
            )
            
    except Exception as e:
        st.error(f"❌ Không thể hiển thị biểu đồ dự báo: {str(e)}")
        st.info("💡 Vui lòng kiểm tra lại dữ liệu đầu vào.")

def render_peak_hours(peak_hours, subreddit):
    """Hiển thị phân tích giờ cao điểm với xử lý lỗi"""
    if not peak_hours:
        st.info("📊 Chưa có đủ dữ liệu để phân tích giờ cao điểm.")
        return

    st.markdown("### 🕐 Giờ Hoạt Động Cao Điểm")
    
    try:
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
        
        chart_key = f"peak_hours_chart_{subreddit}_{_safe_hash_data(peak_hours)}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Top hours
        top_hours = sorted(peak_hours, key=lambda x: x['total_engagement'], reverse=True)[:3]
        for i, hour_data in enumerate(top_hours, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background: var(--bg-card); border-radius: 8px; margin: 5px 0;">
                <span>{emoji} {hour_data['hour']:02d}:00</span>
                <span style="font-weight: bold; color: var(--primary);">{hour_data['total_engagement']} engagement</span>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Không thể hiển thị biểu đồ giờ cao điểm: {str(e)}")

def render_top_topics(topics, subreddit):
    """Hiển thị các chủ đề nổi bật"""
    if not topics:
        st.info("📊 Chưa có đủ dữ liệu để phân tích chủ đề.")
        return

    st.markdown("### 🎯 Chủ Đề Nổi Bật")
    
    for i, topic in enumerate(topics[:5]):
        percentage = topic.get('percentage', 0)
        st.markdown(f"""
        <div class="insider-card" style="padding: 15px; margin-bottom: 10px;">
            <div style="font-weight: 600; margin-bottom: 8px;">{topic['name']}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem; color: var(--text-sub);">
                    {topic['frequency']} bài viết
                </span>
                <span style="font-size: 0.8rem; color: var(--primary); font-weight: bold;">
                    {percentage:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_keywords(keywords, subreddit):
    """Hiển thị từ khóa quan trọng - không dùng wordcloud"""
    if not keywords:
        st.info("📊 Chưa có đủ dữ liệu để trích xuất từ khóa.")
        return

    st.markdown("### 🔑 Từ Khóa Quan Trọng")
    
    # Tag cloud visualization cải tiến
    st.markdown('<div class="tag-cloud">', unsafe_allow_html=True)
    
    for i, kw in enumerate(keywords[:15]):
        # Calculate size and color based on score
        size = max(14, min(28, int(kw['score'] * 150)))
        
        st.markdown(
            f'<span class="tag" style="font-size: {size}px;">{kw["keyword"]}</span>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed keyword scores
    with st.expander("📊 Chi tiết điểm số"):
        for i, kw in enumerate(keywords[:8], 1):
            score_width = min(kw['score'] * 100, 100)
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>#{i} {kw['keyword']}</span>
                    <span style="font-size: 0.8rem; color: var(--text-sub);">{kw['score']:.3f}</span>
                </div>
                <div style="width: 100%; height: 4px; background: var(--border); border-radius: 2px;">
                    <div style="width: {score_width}%; height: 100%; background: var(--primary); border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 6. TRENDING CARD CẢI TIẾN ---
def render_trending_card(post, analyze_callback):
    """Hiển thị card bài viết trending với design cải tiến"""
    title = post.get('title', 'No Title')
    sub = post.get('subreddit', 'reddit')
    author = post.get('author', 'unknown')
    time_str = post.get('time_str', '')
    thumbnail = post.get('thumbnail')
    link = post.get('url')
    post_id = post.get('id', str(hash(title)))

    # Sử dụng container với class trending-post-card
    with st.container():
        st.markdown(f"<div class='trending-post-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<div class='post-thumbnail-container'>", unsafe_allow_html=True)
            if thumbnail and thumbnail.startswith('http'):
                st.image(thumbnail, use_container_width=True, output_format="PNG")
            else:
                # CẢI TIẾN: Icon placeholder lớn hơn, rõ ràng hơn
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
                    <span style="font-size: 2.5rem; margin-bottom: 8px;">📰</span>
                    <span style="font-size: 0.7rem; color: var(--text-sub); text-align: center;">No Image</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='post-content'>", unsafe_allow_html=True)
            # Tiêu đề với link
            st.markdown(f"<h4><a href='{link}' target='_blank' style='text-decoration:none; color:var(--text-main);'>{title}</a></h4>", unsafe_allow_html=True)
            # Meta information
            st.markdown(f"<div class='post-meta'>r/{sub} • {author} • {time_str}</div>", unsafe_allow_html=True)
            
            # Actions - SỬA LỖI: st.link_button không có tham số key
            st.markdown(f"<div class='post-actions'>", unsafe_allow_html=True)
            btn_analyze_key = f"btn_analyze_trend_{post_id}_{_safe_hash_data(title)}"
            if st.button("⚡ Phân tích", key=btn_analyze_key, type="primary", help="Phân tích bài viết này"):
                analyze_callback(link)
            
            # SỬA: Bỏ tham số key trong st.link_button
            st.link_button("🌐 Xem gốc ↗️", link)
            st.markdown("</div>", unsafe_allow_html=True) # Đóng post-actions
            st.markdown("</div>", unsafe_allow_html=True) # Đóng post-content

        st.markdown("</div>", unsafe_allow_html=True) # Đóng trending-post-card
        st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True) # Khoảng cách

# --- 7. ANALYSIS RESULT COMPONENTS CẢI TIẾN ---
def render_analysis_result_full(data):
    """Hiển thị kết quả phân tích bài viết với xử lý lỗi"""
    if not data or 'meta' not in data or 'df' not in data:
        st.error("❌ Dữ liệu phân tích không hợp lệ.")
        return

    meta = data['meta']
    df = data['df']
    summary = data.get('summary', 'Chưa có phân tích AI.')

    st.markdown(f"## 📄 {meta.get('title', 'Không có tiêu đề')}")
    st.caption(f"👤 Tác giả: {meta.get('author', 'Ẩn danh')} • 🏷️ Subreddit: r/{meta.get('subreddit', 'unknown')} • 💬 {meta.get('count', 0)} bình luận")

    # Metrics Overview với stat-box cải tiến
    c1, c2, c3, c4 = st.columns(4)
    total_comments = len(df) if not df.empty else 0
    avg_score = df['polarity'].mean() if not df.empty and 'polarity' in df.columns else 0
    pos_count = len(df[df['sentiment'] == 'Tích cực']) if not df.empty and 'sentiment' in df.columns else 0
    neg_count = len(df[df['sentiment'] == 'Tiêu cực']) if not df.empty and 'sentiment' in df.columns else 0
    
    with c1: 
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{total_comments}</div><div class='stat-lbl'>Bình luận</div></div>", unsafe_allow_html=True)
    with c2: 
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{avg_score:.2f}</div><div class='stat-lbl'>Điểm cảm xúc</div></div>", unsafe_allow_html=True)
    with c3: 
        st.markdown(f"<div class='stat-box'><div class='stat-val' style='color:var(--primary)'>{pos_count}</div><div class='stat-lbl'>Tích cực</div></div>", unsafe_allow_html=True)
    with c4: 
        st.markdown(f"<div class='stat-box'><div class='stat-val' style='color:var(--danger)'>{neg_count}</div><div class='stat-lbl'>Tiêu cực</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # Tabs - SỬA LỖI: st.tabs không có tham số key
    t1, t2, t3 = st.tabs(["🤖 AI Insight", "📊 Biểu Đồ", "📋 Dữ Liệu Chi Tiết"])

    with t1: 
        render_ai_summary_box(summary, meta.get('title', ''))
    with t2: 
        render_charts(df, meta.get('title', ''))
    with t3: 
        render_data_table(df, meta.get('title', ''))

def render_charts(df, title):
    """Hiển thị biểu đồ phân tích với xử lý lỗi và loading status"""
    if df.empty:
        st.info("📊 Chưa có đủ dữ liệu để vẽ biểu đồ.")
        return

    chart_colors = [COLORS['primary'], COLORS['danger'], COLORS['warning'], COLORS['accent']]
    
    with st.status("🔄 Đang tải biểu đồ...", expanded=False) as status:
        try:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("##### 🎭 Phân bố Cảm Xúc")
                if 'sentiment' in df.columns and not df['sentiment'].empty:
                    sentiment_counts = df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['sentiment', 'count']
                    fig = px.pie(sentiment_counts, names='sentiment', values='count', hole=0.6, color_discrete_sequence=chart_colors)
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)", 
                        font_color=COLORS['text_main'], 
                        showlegend=True, 
                        margin=dict(t=20, b=20, l=20, r=20), 
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                    )
                    pie_key = f"pie_chart_{_safe_hash_data(title)}"
                    st.plotly_chart(fig, use_container_width=True, key=pie_key)
                else:
                    st.caption("ℹ️ Không có dữ liệu cảm xúc để vẽ biểu đồ tròn.")
            
            with c2:
                st.markdown("##### 🌊 Diễn biến theo thời gian")
                if 'timestamp' in df.columns and 'polarity' in df.columns and len(df) > 1:
                    df_sorted = df.sort_values('timestamp')
                    df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
                    df_sorted = df_sorted.dropna(subset=['timestamp'])
                    
                    if not df_sorted.empty:
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
                        scatter_key = f"scatter_chart_{_safe_hash_data(title)}"
                        st.plotly_chart(fig2, use_container_width=True, key=scatter_key)
                    else:
                        st.caption("ℹ️ Không đủ dữ liệu hợp lệ để vẽ biểu đồ diễn biến.")
                else:
                    st.info("📊 Cần ít nhất 2 bình luận có thời gian để vẽ biểu đồ.")

            status.update(label="✅ Đã tải xong biểu đồ!", state="complete")
            
        except Exception as e:
            status.update(label="❌ Lỗi tải biểu đồ", state="error")
            st.error(f"Không thể vẽ biểu đồ do lỗi: {str(e)}")
            st.info("💡 Vui lòng kiểm tra lại dữ liệu đầu vào.")

def render_data_table(df, title):
    """Hiển thị bảng dữ liệu với xử lý lỗi"""
    if df.empty:
        st.info("📊 Không có dữ liệu để hiển thị.")
        return

    st.markdown("##### 📥 Dữ liệu chi tiết")
    
    try:
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        c1, c2 = st.columns([3, 1])
        with c2:
            download_key = f"download_{_safe_hash_data(title)}"
            st.download_button(
                label="📥 Tải xuống CSV",
                data=csv,
                file_name='reddit_data.csv',
                mime='text/csv',
                key=download_key,
                use_container_width=True,
                type="primary"
            )
        
        # Data table
        table_key = f"data_table_{_safe_hash_data(title)}"
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
            hide_index=True,
            key=table_key
        )
        
    except Exception as e:
        st.error(f"❌ Không thể hiển thị bảng dữ liệu: {str(e)}")

def render_ai_summary_box(summary_text, title):
    """Hiển thị AI Insight"""
    st.markdown("### 🤖 AI Insight")
    clean_text = summary_text.replace("```html", "").replace("```", "").replace("</div>", "")
    
    st.markdown(f"""
    <div class="ai-insight-box">
        {clean_text}
    </div>
    """, unsafe_allow_html=True)