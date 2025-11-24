import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def render_dashboard(post_info, df_comments, stats):
    st.markdown(f"### {post_info['title']}")
    
    # 1. Header Metrics
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng bình luận", stats['total'])
        c2.metric("Upvotes bài viết", post_info['score'])
        
        # Icon cảm xúc
        score = stats['avg_score']
        icon = "😐"
        if score > 0.1: icon = "😃"
        elif score < -0.1: icon = "😡"
        c3.metric("Điểm Cảm xúc TB", f"{score:.3f} {icon}")

    # Nút Download
    st.download_button(
        "📥 Tải dữ liệu (CSV)", 
        df_comments.to_csv(index=False).encode('utf-8'), 
        "reddit_data.csv", 
        "text/csv"
    )

    # --- BỘ LỌC ---
    with st.expander("⚡ Bộ lọc & Tìm kiếm", expanded=False):
        c1, c2 = st.columns(2)
        if not df_comments.empty:
            min_val = int(df_comments['score'].min())
            max_val = int(df_comments['score'].max())
            # Fix lỗi slider nếu min == max
            if min_val < max_val:
                min_score = c1.slider("Vote tối thiểu", min_val, max_val, min_val)
            else:
                c1.info(f"Tất cả comment đều có {min_val} vote")
                min_score = min_val
        else:
            min_score = 0
        search = c2.text_input("Tìm từ khóa")
    
    # Áp dụng bộ lọc
    if not df_comments.empty:
        filtered_df = df_comments[df_comments['score'] >= min_score]
        if search: 
            filtered_df = filtered_df[filtered_df['original_text'].str.contains(search, case=False)]
    else:
        filtered_df = df_comments

    if filtered_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        return

    # --- VISUALIZATION TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tổng quan", "📈 Xu hướng", "👥 Cộng đồng", "🔠 Chủ đề"])

    with tab1:
        c1, c2 = st.columns(2)
        
        # --- FIX LỖI TẠI ĐÂY: Thay px.donut bằng px.pie với tham số hole ---
        fig_pie = px.pie(
            names=filtered_df['sentiment'].value_counts().index, 
            values=filtered_df['sentiment'].value_counts().values, 
            hole=0.4, # Tạo lỗ rỗng để thành donut
            title="Tỷ lệ Cảm xúc",
            color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'}
        )
        c1.plotly_chart(fig_pie, use_container_width=True)
        
        # Biểu đồ Emotion Bar
        if 'emotion' in filtered_df.columns:
            emo = filtered_df['emotion'].value_counts().head(5)
            if not emo.empty:
                fig_bar = px.bar(x=emo.values, y=emo.index, orientation='h', title="Cảm xúc chi tiết", labels={'x':'Số lượng', 'y':'Cảm xúc'})
                c2.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("##### 📉 Diễn biến thảo luận theo giờ")
        if 'hour_str' in filtered_df.columns:
            trend = filtered_df.groupby('hour_str').agg({'compound_score':'mean', 'text':'count'}).reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(x=trend['hour_str'], y=trend['text'], name='Volume', marker_color='lightgray'))
            fig_trend.add_trace(go.Scatter(x=trend['hour_str'], y=trend['compound_score'], name='Sentiment', yaxis='y2', line=dict(color='orange', width=3)))
            fig_trend.update_layout(yaxis2=dict(overlaying='y', side='right', range=[-1, 1]), title="Sentiment vs Volume")
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.markdown("##### 🏆 Top User")
        user_stats = filtered_df.groupby('author').agg({'text':'count', 'score':'sum'}).reset_index()
        c_a, c_b = st.columns(2)
        with c_a:
            st.info("🗣️ Comment nhiều nhất")
            st.dataframe(user_stats.sort_values(by='text', ascending=False).head(5), hide_index=True, use_container_width=True)
        with c_b:
            st.success("⭐ Được Vote nhiều nhất")
            st.dataframe(user_stats.sort_values(by='score', ascending=False).head(5), hide_index=True, use_container_width=True)

    with tab4:
        st.markdown("##### ☁️ Word Cloud & Bigrams")
        text = " ".join(filtered_df['clean_text'])
        if text:
            wc = WordCloud(width=600, height=300, background_color='#111', colormap='Wistia').generate(text)
            fig, ax = plt.subplots(facecolor='#111')
            ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
            st.pyplot(fig)
            
            # Bigrams
            from collections import Counter
            from nltk import ngrams
            tokens = text.split()
            stops = ['the', 'is', 'to', 'and', 'of', 'it', 'in', 'for', 'that', 'this']
            tokens = [t for t in tokens if t not in stops and len(t) > 2]
            bigrams = Counter(ngrams(tokens, 2)).most_common(10)
            
            if bigrams:
                phrases = [f"{x[0]} {x[1]}" for x, _ in bigrams]
                counts = [x[1] for _, x in bigrams]
                fig_bi = px.bar(x=counts, y=phrases, orientation='h', title="Cụm từ phổ biến")
                st.plotly_chart(fig_bi, use_container_width=True)