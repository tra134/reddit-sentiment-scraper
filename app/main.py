import streamlit as st
import pandas as pd
import numpy as np
import logging
import requests
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
import random

# Optional NLP imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Reddit Analytics Ultimate",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(120deg, #ff4500, #ff8700);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(255, 69, 0, 0.3);
    }
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff4500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4500;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA ENGINE ---

class RedditLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Streamlit Analytics Pro)'})

    def fetch(self, url):
        clean_url = (url.split('?')[0] if '?' in url else url.rstrip('/')) + '.json'
        try:
            resp = self.session.get(clean_url, timeout=15)
            if resp.status_code != 200: return {'success': False, 'error': f'HTTP {resp.status_code}'}
            data = resp.json()
            
            post = data[0]['data']['children'][0]['data']
            comments = []
            
            # Recursive function to get nested comments (Limited depth)
            def process_comments(children_list):
                for item in children_list:
                    if item['kind'] == 't1':
                        d = item['data']
                        body = d.get('body', '')
                        if body and body not in ['[deleted]', '[removed]']:
                            comments.append({
                                'id': d.get('id'),
                                'author': d.get('author', 'Unknown'),
                                'body': body,
                                'score': d.get('score', 0),
                                'created_utc': d.get('created_utc'),
                                'timestamp': datetime.fromtimestamp(d.get('created_utc', 0)),
                                'depth': d.get('depth', 0)
                            })
            
            process_comments(data[1]['data']['children'])
            
            return {
                'success': True,
                'meta': {
                    'title': post.get('title'),
                    'subreddit': post.get('subreddit'),
                    'score': post.get('score'),
                    'upvote_ratio': post.get('upvote_ratio'),
                    'comments_count': post.get('num_comments'),
                    'author': post.get('author'),
                    'created': datetime.fromtimestamp(post.get('created_utc'))
                },
                'comments': comments
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# --- 2. NLP INTELLIGENCE ENGINE ---

class NLPEngine:
    def __init__(self):
        # Emotional lexicon (Simplified)
        self.emotions = {
            'Anger': {'hate', 'stupid', 'garbage', 'trash', 'idiot', 'worst', 'angry', 'mad', 'furious', 'annoying', 'buggy'},
            'Joy': {'love', 'great', 'awesome', 'happy', 'fun', 'enjoy', 'perfect', 'beautiful', 'glad', 'excited'},
            'Trust': {'secure', 'safe', 'agree', 'true', 'support', 'trust', 'recommend', 'reliable', 'solid'},
            'Fear': {'scary', 'worry', 'risk', 'dangerous', 'afraid', 'loss', 'crash', 'broke', 'hack', 'alert'},
            'Anticipation': {'wait', 'hope', 'coming', 'soon', 'expect', 'looking', 'future', 'wish', 'hyped'}
        }
        
    def analyze_text(self, text):
        # 1. Basic Sentiment
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Fallback simple logic
            polarity = 0
            subjectivity = 0.5

        # Categorize Sentiment
        if polarity > 0.1: sentiment = 'Positive'
        elif polarity < -0.1: sentiment = 'Negative'
        else: sentiment = 'Neutral'

        # 2. Emotion Detection
        detected_emotions = []
        words = set(re.findall(r'\w+', text.lower()))
        for emotion, keywords in self.emotions.items():
            if not words.isdisjoint(keywords):
                detected_emotions.append(emotion)
        
        if not detected_emotions: detected_emotions = ['Neutral']
        primary_emotion = detected_emotions[0]

        # 3. Stats
        word_count = len(text.split())
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotions': detected_emotions,
            'primary_emotion': primary_emotion,
            'word_count': word_count
        }

    def process_batch(self, comments):
        processed = []
        for c in comments:
            nlp_res = self.analyze_text(c['body'])
            c.update(nlp_res)
            processed.append(c)
        return processed

# --- 3. VISUALIZATION ENGINE ---

class VizEngine:
    @staticmethod
    def plot_radar_emotions(df):
        """Vẽ biểu đồ mạng nhện cho cảm xúc"""
        all_emotions = [e for sublist in df['emotions'] for e in sublist if e != 'Neutral']
        if not all_emotions: return None
        
        counts = Counter(all_emotions)
        categories = list(counts.keys())
        values = list(counts.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Emotions',
            line_color='#ff4500'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            title="Emotional Footprint"
        )
        return fig

    @staticmethod
    def plot_sunburst(df):
        """Vẽ biểu đồ Sunburst: Sentiment -> Emotion"""
        # Prepare data
        df_explode = df.explode('emotions')
        # Filter out purely neutral paths to make chart interesting
        df_explode = df_explode[df_explode['sentiment'] != 'Neutral']
        
        if df_explode.empty: return None
        
        fig = px.sunburst(
            df_explode, 
            path=['sentiment', 'emotions'],
            color='sentiment',
            color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#FECB52'},
            height=400,
            title="Sentiment to Emotion Hierarchy"
        )
        return fig

    @staticmethod
    def plot_sentiment_timeline(df):
        """Biểu đồ đường thời gian với đường trung bình động"""
        df = df.sort_values('timestamp')
        df['MA'] = df['polarity'].rolling(window=max(1, int(len(df)/10))).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['polarity'], mode='markers', name='Comment', marker=dict(opacity=0.3, color='#888')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA'], mode='lines', name='Trend', line=dict(color='#ff4500', width=3)))
        
        fig.update_layout(
            title="Sentiment Trend Over Time",
            yaxis_title="Polarity (-1 to 1)",
            height=350,
            showlegend=False
        )
        return fig

    @staticmethod
    def plot_length_boxplot(df):
        """Boxplot so sánh độ dài comment"""
        fig = px.box(
            df, x='sentiment', y='word_count', 
            color='sentiment',
            color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#FECB52'},
            points="outliers",
            title="Comment Length Distribution by Sentiment"
        )
        fig.update_layout(height=350)
        return fig

# --- MAIN APP ---

def main():
    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("### Analysis Settings")
        
        min_words = st.slider("Min Word Count Filter", 0, 50, 0)
        remove_stops = st.checkbox("Exclude Stopwords", value=True)
        
        st.divider()
        st.markdown("### History")
        if 'history' not in st.session_state: st.session_state.history = []
        for h in st.session_state.history[-5:][::-1]:
            if st.button(f"r/{h['sub']}: {h['title'][:15]}...", key=h['id']):
                st.session_state.url = h['url']
                st.rerun()

    # --- INPUT SECTION ---
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Reddit Analytics Ultimate</h1>
        <p>Advanced NLP • Emotional Intelligence • Temporal Trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        default_url = st.session_state.get('url', '')
        url = st.text_input("Reddit Thread URL", value=default_url, placeholder="https://reddit.com/r/...")
    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 IGNITE", type="primary", use_container_width=True)

    # --- PROCESS ---
    if analyze_btn and url:
        st.session_state.url = url
        loader = RedditLoader()
        nlp = NLPEngine()
        
        with st.status("🔍 Processing Data Pipeline...", expanded=True) as status:
            st.write("📡 Connecting to Reddit JSON Node...")
            raw_data = loader.fetch(url)
            
            if not raw_data['success']:
                status.update(label="❌ Failed", state="error")
                st.error(raw_data['error'])
                return
            
            st.write(f"📥 Downloaded {len(raw_data['comments'])} comments. Ingesting to NLP Engine...")
            processed_comments = nlp.process_batch(raw_data['comments'])
            
            # Create DataFrame
            df = pd.DataFrame(processed_comments)
            # Filter
            df = df[df['word_count'] >= min_words]
            
            # Save History
            hist_entry = {'id': str(time.time()), 'url': url, 'title': raw_data['meta']['title'], 'sub': raw_data['meta']['subreddit']}
            if hist_entry['title'] not in [x['title'] for x in st.session_state.history]:
                st.session_state.history.append(hist_entry)
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # --- DASHBOARD LAYOUT ---
        meta = raw_data['meta']
        
        # 1. EXECUTIVE SUMMARY (KPIs)
        st.markdown("### 🏆 Executive Summary")
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            st.metric("Total Engagement", f"{meta['score']:,}", delta=f"{meta['upvote_ratio']*100:.0f}% Ratio")
        with k2:
            st.metric("Comments Analyzed", len(df), delta=f"{len(df)/meta['comments_count']*100:.1f}% of Total")
        with k3:
            avg_pol = df['polarity'].mean()
            st.metric("Avg Polarity", f"{avg_pol:.2f}", delta="Positive" if avg_pol > 0 else "Negative")
        with k4:
            # Subjectivity KPI
            avg_sub = df['subjectivity'].mean()
            label = "Opinionated" if avg_sub > 0.5 else "Objective"
            st.metric("Tone", label, delta=f"{avg_sub:.2f}")

        # 2. TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Macro View", "🧠 Emotional AI", "⏳ Temporal", "☁️ Linguistics", "🔬 Micro Explorer"
        ])
        
        # TAB 1: MACRO VIEW (General Sentiment)
        with tab1:
            c1, c2 = st.columns([1, 2])
            with c1:
                # Donut Chart
                sentiment_counts = df['sentiment'].value_counts()
                fig_donut = px.pie(
                    values=sentiment_counts.values, names=sentiment_counts.index, 
                    hole=0.6, color=sentiment_counts.index,
                    color_discrete_map={'Positive': '#00cc96', 'Negative': '#ef553b', 'Neutral': '#fecb52'},
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            with c2:
                # Sunburst
                st.plotly_chart(VizEngine.plot_sunburst(df), use_container_width=True)
        
        # TAB 2: EMOTIONAL AI (Deep Dive)
        with tab2:
            st.info("💡 **AI Analysis:** Detecting nuances like Anger, Joy, Trust beyond simple Positive/Negative.")
            c1, c2 = st.columns(2)
            with c1:
                # Radar Chart
                radar = VizEngine.plot_radar_emotions(df)
                if radar: st.plotly_chart(radar, use_container_width=True)
                else: st.warning("Not enough emotional data.")
            with c2:
                # Box Plot (Length vs Sentiment)
                st.plotly_chart(VizEngine.plot_length_boxplot(df), use_container_width=True)

        # TAB 3: TEMPORAL (Time Series)
        with tab3:
            st.plotly_chart(VizEngine.plot_sentiment_timeline(df), use_container_width=True)
            
            # Heatmap of Activity
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df['hour'].value_counts().sort_index()
            fig_bar = px.bar(x=hourly_counts.index, y=hourly_counts.values, labels={'x':'Hour of Day', 'y':'Volume'}, title="Comment Volume by Hour")
            st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 4: LINGUISTICS (Word Cloud & Entities)
        with tab4:
            if WORDCLOUD_AVAILABLE:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("##### Word Cloud (Positive vs Negative)")
                    # Generate Positive Word Cloud
                    pos_text = " ".join(df[df['sentiment']=='Positive']['body'])
                    if pos_text:
                        wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
                        st.image(wc.to_array(), caption="Positive Vocabulary")
                with c2:
                    st.markdown("##### Top Negative Keywords")
                    neg_text = " ".join(df[df['sentiment']=='Negative']['body'])
                    if neg_text:
                        words = [w for w in re.findall(r'\w+', neg_text.lower()) if len(w)>3]
                        common = Counter(words).most_common(10)
                        st.dataframe(pd.DataFrame(common, columns=['Word', 'Count']), hide_index=True)
            else:
                st.warning("Install `wordcloud` to see this tab.")

        # TAB 5: MICRO EXPLORER (Data Grid)
        with tab5:
            # Filters
            f1, f2, f3 = st.columns(3)
            with f1: f_sent = st.multiselect("Filter Sentiment", ['Positive', 'Negative', 'Neutral'], default=['Positive', 'Negative'])
            with f2: f_emo = st.multiselect("Filter Emotion", list(nlp.emotions.keys()))
            with f3: sort = st.selectbox("Sort By", ["Score (High-Low)", "Polarity (Most Neg)", "Polarity (Most Pos)", "Newest"])
            
            # Apply Filters
            filtered_df = df[df['sentiment'].isin(f_sent)]
            if f_emo:
                # Filter if emotions list contains selected emotion
                filtered_df = filtered_df[filtered_df['emotions'].apply(lambda x: any(e in f_emo for e in x))]
            
            # Sort
            if sort == "Score (High-Low)": filtered_df = filtered_df.sort_values('score', ascending=False)
            elif sort == "Polarity (Most Neg)": filtered_df = filtered_df.sort_values('polarity', ascending=True)
            elif sort == "Polarity (Most Pos)": filtered_df = filtered_df.sort_values('polarity', ascending=False)
            elif sort == "Newest": filtered_df = filtered_df.sort_values('timestamp', ascending=False)
            
            st.markdown(f"**Showing {len(filtered_df)} comments**")
            
            for _, row in filtered_df.head(50).iterrows():
                # Dynamic Badge Color
                badge_color = "#00cc96" if row['sentiment']=='Positive' else "#ef553b" if row['sentiment']=='Negative' else "#fecb52"
                
                st.markdown(f"""
                <div style="background:white; padding:15px; border-radius:10px; margin-bottom:10px; border-left: 5px solid {badge_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="display:flex; justify-content:space-between;">
                        <b>u/{row['author']}</b>
                        <span style="background:{badge_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.8em;">{row['sentiment']} ({row['polarity']:.2f})</span>
                    </div>
                    <div style="margin:5px 0; font-size:0.9em; color:#666;">
                        Emotion: <em>{', '.join(row['emotions'])}</em> • Subjectivity: {row['subjectivity']:.2f}
                    </div>
                    <div style="color:#333; line-height:1.5;">{row['body']}</div>
                    <div style="margin-top:5px; font-size:0.8em; color:#888;">
                        ⬆️ {row['score']} • 📅 {row['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📥 Export Raw Data"):
                st.dataframe(filtered_df)

if __name__ == "__main__":
    main()