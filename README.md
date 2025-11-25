Reddit Sentiment Scraper – Enterprise Social Listening Tool

Reddit Sentiment Scraper is an in-depth social listening platform designed for Reddit. This project allows for the collection and analysis of user sentiment, trends, and behavior without requiring a Reddit API Key, helping bypass rate limits and API costs.

Key Features
1. Enterprise User System

-Authentication: Secure registration & login with Bcrypt password hashing.

-Personalization: Manage a personal list of favorite Subreddits.

-History Tracking: Automatically saves user analysis history.

2. Smart Data Collection Technology

-No-API Mode: Uses JSON simulation techniques to retrieve real Reddit data without API keys.

-Auto Discovery: Automatically scans and suggests hot posts from tracked groups.

3. Advanced AI Analysis (NLP)

.Sentiment Analysis: Positive, Negative, Neutral.

.Emotion Detection: Joy, Anger, Sadness, Surprise, etc.

.Intent Classification: Questions, Complaints, Appreciation, News.

.Topic Modeling: Bigrams, WordCloud, theme extraction.

4. Interactive Data Visualization

Charts: Pie charts, bar charts, line charts (Plotly).

Radar Chart: Emotional analysis.

Time-series Analysis: Discussion evolution.

Data Export: Export results to CSV.

Project Structure

The project follows the MVC / Service Layer architecture.

```bash
reddit-sentiment-scraper/
├── app/
│   ├── core/               # Database & Auth configuration
│   ├── models/             # SQLAlchemy ORM models
│   ├── services/           # Business logic layer
│   ├── scrapers/           # Reddit data collection
│   └── main.py             # Streamlit entry point
├── data/                   # SQLite database (.db)
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

Installation Guide
1. Clone Repository
```Bash
git clone https://github.com/tra134/reddit-sentiment-scraper.git
cd reddit-sentiment-scraper
```

2. Create Virtual Environment (Recommended)

Windows
```Bash

python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux
Bash
```
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```Bash
pip install -r requirements*.txt
```
4. Run Application

(Entry file is inside the app directory)
```Bash
streamlit run app/main.py
```

After running, open:
http://localhost:8501

User Guide
Register Account

Open the Register tab.
Enter username + password.
Data is stored locally at data/database/user_data.db.
Manage Groups (My Feed)
Log in → open the sidebar.
Enter a Subreddit name and click Add.
Click Refresh Feed to fetch hot posts.

Analyze Posts
Option 1: Click the Analyze button on any post.
Option 2: Paste any Reddit post link into Single Analysis tab and run.


View Reports
Dashboard includes:
Overview
Emotions
Trends
Comment Details


Tech Stack
Frontend: Streamlit
Backend: Python
Database: SQLite + SQLAlchemy
NLP: NLTK, TextBlob
Visualization: Plotly, WordCloud
Data Handling: Pandas, NumPy


License
Distributed under the MIT License.
See the LICENSE file for details.
