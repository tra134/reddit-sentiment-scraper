🧠 Reddit Analytics Pro - Enterprise Social Listening Tool

Reddit Analytics Pro is an in-depth social listening platform designed for Reddit. This project allows for the collection and analysis of user sentiment, trends, and behavior without the need for a Reddit API Key, helping to overcome barriers related to request limits and costs.

✨ Key Features

1. 🛡️ Enterprise User System

Authentication: Secure registration and login with password hashing (Bcrypt).

Personalization: Manage a list of favorite Subreddits for each account.

History Tracking: Save personal analysis history for later review.

2. 🔍 Smart Data Collection Technology

No-API Mode: Uses JSON simulation techniques to retrieve real data from Reddit without requiring a Developer account or API Key.

Auto Discovery: Automatically scans and suggests "Hot" posts from tracked user groups.

3. 🧠 Advanced AI Analysis (NLP)

Sentiment Analysis: Classifies sentiment (Positive, Negative, Neutral) using VADER/TextBlob.

Emotion Detection: Identifies specific emotions (Joy, Anger, Sadness, Surprise...).

Intent Classification: Classifies user intent (Questions, Complaints, Appreciation, News).

Topic Modeling: Analyzes phrases (Bigrams) and generates WordClouds to capture main themes.

4. 📊 Interactive Data Visualization

Real-time Charts: Interactive charts using Plotly (Pie charts, Bar charts, Trend lines).

Radar Chart: Spider web charts analyzing emotional footprints.

Time-series Analysis: Tracks discussion evolution over time.

Data Export: Exports data reports to CSV files.

📂 Project Structure

The project is built using the MVC / Service Layer pattern to ensure scalability and maintainability.

reddit-sentiment-scraper/
├── app/
│   ├── core/               # Database & Auth Configuration
│   ├── models/             # Data Structure Definitions (SQLAlchemy ORM)
│   ├── services/           # Business Logic (User, Trend Service)
│   ├── scrapers/           # Reddit Data Collection Core
│   └── main.py             # Main Interface (Streamlit Entry point)
├── data/                   # SQLite Database Storage (.db)
├── requirements.txt        # Required Libraries
└── README.md               # Documentation


🚀 Installation Guide

1. Clone Repository

git clone [https://github.com/tra134/reddit-sentiment-scraper.git](https://github.com/tra134/reddit-sentiment-scraper.git)
cd reddit-sentiment-scraper


2. Create Virtual Environment (Recommended)

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements*.txt


4. Run Application

Note: The main execution file is located in the app folder.

streamlit run app/main.py


Once running, the browser will automatically open at: http://localhost:8501

📖 User Guide

Register Account:

On the main screen, select the Register tab.

Enter a Username and Password to create an account (Data is stored locally in data/database/user_data.db).

Manage Groups (My Feed):

After logging in, open the left Sidebar.

Enter a Subreddit name (e.g., technology, bitcoin, vietnam) and click Add.

Click the Refresh Feed button to view hot posts from these groups.

Analyze Posts:

Method 1: Click the Analyze button directly on a post in the feed.

Method 2: Copy any Reddit post link, paste it into the Single Analysis tab, and click run.

View Reports:

The system will display a Dashboard with tabs for: Overview, Emotions, Trends, and Comment Details.

🛠️ Tech Stack

Frontend: Streamlit

Backend Logic: Python

Database: SQLite & SQLAlchemy

NLP & AI: NLTK, TextBlob

Visualization: Plotly, WordCloud

Data Handling: Pandas, NumPy

🤝 Contributing

Contributions are welcome! If you want to improve this project:

Fork this repository.

Create a new feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License

This project is distributed under the MIT License. See the LICENSE file for more details.
