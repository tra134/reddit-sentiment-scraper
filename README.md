Reddit Sentiment Scraper - Enterprise Social Listening Tool

Reddit Analytics Pro is an in-depth social listening platform designed for Reddit. This project enables the collection and analysis of user sentiment, trends, and behaviors without requiring a Reddit API Key, helping overcome request limits and API cost barriers.

Key Features
1. Enterprise User System

Secure registration and login with password hashing (Bcrypt)

Manage a personalized list of favorite subreddits

Save analysis history for later review

2. Smart Data Collection Technology

Retrieve Reddit data using a No-API JSON simulation technique (no Developer account required)

Automatically discover and recommend trending posts from tracked groups

3. Advanced AI Analysis (NLP)

Sentiment classification: Positive, Negative, Neutral

Emotion detection: Joy, Anger, Sadness, Surprise, etc.

Intent classification: Questions, Complaints, Appreciation, News

Topic modeling: Extract bigrams and generate word clouds

4. Interactive Data Visualization

Real-time interactive charts using Plotly

Radar charts for emotion distribution

Time-series analysis to track discussion trends

Export analysis results to CSV files

Project Structure

The project follows an MVC / Service Layer architecture for scalability and maintainability.

reddit-sentiment-scraper/
├── app/
│   ├── core/               # Database and authentication configuration
│   ├── models/             # SQLAlchemy ORM models
│   ├── services/           # Business logic (User Service, Trend Service)
│   ├── scrapers/           # Reddit data collection engine
│   └── main.py             # Streamlit entry point
├── data/                   # SQLite database storage (.db)
├── requirements.txt        # Required libraries
└── README.md               # Documentation

Installation Guide
1. Clone the repository
git clone https://github.com/tra134/reddit-sentiment-scraper.git
cd reddit-sentiment-scraper

2. Create a virtual environment (recommended)

Windows:

python -m venv venv
venv\Scripts\activate


macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the application

(The main Streamlit entry point is inside the app folder.)

streamlit run app/main.py


The browser will automatically open at:

http://localhost:8501

User Guide
Register an account

On the main screen, select the Register tab

Enter a username and password

User data is stored locally in: data/database/user_data.db

Manage Subreddit Groups (My Feed)

After logging in, open the Sidebar

Enter a subreddit name (example: technology, bitcoin, vietnam) and click Add

Click Refresh Feed to load trending posts

Analyze posts

Method 1: Click the Analyze button directly on a post
Method 2: Paste any Reddit post URL into the Single Analysis tab and click Run

View reports

The dashboard provides:

Overview

Emotions

Trends

Comment Details

Tech Stack

Frontend: Streamlit

Backend: Python

Database: SQLite with SQLAlchemy

NLP: NLTK, VADER, TextBlob

Visualization: Plotly, WordCloud

Data Processing: Pandas, NumPy

Contributing

Contributions are welcome.

Fork this repository

Create a feature branch:

git checkout -b feature/AmazingFeature


Commit your changes:

git commit -m "Add AmazingFeature"


Push your branch:

git push origin feature/AmazingFeature


Open a Pull Request

License

This project is licensed under the MIT License. See the LICENSE file for details.
