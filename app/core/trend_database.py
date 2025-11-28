# app/core/trend_database.py
import sqlite3
import hashlib
import json
from datetime import datetime
import logging

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendDBManager:
    def __init__(self, db_name="reddit_analytics.db"):
        """Khởi tạo kết nối Database SQLite"""
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại"""
        c = self.conn.cursor()
        
        # 1. Bảng Users (Người dùng)
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT
            )
        ''')

        # 2. Bảng User Groups (Nhóm theo dõi)
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                subreddit TEXT NOT NULL,
                added_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(user_id, subreddit)
            )
        ''')

        # 3. Bảng Analysis History (Lịch sử phân tích Single Post)
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                url TEXT,
                summary TEXT,
                sentiment_score REAL,
                timestamp TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        # 4. Bảng Trend Snapshots (Lưu trữ xu hướng theo ngày để dự đoán)
        # Bảng này dùng cho TrendAnalysisService để lưu mốc lịch sử
        c.execute('''
            CREATE TABLE IF NOT EXISTS trend_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subreddit TEXT,
                date TEXT,
                total_posts INTEGER,
                avg_sentiment REAL,
                engagement_score INTEGER,
                top_keywords TEXT, -- Lưu dưới dạng JSON string
                recorded_at TEXT
            )
        ''')
        
        self.conn.commit()

    # --- AUTHENTICATION ---

    def register_user(self, username, password):
        """Đăng ký người dùng mới"""
        c = self.conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)", 
                      (username, hashed, created_at))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False # Username đã tồn tại

    def login_user(self, username, password):
        """Kiểm tra đăng nhập"""
        c = self.conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        c.execute("SELECT id, username FROM users WHERE username=? AND password_hash=?", (username, hashed))
        return c.fetchone() # Trả về (id, username) hoặc None

    # --- GROUP MANAGEMENT ---

    def add_group(self, user_id, subreddit):
        """Thêm subreddit vào danh sách theo dõi"""
        c = self.conn.cursor()
        clean_sub = subreddit.replace('r/', '').replace('/', '').strip()
        if not clean_sub: return False
        
        added_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            c.execute("INSERT INTO user_groups (user_id, subreddit, added_at) VALUES (?, ?, ?)", 
                      (user_id, clean_sub, added_at))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False # Đã tồn tại

    def get_user_groups(self, user_id):
        """Lấy danh sách nhóm của user"""
        c = self.conn.cursor()
        c.execute("SELECT id, subreddit FROM user_groups WHERE user_id=? ORDER BY added_at DESC", (user_id,))
        rows = c.fetchall()
        return [{'id': r[0], 'subreddit': r[1]} for r in rows]

    def remove_group(self, group_id):
        """Xóa nhóm khỏi danh sách"""
        c = self.conn.cursor()
        c.execute("DELETE FROM user_groups WHERE id=?", (group_id,))
        self.conn.commit()

    # --- HISTORY MANAGEMENT ---

    def log_analysis(self, user_id, title, url, summary=None, sentiment_score=0.0):
        """Lưu lịch sử phân tích bài viết"""
        c = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Tránh lưu trùng lặp liên tiếp
        c.execute("SELECT id FROM analysis_history WHERE user_id=? AND url=? ORDER BY id DESC LIMIT 1", (user_id, url))
        if not c.fetchone():
            c.execute("""
                INSERT INTO analysis_history (user_id, title, url, summary, sentiment_score, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, title, url, summary, sentiment_score, ts))
            self.conn.commit()

    def get_user_history(self, user_id, limit=20):
        """Lấy lịch sử phân tích"""
        c = self.conn.cursor()
        c.execute("SELECT id, title, url, timestamp, sentiment_score FROM analysis_history WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
        rows = c.fetchall()
        return [
            {'id': r[0], 'title': r[1], 'url': r[2], 'timestamp': r[3], 'sentiment': r[4]} 
            for r in rows
        ]

    def delete_history_item(self, hist_id):
        """Xóa một mục lịch sử"""
        c = self.conn.cursor()
        c.execute("DELETE FROM analysis_history WHERE id=?", (hist_id,))
        self.conn.commit()

    # --- TREND DATA FOR PREDICTION (NEW) ---

    def save_daily_trend_snapshot(self, subreddit, metrics):
        """
        Lưu snapshot xu hướng (dùng để vẽ biểu đồ dài hạn và dự đoán).
        Hàm này nên được gọi mỗi khi User bấm 'Quét Tin' thành công.
        """
        c = self.conn.cursor()
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Chuyển list keywords thành JSON string để lưu
        keywords_json = json.dumps(metrics.get('top_topics', []))
        
        # Kiểm tra xem hôm nay đã lưu chưa (để tránh duplicate data point cho biểu đồ)
        c.execute("SELECT id FROM trend_snapshots WHERE subreddit=? AND date=?", (subreddit, date_str))
        if not c.fetchone():
            c.execute("""
                INSERT INTO trend_snapshots (subreddit, date, total_posts, avg_sentiment, engagement_score, top_keywords, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                subreddit, 
                date_str, 
                metrics.get('total_posts_analyzed', 0),
                metrics.get('avg_sentiment', 0.0), # Cần tính toán từ service
                metrics.get('total_engagement', 0),
                keywords_json,
                datetime.now().strftime("%H:%M:%S")
            ))
            self.conn.commit()

    def get_historical_trends(self, subreddit, limit_days=30):
        """Lấy dữ liệu lịch sử để Service thực hiện dự đoán (Linear Regression)"""
        c = self.conn.cursor()
        c.execute("""
            SELECT date, total_posts, engagement_score, avg_sentiment 
            FROM trend_snapshots 
            WHERE subreddit=? 
            ORDER BY date ASC 
            LIMIT ?
        """, (subreddit, limit_days))
        
        rows = c.fetchall()
        return [
            {'date': r[0], 'total_posts': r[1], 'engagement': r[2], 'sentiment': r[3]}
            for r in rows
        ]

# Khởi tạo instance singleton để dùng trong app
trend_db = TrendDBManager()