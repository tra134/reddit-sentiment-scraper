from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

# Tạo base riêng cho user models để không xung đột
UserBase = declarative_base()

class User(UserBase):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserGroup(UserBase):
    __tablename__ = "user_groups"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    group_name = Column(String(200), nullable=False)
    subreddit = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "group_name": self.group_name,
            "subreddit": self.subreddit,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class UserPreference(UserBase):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    preferences = Column(Text, default='{}')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_preferences(self):
        return json.loads(self.preferences)
    
    def set_preferences(self, prefs_dict):
        self.preferences = json.dumps(prefs_dict)

class UserDatabaseManager:
    """Database manager riêng cho user data để không ảnh hưởng database gốc"""
    
    def __init__(self):
        # Sử dụng database riêng cho user data
        self.database_url = os.getenv("USER_DATABASE_URL", "sqlite:///./data/database/user_data.db")
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create user tables"""
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs("./data/database", exist_ok=True)
            UserBase.metadata.create_all(bind=self.engine)
            print("✅ User database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating user tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

# Global user database instance
user_db_manager = UserDatabaseManager()