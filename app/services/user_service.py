# app/services/user_service.py
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from typing import List, Optional, Dict, Any
from app.core.user_database import User, UserGroup, UserPreference, user_db_manager
from app.core.auth import get_password_hash, verify_password, authenticate_user
import logging

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self, db: Session = None):
        self.db = db or user_db_manager.get_session()

    def create_user(self, username: str, email: str, password: str, full_name: str = None) -> User:
        """Create new user with comprehensive error handling"""
        logger.info(f"Attempting to create user: {username}")
        
        # Validation
        if not username or not email or not password:
            raise ValueError("Username, email and password are required")
        
        username = username.strip()
        email = email.lower().strip()
        
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        if "@" not in email:
            raise ValueError("Invalid email format")
        
        try:
            # Check existing user
            existing = self.db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing:
                if existing.username == username:
                    raise ValueError(f"Username '{username}' already exists")
                else:
                    raise ValueError(f"Email '{email}' already registered")
            
            # Create user
            hashed_password = get_password_hash(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name.strip() if full_name else None,
                is_active=True
            )
            
            self.db.add(user)
            self.db.flush()
            
            # Create preferences
            preferences = UserPreference(user_id=user.id)
            preferences.set_preferences({
                "theme": "light",
                "notifications": True,
                "language": "en"
            })
            self.db.add(preferences)
            
            self.db.commit()
            logger.info(f"User created successfully: {username} (ID: {user.id})")
            return user
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity error: {e}")
            raise ValueError("User registration failed - duplicate data")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error: {e}")
            raise ValueError(f"Registration failed: {str(e)}")

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        return authenticate_user(self.db, username, password)

    # THÊM METHOD GET_USER_GROUPS
    def get_user_groups(self, user_id: int) -> List[UserGroup]:
        """Get user's active interest groups"""
        try:
            return self.db.query(UserGroup).filter(
                UserGroup.user_id == user_id,
                UserGroup.is_active == True
            ).all()
        except Exception as e:
            logger.error(f"Error getting user groups: {e}")
            return []

    # THÊM METHOD ADD_USER_GROUP
    def add_user_group(self, user_id: int, group_name: str, subreddit: str) -> UserGroup:
        """Add new interest group for user"""
        try:
            # Check if group already exists
            existing_group = self.db.query(UserGroup).filter(
                UserGroup.user_id == user_id,
                UserGroup.subreddit == subreddit
            ).first()
            
            if existing_group:
                existing_group.is_active = True
                existing_group.group_name = group_name
                self.db.commit()
                return existing_group
            
            # Create new group
            group = UserGroup(
                user_id=user_id,
                group_name=group_name,
                subreddit=subreddit
            )
            
            self.db.add(group)
            self.db.commit()
            self.db.refresh(group)
            return group
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error adding user group: {e}")
            raise

    # THÊM METHOD REMOVE_USER_GROUP
    def remove_user_group(self, user_id: int, group_id: int) -> bool:
        """Remove user interest group (soft delete)"""
        try:
            group = self.db.query(UserGroup).filter(
                UserGroup.id == group_id,
                UserGroup.user_id == user_id
            ).first()
            
            if group:
                group.is_active = False
                self.db.commit()
                return True
            return False
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error removing user group: {e}")
            return False

    # THÊM METHOD UPDATE_USER_PREFERENCES
    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> UserPreference:
        """Update user preferences"""
        try:
            user_pref = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).first()
            
            if not user_pref:
                user_pref = UserPreference(user_id=user_id)
                user_pref.set_preferences(preferences)
                self.db.add(user_pref)
            else:
                current_prefs = user_pref.get_preferences()
                current_prefs.update(preferences)
                user_pref.set_preferences(current_prefs)
            
            self.db.commit()
            self.db.refresh(user_pref)
            return user_pref
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating user preferences: {e}")
            raise