import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimpleAuth:
    """Simple authentication system for the application"""
    
    def __init__(self):
        self.sessions = {}
        self.api_keys = {}
    
    def generate_api_key(self, name: str) -> str:
        """Generate a new API key"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'name': name,
            'created_at': datetime.now(),
            'last_used': None,
            'usage_count': 0
        }
        logger.info(f"🔑 Generated API key for: {name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['last_used'] = datetime.now()
            self.api_keys[api_key]['usage_count'] += 1
            return True
        return False
    
    def create_session(self, user_data: Dict[str, Any]) -> str:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(16)
        self.sessions[session_id] = {
            'user_data': user_data,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return user data"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Check if session is expired (24 hours)
            if datetime.now() - session['created_at'] > timedelta(hours=24):
                del self.sessions[session_id]
                return None
            
            session['last_activity'] = datetime.now()
            return session['user_data']
        return None
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_keys = []
        for session_id, session in self.sessions.items():
            if datetime.now() - session['created_at'] > timedelta(hours=24):
                expired_keys.append(session_id)
        
        for key in expired_keys:
            del self.sessions[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired sessions")

# Global auth instance
auth_manager = SimpleAuth()