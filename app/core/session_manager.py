import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manage application sessions and state"""
    
    def __init__(self):
        self.sessions = {}
        self.session_timeout = timedelta(hours=1)
    
    def create_session(self, initial_data: Dict[str, Any] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'data': initial_data or {}
        }
        logger.info(f"🆕 Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check if session is expired
            if datetime.now() - session['last_accessed'] > self.session_timeout:
                del self.sessions[session_id]
                return None
            
            # Update last accessed time
            session['last_accessed'] = datetime.now()
            return session['data']
        
        return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id]['data'].update(data)
            self.sessions[session_id]['last_accessed'] = datetime.now()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️ Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_keys = []
        for session_id, session in self.sessions.items():
            if datetime.now() - session['last_accessed'] > self.session_timeout:
                expired_keys.append(session_id)
        
        for key in expired_keys:
            del self.sessions[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired sessions")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_sessions = len(self.sessions)
        active_sessions = 0
        
        for session in self.sessions.values():
            if datetime.now() - session['last_accessed'] < timedelta(minutes=5):
                active_sessions += 1
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'session_timeout_minutes': self.session_timeout.total_seconds() / 60
        }

# Global session manager
session_manager = SessionManager()