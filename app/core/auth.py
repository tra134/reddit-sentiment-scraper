# app/core/auth.py
from datetime import datetime, timedelta
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import Optional
import streamlit as st
import os
import hashlib
import secrets
import hmac

# Password hashing với fallback mạnh hơn
try:
    # Thử sử dụng scheme khác nếu bcrypt lỗi
    pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
    pwd_context.hash("test")  # Test ngay lập tức
    BC_AVAILABLE = True
    print("✅ Password hashing backend available")
except Exception as e:
    print(f"⚠️ Advanced hashing not available: {e}, using secure fallback")
    BC_AVAILABLE = False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    if BC_AVAILABLE:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # Fallback nếu có lỗi
            return fallback_verify_password(plain_password, hashed_password)
    else:
        return fallback_verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    if BC_AVAILABLE:
        try:
            # Giới hạn password length cho bcrypt
            if len(password) > 72:
                password = password[:72]
            return pwd_context.hash(password)
        except Exception:
            # Fallback nếu có lỗi
            return fallback_get_password_hash(password)
    else:
        return fallback_get_password_hash(password)

def fallback_verify_password(plain_password: str, hashed_password: str) -> bool:
    """Secure fallback password verification"""
    try:
        # Format: algorithm$iterations$salt$hash
        parts = hashed_password.split('$')
        if len(parts) != 4:
            return False
            
        algorithm, iterations, salt, stored_hash = parts
        iterations = int(iterations)
        
        # Hash password với salt và iterations
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            plain_password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations
        ).hex()
        
        return hmac.compare_digest(password_hash, stored_hash)
    except:
        return False

def fallback_get_password_hash(password: str) -> str:
    """Secure fallback password hashing"""
    # Tạo salt ngẫu nhiên
    salt = secrets.token_hex(16)
    iterations = 100000  # Số lần hash
    
    # Hash password với PBKDF2
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations
    ).hex()
    
    # Format: algorithm$iterations$salt$hash
    return f"pbkdf2_sha256${iterations}${salt}${password_hash}"

def authenticate_user(db: Session, username: str, password: str):
    """Authenticate user credentials"""
    # Import inside function to avoid circular import
    from app.core.user_database import User
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user():
    """Get current user from Streamlit session state"""
    return st.session_state.get("user")

def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    if 'user_groups' in st.session_state:
        del st.session_state.user_groups
    st.rerun()