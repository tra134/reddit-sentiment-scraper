# app/components/auth_components.py
import streamlit as st
import sys
import os
import time

# Thêm path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from services.user_service import UserService
    from core.auth import get_password_hash, verify_password, authenticate_user
except ImportError as e:
    st.error(f"Import error in auth_components: {e}")

def show_login_form():
    """Hiển thị form đăng nhập"""
    st.subheader("🔐 Đăng nhập")
    
    # Tạo unique key cho form
    form_key = f"login_form_{int(time.time())}"
    
    with st.form(form_key):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submit = st.form_submit_button("Đăng nhập")
        
        if submit:
            if not username or not password:
                st.error("Vui lòng nhập đầy đủ thông tin")
                return
                
            try:
                user_service = UserService()
                user = user_service.authenticate_user(username, password)
                
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.success(f"Đăng nhập thành công! Chào mừng {user.username}")
                    st.rerun()
                else:
                    st.error("Tên đăng nhập hoặc mật khẩu không đúng")
                    
            except Exception as e:
                st.error(f"Lỗi đăng nhập: {str(e)}")

def show_register_form():
    """Hiển thị form đăng ký"""
    st.subheader("📝 Đăng ký tài khoản mới")
    
    # Tạo unique key cho form
    form_key = f"register_form_{int(time.time())}"
    
    with st.form(form_key):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Tên đăng nhập *")
            full_name = st.text_input("Họ và tên")
            
        with col2:
            email = st.text_input("Email *")
            password = st.text_input("Mật khẩu *", type="password")
            confirm_password = st.text_input("Xác nhận mật khẩu *", type="password")
        
        st.markdown("**Lưu ý:** Mật khẩu phải có ít nhất 6 ký tự")
        submit = st.form_submit_button("Đăng ký")
        
        if submit:
            # Validation
            if not all([username, email, password, confirm_password]):
                st.error("Vui lòng điền đầy đủ các trường bắt buộc (*)")
                return
                
            if password != confirm_password:
                st.error("Mật khẩu xác nhận không khớp")
                return
                
            if len(password) < 6:
                st.error("Mật khẩu phải có ít nhất 6 ký tự")
                return
                
            try:
                user_service = UserService()
                user = user_service.create_user(
                    username=username,
                    email=email,
                    password=password,
                    full_name=full_name
                )
                
                st.success("🎉 Đăng ký thành công! Bạn có thể đăng nhập ngay bây giờ.")
                st.info("Vui lòng chuyển sang tab Đăng nhập")
                
            except ValueError as e:
                st.error(f"Lỗi đăng ký: {str(e)}")
            except Exception as e:
                st.error(f"Lỗi hệ thống: {str(e)}")

def show_auth_section():
    """Hiển thị section xác thực"""
    if is_authenticated():
        user = get_current_user()
        st.sidebar.success(f"👋 Chào {user.username}!")
        
        if st.sidebar.button("🚪 Đăng xuất"):
            logout()
            
    else:
        # Tab cho đăng nhập/đăng ký
        tab1, tab2 = st.tabs(["🔐 Đăng nhập", "📝 Đăng ký"])
        
        with tab1:
            show_login_form()
            
        with tab2:
            show_register_form()

def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def get_current_user():
    """Get current user from Streamlit session state"""
    return st.session_state.get("user")

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    if 'user_groups' in st.session_state:
        del st.session_state.user_groups
    st.rerun()