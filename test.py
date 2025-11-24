# test_registration_debug.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_registration_debug():
    print("🧪 Testing Registration Debug")
    print("=" * 40)
    
    try:
        from app.core.user_database import user_db_manager
        from app.services.user_service import UserService
        
        print("✅ Modules imported successfully")
        
        # Test database connection
        db = user_db_manager.get_session()
        print("✅ Database connection established")
        
        # Test UserService
        user_service = UserService(db)
        print("✅ UserService created")
        
        # Test registration
        test_username = f"testuser_{int(time.time())}"
        test_email = f"test{int(time.time())}@test.com"
        
        print(f"🔧 Testing with: {test_username}, {test_email}")
        
        user = user_service.create_user(
            username=test_username,
            email=test_email,
            password="testpassword123"
        )
        
        print(f"✅ User created: {user.username} (ID: {user.id})")
        print("🎉 Registration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Registration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import time
    test_registration_debug()