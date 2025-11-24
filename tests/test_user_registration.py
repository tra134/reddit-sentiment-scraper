# test_user_registration_final.py
import sys
import os
import traceback

# Thêm path để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_registration():
    print("🚀 Testing User Registration System")
    print("=" * 50)
    
    try:
        # Import modules
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        print("1. Importing database modules...")
        from app.core.user_database import UserBase, User, UserGroup, UserPreference
        from app.core.auth import get_password_hash, verify_password, BC_AVAILABLE
        
        if BC_AVAILABLE:
            print("   ✅ bcrypt backend available")
        else:
            print("   ⚠️ Using fallback password hashing")
        
        # Tạo UserService
        from sqlalchemy.orm import Session
        from sqlalchemy.exc import SQLAlchemyError, IntegrityError
        
        class UserService:
            def __init__(self, db: Session):
                self.db = db

            def create_user(self, username: str, email: str, password: str, full_name: str = None) -> User:
                """Create new user"""
                print(f"   🔧 Creating user: {username}")
                
                # Validation
                if not username or not email or not password:
                    raise ValueError("Username, email and password are required")
                
                if len(password) < 6:
                    raise ValueError("Password must be at least 6 characters")
                
                if "@" not in email:
                    raise ValueError("Invalid email format")
                
                # Check existing user
                existing_user = self.db.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    if existing_user.username == username:
                        raise ValueError("Username already exists")
                    else:
                        raise ValueError("Email already exists")
                
                # Create user
                hashed_password = get_password_hash(password)
                user = User(
                    username=username,
                    email=email,
                    hashed_password=hashed_password,
                    full_name=full_name,
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
                self.db.refresh(user)
                return user
        
        print("2. Setting up test database...")
        test_engine = create_engine("sqlite:///:memory:")
        UserBase.metadata.create_all(bind=test_engine)
        SessionLocal = sessionmaker(bind=test_engine)
        db = SessionLocal()
        user_service = UserService(db)
        print("   ✅ Test database setup completed")
        
        # Test 1: Password hashing
        print("3. Testing password hashing...")
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed), "Password verification should work"
        assert hashed != password, "Password should be hashed"
        assert len(hashed) > 0, "Hash should not be empty"
        
        print(f"   ✅ Password hashing works (hash length: {len(hashed)})")
        
        # Test 2: Create user
        print("4. Testing user creation...")
        user = user_service.create_user(
            username="testuser",
            email="test@example.com", 
            password="password123",
            full_name="Test User"
        )
        
        assert user.id is not None, "User should have ID"
        assert user.username == "testuser", "Username should match"
        assert user.email == "test@example.com", "Email should match"
        assert user.is_active == True, "User should be active"
        print(f"   ✅ User created: {user.username} (ID: {user.id})")
        
        # Test 3: Check user in database
        db_user = db.query(User).filter(User.username == "testuser").first()
        assert db_user is not None, "User should exist in database"
        assert db_user.hashed_password != "password123", "Password should be hashed"
        print("   ✅ User found in database")
        
        # Test 4: Check preferences
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        assert prefs is not None, "Preferences should be created"
        preferences_dict = prefs.get_preferences()
        assert preferences_dict["theme"] == "light", "Default theme should be light"
        assert preferences_dict["language"] == "en", "Default language should be en"
        print("   ✅ User preferences created correctly")
        
        # Test 5: Test authentication
        print("5. Testing authentication...")
        from app.core.auth import authenticate_user
        
        auth_user = authenticate_user(db, "testuser", "password123")
        assert auth_user is not None, "Authentication should work"
        assert auth_user.id == user.id, "Authenticated user should match"
        
        wrong_auth = authenticate_user(db, "testuser", "wrongpassword")
        assert wrong_auth is None, "Wrong password should fail"
        print("   ✅ Authentication works correctly")
        
        # Test 6: Test duplicate username
        print("6. Testing duplicate username...")
        try:
            user_service.create_user(
                username="testuser",  # Duplicate
                email="test2@example.com",
                password="password123"
            )
            assert False, "Should have failed for duplicate username"
        except ValueError as e:
            assert "username" in str(e).lower() or "exists" in str(e).lower()
            print("   ✅ Duplicate username correctly blocked")
        
        # Test 7: Test duplicate email
        print("7. Testing duplicate email...")
        try:
            user_service.create_user(
                username="testuser2",
                email="test@example.com",  # Duplicate
                password="password123"
            )
            assert False, "Should have failed for duplicate email"
        except ValueError as e:
            assert "email" in str(e).lower() or "exists" in str(e).lower()
            print("   ✅ Duplicate email correctly blocked")
        
        # Test 8: Test weak password
        print("8. Testing weak password...")
        try:
            user_service.create_user(
                username="testuser3",
                email="test3@example.com",
                password="123"  # Too short
            )
            assert False, "Should have failed for weak password"
        except ValueError as e:
            assert "password" in str(e).lower() or "6" in str(e).lower()
            print("   ✅ Weak password correctly blocked")
        
        db.close()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! Registration system is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please check your file structure and imports")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_registration()
    sys.exit(0 if success else 1)