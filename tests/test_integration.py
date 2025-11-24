# tests/debug_paths.py
import sys
import os

print("🔍 Debugging Paths")
print("=" * 40)

# In các đường dẫn hiện tại
print("Current directory:", os.getcwd())
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("Python path:")
for path in sys.path:
    print("  ", path)

# Thử import trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(f"\nParent directory: {parent_dir}")

# Thêm parent directory
sys.path.insert(0, parent_dir)

print(f"\nUpdated Python path:")
for path in sys.path:
    print("  ", path)

# Kiểm tra xem app có tồn tại không
app_path = os.path.join(parent_dir, 'app')
print(f"\nApp directory exists: {os.path.exists(app_path)}")
if os.path.exists(app_path):
    print("App directory contents:", os.listdir(app_path))