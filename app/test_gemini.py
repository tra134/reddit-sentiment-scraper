import google.generativeai as genai

# --- DÁN KEY CỦA BẠN VÀO ĐÂY ---
MY_API_KEY = "" 
# -------------------------------

print(f"1. Đang kiểm tra Key: {MY_API_KEY[:10]}...")

try:
    genai.configure(api_key=MY_API_KEY)
    
    # Thử liệt kê model để xem Key có hoạt động không
    print("2. Key hợp lệ! Đang lấy danh sách model...")
    found_flash = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - Tìm thấy model: {m.name}")
            if 'gemini-1.5-flash' in m.name:
                found_flash = True

    # Thử chạy thử 1 câu lệnh
    print("\n3. Đang test thử tạo văn bản...")
    model = genai.GenerativeModel('gemini-1.5-flash' if found_flash else 'gemini-pro')
    response = model.generate_content("Chào bạn, bạn có khỏe không?")
    print(f"   >>> AI Trả lời: {response.text}")
    print("\n✅ THÀNH CÔNG! Mọi thứ đều ổn.")

except Exception as e:
    print(f"\n❌ LỖI RỒI: {e}")