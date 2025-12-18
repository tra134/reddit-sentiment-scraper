# test_reddit_urls.py
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print("Testing RedditLoader with correct URL handling...")
print("=" * 60)

from app.main import RedditLoader

loader = RedditLoader()

# Test với các URL đúng
test_urls = [
    # Post URLs (sẽ được tự động thêm .json)
    "https://www.reddit.com/r/funny/comments/3g1jfi/buttons/",
    "https://www.reddit.com/r/AskReddit/comments/1g76q6q/what_is_a_fun_fact_that_you_just_made_up/",
    
    # URLs đã có .json
    "https://www.reddit.com/r/funny/comments/3g1jfi/buttons.json",
    "https://www.reddit.com/r/programming/comments/1g8w3v5/why_is_c_still_the_king_of_programming_languages.json",
    
    # Subreddit URLs
    "https://www.reddit.com/r/python/hot/",
    "https://www.reddit.com/r/technology/new.json"
]

for i, url in enumerate(test_urls, 1):
    print(f"\nTest {i}: {url}")
    print("-" * 40)
    
    data, error = loader.fetch_data(url)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"✅ Success!")
        
        if 'meta' in data and data['meta']:
            print(f"   Title: {data['meta'].get('title', 'Unknown')}")
            print(f"   Subreddit: r/{data['meta'].get('subreddit', 'unknown')}")
        
        items_count = len(data.get('comments', []))
        print(f"   Items found: {items_count}")
        
        if items_count > 0 and 'comments' in data:
            first_item = data['comments'][0]
            if 'body' in first_item:
                print(f"   First comment: {first_item['body'][:60]}...")
            elif 'title' in first_item:
                print(f"   First post: {first_item['title'][:60]}...")

print("\n" + "=" * 60)
print("Testing URL variants for 404 handling...")

# Test URL không tồn tại
non_existent_url = "https://www.reddit.com/r/test/comments/3g1jfi/this_is_a_test_post/"
print(f"\nTesting non-existent URL: {non_existent_url}")
data, error = loader.fetch_data(non_existent_url)
print(f"Result: {error}")