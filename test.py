import random
import requests

def test_pollinations():
    """Test rapide de Pollinations."""
    
    import urllib.parse
    import random
    
    prompt = "beautiful princess"
    encoded_prompt = urllib.parse.quote(prompt)
    seed = random.randint(1, 1000000)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&seed={seed}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
    print(f"Content-Length: {len(response.content)} bytes")
    
    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
        print("✅ Pollinations fonctionne !")
        return True
    else:
        print("❌ Pollinations ne fonctionne pas")
        return False

# Copiez cette fonction et exécutez-la
test_pollinations()