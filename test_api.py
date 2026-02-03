import requests
import base64
import json

def test_voice_detection_api(audio_file_path, language, api_key):
    """Test the voice detection API"""
    
    # Read and encode audio file
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request
    url = "http://localhost:5000/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Send request
    print(f"Testing with audio file: {audio_file_path}")
    print(f"Language: {language}")
    print(f"Sending request to {url}...")
    
    response = requests.post(url, headers=headers, json=payload)
    
    # Print response
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    return response.json()


if __name__ == "__main__":
    # Test configuration
    API_KEY = "sk_test_123456789"
    AUDIO_FILE = "tests/test_audio.mp3"
    LANGUAGE = "English"
    
    # Run test
    result = test_voice_detection_api(AUDIO_FILE, LANGUAGE, API_KEY)
