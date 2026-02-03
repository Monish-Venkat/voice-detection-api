import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # API Configuration
    API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'your-secret-key-here')
    
    # Supported Languages
    SUPPORTED_LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
    
    # Audio Processing
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 60  # seconds
    MIN_AUDIO_LENGTH = 1   # seconds
    
    # Model Configuration
    WAVLM_MODEL = "microsoft/wavlm-base-plus"
    MODEL_PATH = os.getenv('MODEL_PATH', "models/wav2vec_aasist.pth")
    DEVICE = "cuda" if os.getenv('USE_GPU', 'false').lower() == 'true' else "cpu"
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.65
    
    # Classification Labels
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"
