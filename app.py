from flask import Flask, request, jsonify
from functools import wraps
import traceback

from config import Config
from utils.audio_processor import AudioProcessor
from feature_extraction import FeatureExtractor
from utils.explanation_generator import ExplanationGenerator
from model import VoiceDetectionModel

app = Flask(__name__)
config = Config()

# Initialize components
audio_processor = AudioProcessor(target_sr=config.SAMPLE_RATE)
feature_extractor = FeatureExtractor(
    model_name=config.WAVLM_MODEL, 
    device=config.DEVICE,
    enable_wavlm=False
)
explanation_generator = ExplanationGenerator()
try:
    detection_model = VoiceDetectionModel(
        model_path=config.MODEL_PATH,
        device=config.DEVICE
    )
except Exception as e:
    detection_model = None
    print(f"Failed to initialize detection model: {str(e)}")


# API Key Authentication Decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        
        if not api_key or api_key != config.API_SECRET_KEY:
            return jsonify({
                "status": "error",
                "message": "Invalid API key or malformed request"
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


@app.route('/api/voice-detection', methods=['POST'])
@require_api_key
def detect_voice():
    """Main endpoint for voice detection"""
    try:
        # Parse request
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({
                "status": "error",
                "message": "Invalid API key or malformed request"
            }), 400

        if detection_model is None:
            return jsonify({
                "status": "error",
                "message": "Server misconfigured: model weights not loaded"
            }), 503
        
        # Validate required fields
        required_fields = ['language', 'audioFormat', 'audioBase64']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        language = data['language']
        audio_format = data['audioFormat']
        audio_base64 = data['audioBase64']
        
        # Validate language
        if language not in config.SUPPORTED_LANGUAGES:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(config.SUPPORTED_LANGUAGES)}"
            }), 400
        
        # Validate audio format
        if audio_format.lower() != 'mp3':
            return jsonify({
                "status": "error",
                "message": "Only MP3 format is supported"
            }), 400
        
        # Decode base64 audio
        try:
            audio_bytes = audio_processor.decode_base64_audio(audio_base64)
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
        
        # Load audio from bytes
        try:
            audio_data, sr = audio_processor.load_audio_from_bytes(audio_bytes)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to load audio: {str(e)}"
            }), 400

        if language not in config.SUPPORTED_LANGUAGES:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(config.SUPPORTED_LANGUAGES)}"
            }), 400
        
        # Validate audio duration
        try:
            audio_processor.validate_audio_duration(
                audio_data, sr, 
                min_duration=config.MIN_AUDIO_LENGTH,
                max_duration=config.MAX_AUDIO_LENGTH
            )
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
        
        # Preprocess audio
        audio_data = audio_processor.preprocess_audio(audio_data, sr)

        # Extract handcrafted features for explanations (no audio modification)
        handcrafted_features = feature_extractor.extract_handcrafted_features(audio_data, sr)

        # Predict
        classification, confidence = detection_model.predict(audio_data, sr, language)
        
        # Generate explanation
        explanation = explanation_generator.generate_explanation(
            classification, confidence, handcrafted_features
        )

        # Defensive cleanup for duplicated tokens
        while "detected detected" in explanation:
            explanation = explanation.replace("detected detected", "detected")
        while "confirmed confirmed" in explanation:
            explanation = explanation.replace("confirmed confirmed", "confirmed")
        
        # Return response
        response = {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(float(confidence), 2),
            "explanation": explanation
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Log error for debugging
        print(f"Error in voice detection: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "message": "Internal server error occurred during processing"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Voice Detection API",
        "supported_languages": config.SUPPORTED_LANGUAGES
    }), 200


if __name__ == '__main__':
    print(f"Starting AI Voice Detection API on {config.DEVICE}...")
    print(f"Supported languages: {', '.join(config.SUPPORTED_LANGUAGES)}")
    app.run(host='0.0.0.0', port=5000, debug=False)
