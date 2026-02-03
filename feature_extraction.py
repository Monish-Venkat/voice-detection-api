import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel  # Changed import

class FeatureExtractor:
    def __init__(self, model_name="microsoft/wavlm-base-plus", device="cpu", enable_wavlm: bool = True):
        """Initialize WavLM feature extractor"""
        self.device = device
        self.enable_wavlm = enable_wavlm

        self.processor = None
        self.model = None

        if self.enable_wavlm:
            # Use FeatureExtractor instead of Processor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = WavLMModel.from_pretrained(model_name).to(device)
            self.model.eval()
    
    def extract_wavlm_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extract WavLM embeddings from audio"""
        if not self.enable_wavlm or self.processor is None or self.model is None:
            raise RuntimeError("WavLM feature extraction is disabled")

        # Resample if necessary
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        # Process audio - corrected for feature extractor
        inputs = self.processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(input_values)
            # Use last hidden state
            hidden_states = outputs.last_hidden_state
            
            # Average pool across time dimension
            features = torch.mean(hidden_states, dim=1)
        
        return features.cpu().numpy()
    
    def extract_handcrafted_features(self, audio_data: np.ndarray, sr: int) -> dict:
        """Extract traditional audio features for explanation generation"""
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Pitch/F0
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        
        return features
    
    def extract_all_features(self, audio_data: np.ndarray, sr: int) -> tuple:
        """Extract both WavLM and handcrafted features"""
        wavlm_features = self.extract_wavlm_features(audio_data, sr)
        handcrafted_features = self.extract_handcrafted_features(audio_data, sr)
        
        return wavlm_features, handcrafted_features
