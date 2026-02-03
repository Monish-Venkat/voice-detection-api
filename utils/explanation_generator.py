import numpy as np

class ExplanationGenerator:
    def __init__(self):
        # Define thresholds for various features
        self.thresholds = {
            'pitch_std_low': 15.0,      # Low pitch variation indicates AI
            'pitch_std_high': 50.0,     # High pitch variation indicates human
            'zcr_std_low': 0.05,        # Low ZCR variation indicates AI
            'spectral_consistency': 500.0,  # High spectral consistency indicates AI
        }
    
    def generate_explanation(self, 
                           classification: str, 
                           confidence: float, 
                           handcrafted_features: dict) -> str:
        """Generate human-readable explanation for the classification"""
        
        explanations = []
        
        if classification == "AI_GENERATED":
            # Check pitch consistency
            if handcrafted_features['pitch_std'] < self.thresholds['pitch_std_low']:
                explanations.append("unnatural pitch consistency")
            
            # Check spectral characteristics
            if handcrafted_features['spectral_centroid_std'] < self.thresholds['spectral_consistency']:
                explanations.append("robotic spectral patterns")
            
            # Check zero crossing rate
            if handcrafted_features['zcr_std'] < self.thresholds['zcr_std_low']:
                explanations.append("mechanical speech rhythm")
            
            # Check energy patterns
            if handcrafted_features['energy_std'] < 0.02:
                explanations.append("unnatural energy distribution")
            
            # Default explanation if none found
            if not explanations:
                explanations.append("synthetic voice artifacts")
            
            reason = " and ".join(explanations)
            return f"{reason.capitalize()} detected"
        
        else:  # HUMAN
            # Check natural variations
            if handcrafted_features['pitch_std'] > self.thresholds['pitch_std_high']:
                explanations.append("natural pitch variations")
            
            # Check breathing and pauses
            if handcrafted_features['energy_std'] > 0.03:
                explanations.append("organic energy fluctuations")
            
            # Check spectral diversity
            if handcrafted_features['spectral_centroid_std'] > self.thresholds['spectral_consistency']:
                explanations.append("human-like spectral dynamics")
            
            # Default explanation
            if not explanations:
                explanations.append("natural human speech characteristics")
            
            reason = " and ".join(explanations)
            return f"{reason.capitalize()} confirmed"
