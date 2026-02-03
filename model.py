class VoiceDetectionModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize a pretrained deepfake detector.

        Note: model_path is kept for backward compatibility with existing config,
        but weights are loaded from Hugging Face at runtime.
        """
        self.device = device

        # Lazy import to keep module import lightweight.
        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        self.classifier = pipeline(
            task="audio-classification",
            model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
            device=device_arg,
        )

    def _map_label_to_class(self, label: str) -> str:
        normalized = (label or "").strip().lower()
        if any(token in normalized for token in ["fake", "spoof", "ai", "synth", "generated"]):
            return "AI_GENERATED"
        if any(token in normalized for token in ["real", "bonafide", "human"]):
            return "HUMAN"
        # Conservative fallback: if the label is unknown, treat as AI to reduce false accepts.
        return "AI_GENERATED"

    def predict(self, audio_data, sr: int, language: str):
        """Predict whether voice is AI-generated or human.

        Args:
            audio_data: 1D float32 numpy array in range [-1, 1]
            sr: sampling rate
            language: one of the supported languages (currently not used by the model)

        Returns:
            Tuple of (classification, confidence_score)
        """
        # transformers pipeline accepts {"array": np.ndarray, "sampling_rate": int}
        results = self.classifier({"array": audio_data, "sampling_rate": sr})

        if not results:
            return "AI_GENERATED", 0.5

        top = results[0]
        classification = self._map_label_to_class(top.get("label"))
        confidence = float(top.get("score", 0.0))

        return classification, confidence
