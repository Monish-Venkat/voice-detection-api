import numpy as np


class LanguageIdentifier:
    def __init__(self, device: str = "cpu"):
        # Lazy import so the rest of the service can run even if speechbrain isn't installed.
        from speechbrain.inference.classifiers import EncoderClassifier

        run_opts = {"device": device}
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id-voxlingua107-ecapa",
            run_opts=run_opts,
        )

        # Map VoxLingua language codes to the 5 allowed languages.
        self.allowed_code_to_name = {
            "ta": "Tamil",
            "en": "English",
            "hi": "Hindi",
            "ml": "Malayalam",
            "te": "Telugu",
        }

    def detect_language(self, audio_data: np.ndarray, sr: int) -> tuple[str, float]:
        """Detect language among the supported 5.

        Returns:
            (language_name, confidence)
        """
        # SpeechBrain expects torch tensor (batch, time) and sample rate 16000 is recommended.
        import torch
        import torchaudio

        wav = torch.from_numpy(audio_data).float().unsqueeze(0)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)

        # classify_batch returns (probabilities, score, predicted_index, predicted_label)
        probs, _, _, _labels = self.classifier.classify_batch(wav)

        probs_np = probs[0].detach().cpu().numpy()
        ind2lab = getattr(self.classifier.hparams.label_encoder, "ind2lab", {})

        code_to_prob: dict[str, float] = {}
        for i in range(len(probs_np)):
            code = str(ind2lab.get(i, "")).strip().lower()
            if code:
                code_to_prob[code] = float(probs_np[i])

        best_code = None
        best_prob = -1.0
        for code in self.allowed_code_to_name.keys():
            p = code_to_prob.get(code, 0.0)
            if p > best_prob:
                best_prob = p
                best_code = code

        if best_code is None:
            return "English", 0.0

        return self.allowed_code_to_name[best_code], float(best_prob)
