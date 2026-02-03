import io
import base64
import librosa
import numpy as np
import shutil
import os

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None


def _find_winget_ffmpeg_exe(exe_name: str) -> str | None:
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if not base or not os.path.isdir(base):
        return None

    # Common winget package layout (as observed on this machine).
    candidates = [
        os.path.join(base, "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"),
    ]

    for root in candidates:
        if not os.path.isdir(root):
            continue

        for dirpath, _dirnames, filenames in os.walk(root):
            if exe_name in filenames and os.path.basename(dirpath).lower() == "bin":
                return os.path.join(dirpath, exe_name)

    # Fallback: full walk (bounded by the base folder)
    for dirpath, _dirnames, filenames in os.walk(base):
        if exe_name in filenames and os.path.basename(dirpath).lower() == "bin":
            return os.path.join(dirpath, exe_name)

    return None

class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def decode_base64_audio(self, audio_base64: str) -> bytes:
        """Decode base64 string to audio bytes"""
        try:
            if not audio_base64 or not isinstance(audio_base64, str):
                raise ValueError("audioBase64 must be a non-empty base64 string")

            # Support data URL prefix: data:audio/mp3;base64,...
            if "," in audio_base64 and audio_base64.strip().lower().startswith("data:"):
                audio_base64 = audio_base64.split(",", 1)[1]

            audio_bytes = base64.b64decode(audio_base64)
            return audio_bytes
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {str(e)}")
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> tuple:
        """Load audio from MP3 bytes.
        """
        if AudioSegment is None:
            raise ValueError(
                "MP3 decoding is unavailable because pydub is not installed correctly. "
                "Reinstall dependencies from requirement.txt."
            )

        ffmpeg_path = shutil.which("ffmpeg") or _find_winget_ffmpeg_exe("ffmpeg.exe")
        ffprobe_path = shutil.which("ffprobe") or _find_winget_ffmpeg_exe("ffprobe.exe")

        if ffmpeg_path and not os.path.exists(ffmpeg_path):
            ffmpeg_path = None
        if ffprobe_path and not os.path.exists(ffprobe_path):
            ffprobe_path = None

        if ffmpeg_path or ffprobe_path:
            bin_dir = os.path.dirname(ffmpeg_path or ffprobe_path)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path
        if ffprobe_path:
            AudioSegment.ffprobe = ffprobe_path

        if not ffmpeg_path or not ffprobe_path:
            raise ValueError(
                "MP3 decoding requires ffmpeg and ffprobe on PATH, but they were not found. "
                "Install FFmpeg and/or ensure it is available. If installed via winget, restart the server."
            )

        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except FileNotFoundError as e:
            raise ValueError(
                "Failed to load audio: ffmpeg/ffprobe executable could not be invoked. "
                f"ffmpeg_path={ffmpeg_path}, ffprobe_path={ffprobe_path}"
            ) from e

        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        audio_segment = audio_segment.set_frame_rate(self.target_sr)
        samples = np.array(audio_segment.get_array_of_samples())
        audio_data = samples.astype(np.float32) / (1 << 15)

        return audio_data, self.target_sr
    
    def validate_audio_duration(self, audio_data: np.ndarray, sr: int, 
                               min_duration: float = 1.0, 
                               max_duration: float = 60.0) -> bool:
        """Validate audio duration is within acceptable range"""
        duration = len(audio_data) / sr
        
        if duration < min_duration:
            raise ValueError(f"Audio too short: {duration:.2f}s (minimum {min_duration}s)")
        
        if duration > max_duration:
            raise ValueError(f"Audio too long: {duration:.2f}s (maximum {max_duration}s)")
        
        return True
    
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Apply preprocessing to audio"""
        # Audio must not be modified per evaluation rules.
        return audio_data
