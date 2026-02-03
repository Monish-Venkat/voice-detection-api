import base64

with open("tests/test_audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

print(audio_base64)