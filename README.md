# AI-Generated Voice Detection API (Tamil/English/Hindi/Malayalam/Telugu)

A REST API that accepts one Base64-encoded MP3 audio sample per request and returns whether the voice is **AI-generated** or **human**, with a confidence score and short explanation.

## Supported Languages (Fixed)
- Tamil
- English
- Hindi
- Malayalam
- Telugu

## API Authentication
All requests must include an API key header:

- `x-api-key: YOUR_SECRET_API_KEY`

Set your key in `.env`:

```env
API_SECRET_KEY=sk_test_123456789
USE_GPU=false
```

## Endpoints

### POST `/api/voice-detection`

Request JSON:

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "..."
}
```

Response JSON:

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### GET `/health`

## Local Run

1) Install dependencies:

```bash
python -m pip install -r requirement.txt
```

2) Create `.env` from `.env.example`.

3) Start server:

```bash
python app.py
```

4) Test:

```bash
python test_api.py
```

## Deployment

### Docker (recommended)
This project includes a `Dockerfile` that installs `ffmpeg` (needed for MP3 decoding) and runs the API using `gunicorn`.

Set these environment variables on your deployment platform:
- `API_SECRET_KEY`
- `USE_GPU=false`
- Optional: `HF_TOKEN`

## Notes
- The server uses a pretrained open-source deepfake-audio classifier downloaded at runtime on first start.
- Do not commit `.env`.
