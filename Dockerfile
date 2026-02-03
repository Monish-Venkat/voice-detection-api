FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# System deps for MP3 decoding via pydub/ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

COPY . /app

EXPOSE 5000

CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 180"]
