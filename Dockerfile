# CPU Dockerfile
FROM python:3.10

# System dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    build-essential \
    ffmpeg \
    alsa-utils \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Runtime environment variables
ENV MODEL_PATH=/app/models
ENV WHISPER_BACKEND=faster-whisper

# Port and entrypoint
EXPOSE 9090 9091
CMD ["python3", "-m", "app.main"]
