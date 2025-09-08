# Use a base Python image
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg build-essential

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code
COPY . .

# Set environment variables for model/backend if needed
ENV MODEL_PATH=/app/models
ENV WHISPER_BACKEND=faster-whisper

# Expose the service port (match what whisper_live uses)
EXPOSE 5000

# Command to start Whisper Live with faster-whisper backend. Use base, small or medium...
CMD ["whisper_live", "--model", "small", "--backend", "faster-whisper"]