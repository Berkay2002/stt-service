@echo off
REM STT Service High-End RTX Deployment Script for Windows
REM Run this script to build and deploy the GPU-optimized container

echo 🚀 STT Service High-End RTX Deployment
echo ==================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if nvidia-docker is available
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ NVIDIA Docker runtime not available. Please install nvidia-docker2.
    pause
    exit /b 1
)

REM Create required directories
echo 📁 Creating required directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Build the container
echo 🔨 Building high-end RTX optimized container...
docker build -f Dockerfile.gpu -t stt-service:gpu .

REM Run the container
echo 🚀 Starting STT Service container...
docker-compose -f docker-compose.gpu.yml up -d

REM Wait for service to be ready
echo ⏳ Waiting for service to be ready...
timeout /t 10 /nobreak >nul

REM Check health
echo 🔍 Checking service health...
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Service health check failed. Check logs:
    docker-compose -f docker-compose.gpu.yml logs
    pause
    exit /b 1
)

echo ✅ WebSocket STT Service is running successfully!
echo.
echo 📊 Service URLs:
echo    WebSocket STT: ws://localhost:8000/ws/transcribe
echo    Health Check: http://localhost:8000/health
echo    Service Stats: http://localhost:8000/stats
echo    Monitoring: http://localhost:9091/health
echo    Metrics: http://localhost:9091/metrics
echo.
echo 📋 Useful Commands:
echo    View logs: docker-compose -f docker-compose.gpu.yml logs -f
echo    Stop service: docker-compose -f docker-compose.gpu.yml down
echo    Restart: docker-compose -f docker-compose.gpu.yml restart
echo.
echo 🖥️  GPU Status:
docker exec stt-gpu-service nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

pause