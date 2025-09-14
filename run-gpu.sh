#!/bin/bash

# STT Service High-End RTX Deployment Script
# Run this script to build and deploy the GPU-optimized container

set -e

echo "🚀 STT Service High-End RTX Deployment"
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA Docker runtime not available. Please install nvidia-docker2."
    exit 1
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p models logs

# Build the container
echo "🔨 Building high-end RTX optimized container..."
docker build -f Dockerfile.gpu -t stt-service:gpu .

# Run the container
echo "🚀 Starting STT Service container..."
docker-compose -f docker-compose.gpu.yml up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ WebSocket STT Service is running successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "   WebSocket STT: ws://localhost:8000/ws/transcribe"
    echo "   Health Check: http://localhost:8000/health"
    echo "   Service Stats: http://localhost:8000/stats"
    echo "   Monitoring: http://localhost:9091/health"
    echo "   Metrics: http://localhost:9091/metrics"
    echo ""
    echo "📋 Useful Commands:"
    echo "   View logs: docker-compose -f docker-compose.gpu.yml logs -f"
    echo "   Stop service: docker-compose -f docker-compose.gpu.yml down"
    echo "   Restart: docker-compose -f docker-compose.gpu.yml restart"
    echo ""
    echo "🖥️  GPU Status:"
    docker exec stt-gpu-service nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "❌ Service health check failed. Check logs:"
    docker-compose -f docker-compose.gpu.yml logs
    exit 1
fi