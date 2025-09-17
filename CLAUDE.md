# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the STT Service
- `python main.py websocket` - Start WebSocket server on default port 8000
- `python main.py websocket --port 8080` - Start WebSocket server on custom port
- `python main.py file input.wav` - Process single audio file
- `python main.py microphone` - Record from microphone and transcribe
- `python main.py config --show` - Show current configuration

### Setup and Validation
- `python setup_websocket_server.py` - Check dependencies and system readiness
- `python setup_websocket_server.py --generate-files` - Generate Docker deployment files

### Testing
- `pytest` - Run unit tests
- `python app/client_examples/websocket_client_example.py` - Test WebSocket client
- `python scripts/test_endpoints.bat` - Test HTTP endpoints (Windows)

### Docker Operations
- `docker build -t stt-service .` - Build CPU container
- `docker build -f Dockerfile.gpu -t stt-service-gpu .` - Build GPU container
- `docker-compose -f docker-compose.gpu.yml up` - Run with GPU support

## Architecture Overview

### Core Components

**Main Entry Point (`main.py`)**
- Command-line interface with multiple modes: websocket, file, microphone, config
- Integrates WebSocket server, file processing, and microphone capture

**Core Modules (`app/core/`)**
- `whisper_handler.py` - Whisper model adapter for transcription
- `websocket_server.py` - WebSocket server implementation
- `realtime_transcription.py` - Real-time transcription processing
- `audio_processor.py` - Audio preprocessing and format handling
- `microphone_capture.py` - Microphone input capture utilities

**Utilities (`app/utils/`)**
- `config.py` - Configuration management with environment variable support
- `logger.py` - Centralized logging setup
- `error_handler.py` - Error handling and recovery utilities

**Monitoring (`app/monitoring/`)**
- `monitoring_service.py` - Health checks and service status endpoints
- `service_monitor.py` - Background monitoring and metrics collection

### WebSocket Architecture
- FastAPI-based WebSocket server on port 8000
- Real-time audio streaming with chunked processing
- Monitoring service on port 9091 for health checks
- Supports concurrent connections with configurable limits

### Model Integration
- Uses faster-whisper for efficient GPU/CPU transcription
- Configurable model sizes (base, small, medium, large, large-v3)
- Supports both batch and streaming transcription modes
- GPU acceleration with CUDA support when available

### Configuration System
- Environment variable override support
- Default configurations in `app/utils/config.py`
- Model, server, and processing parameters are configurable
- Validation utilities to check setup completeness

### Dependencies
- Primary: FastAPI, uvicorn, websockets, faster-whisper, torch
- Audio: pyaudio, soundfile, numpy, scipy
- Monitoring: flask, psutil
- GPU: CUDA toolkit, nvidia-cudnn-cu12

### Development Notes
- This is a Python-based microservice (not Node.js)
- GPU support requires CUDA 12.1 and compatible drivers
- WebSocket endpoints: `/ws/transcribe`, `/health`, `/stats`
- Uses requirements.txt for dependency management
- Supports both real-time and batch transcription workflows