# setup_websocket_server.py

"""
Setup and validation script for the WebSocket STT Server
Checks dependencies, validates configuration, and provides deployment guidance
"""

import sys
import subprocess
import importlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_required_packages() -> List[Tuple[str, bool, str]]:
    """Check if required packages are installed"""
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("websockets", "WebSocket support"),
        ("numpy", "Numerical computing"),
        ("torch", "PyTorch for GPU support"),
        ("torchaudio", "Audio processing"),
        ("faster_whisper", "Whisper model"),
        ("pyaudio", "Audio I/O (optional)"),
        ("psutil", "System monitoring"),
        ("flask", "Monitoring endpoints"),
    ]

    results = []
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            results.append((package, True, description))
        except ImportError:
            results.append((package, False, description))

    return results

def check_gpu_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """Check GPU availability and configuration"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available", {}

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        # Get memory info
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        total_memory_gb = total_memory / (1024**3)

        allocated = torch.cuda.memory_allocated(current_device) / (1024**3)

        gpu_info = {
            "device_count": device_count,
            "current_device": current_device,
            "device_name": device_name,
            "total_memory_gb": round(total_memory_gb, 1),
            "allocated_memory_gb": round(allocated, 2),
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        }

        return True, f"GPU available: {device_name}", gpu_info

    except ImportError:
        return False, "PyTorch not installed", {}
    except Exception as e:
        return False, f"GPU check failed: {e}", {}

def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate the current configuration"""
    try:
        from app.utils.config import load_config, validate_config

        config = load_config()
        is_valid, warnings = validate_config(config)

        return is_valid, warnings

    except ImportError:
        return False, ["Configuration module not found"]
    except Exception as e:
        return False, [f"Configuration validation failed: {e}"]

def check_port_availability(port: int = 8000) -> bool:
    """Check if the specified port is available"""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def generate_docker_compose() -> str:
    """Generate a Docker Compose configuration"""
    return """version: '3.8'

services:
  stt-websocket:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8000:8000"
      - "9091:9091"
    environment:
      - WEBSOCKET_HOST=0.0.0.0
      - WEBSOCKET_PORT=8000
      - WEBSOCKET_MAX_CONNECTIONS=50
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add a reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - stt-websocket
    restart: unless-stopped"""

def generate_nginx_config() -> str:
    """Generate NGINX configuration for WebSocket proxy"""
    return """events {
    worker_connections 1024;
}

http {
    upstream stt_backend {
        server stt-websocket:8000;
    }

    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    server {
        listen 80;
        server_name localhost;

        # WebSocket proxy
        location /ws/ {
            proxy_pass http://stt_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket specific settings
            proxy_read_timeout 86400;
            proxy_send_timeout 86400;
            proxy_connect_timeout 86400;
        }

        # HTTP API endpoints
        location / {
            proxy_pass http://stt_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}"""

def print_setup_report():
    """Print comprehensive setup report"""
    print("\\n" + "="*70)
    print("WEBSOCKET STT SERVER SETUP REPORT")
    print("="*70)

    # Python version check
    python_ok, python_info = check_python_version()
    print(f"\\n📋 Python Version: {'✅' if python_ok else '❌'} {python_info}")

    # Package checks
    print("\\n📦 Required Packages:")
    package_results = check_required_packages()
    all_packages_ok = True

    for package, installed, description in package_results:
        status = "✅" if installed else "❌"
        print(f"   {status} {package:<20} - {description}")
        if not installed:
            all_packages_ok = False

    # GPU check
    gpu_ok, gpu_info, gpu_details = check_gpu_availability()
    print(f"\\n🔥 GPU Support: {'✅' if gpu_ok else '❌'} {gpu_info}")

    if gpu_ok and gpu_details:
        print(f"   Device: {gpu_details.get('device_name', 'Unknown')}")
        print(f"   Memory: {gpu_details.get('total_memory_gb', 0)}GB total")
        print(f"   CUDA: {gpu_details.get('cuda_version', 'Unknown')}")

    # Configuration check
    config_ok, config_warnings = validate_configuration()
    print(f"\\n⚙️  Configuration: {'✅' if config_ok else '❌'}")

    if config_warnings:
        for warning in config_warnings:
            print(f"   ⚠️  {warning}")

    # Port availability
    port_8000_ok = check_port_availability(8000)
    port_9091_ok = check_port_availability(9091)
    print(f"\\n🌐 Port Availability:")
    print(f"   {'✅' if port_8000_ok else '❌'} Port 8000 (WebSocket)")
    print(f"   {'✅' if port_9091_ok else '❌'} Port 9091 (Monitoring)")

    # Overall status
    overall_ok = all([python_ok, all_packages_ok, gpu_ok, config_ok, port_8000_ok])
    print(f"\\n🚀 Overall Status: {'✅ READY' if overall_ok else '❌ NEEDS ATTENTION'}")

    # Installation guidance
    if not all_packages_ok:
        print("\\n📥 Installation Commands:")
        print("   pip install -r requirements.txt")

        missing_packages = [pkg for pkg, installed, _ in package_results if not installed]
        if missing_packages:
            print(f"   # Missing: {', '.join(missing_packages)}")

    # Usage examples
    print("\\n🎯 Usage Examples:")
    print("   python main.py websocket                 # Start WebSocket server")
    print("   python main.py websocket --port 8080    # Custom port")
    print("   python main.py file audio.wav           # Process file")
    print("   python main.py microphone               # Record from mic")

    # Docker deployment
    print("\\n🐳 Docker Deployment:")
    print("   docker-compose up --build               # Start with Docker")
    print("   # Use generated docker-compose.yml and nginx.conf")

    # Testing
    print("\\n🧪 Testing:")
    print("   python app/client_examples/websocket_client_example.py")

    print("\\n" + "="*70)

def setup_deployment_files():
    """Generate deployment configuration files"""
    # Generate Docker Compose
    docker_compose_path = Path("docker-compose.websocket.yml")
    with open(docker_compose_path, "w") as f:
        f.write(generate_docker_compose())
    print(f"Generated: {docker_compose_path}")

    # Generate NGINX config
    nginx_config_path = Path("nginx.conf")
    with open(nginx_config_path, "w") as f:
        f.write(generate_nginx_config())
    print(f"Generated: {nginx_config_path}")

    # Generate environment template
    env_template = """# WebSocket STT Server Configuration

# Server Configuration
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8000
WEBSOCKET_MAX_CONNECTIONS=50

# Error Handling
TRANSCRIPTION_FAILURE_THRESHOLD=5
GPU_FAILURE_THRESHOLD=3
MAX_SESSION_ERRORS=10
MAX_RECENT_ERRORS=50

# Processing Configuration
BUFFER_DURATION=2.0
MIN_AUDIO_LENGTH=0.5
MAX_SILENCE_DURATION=2.0

# Logging
LOG_LEVEL=INFO

# Model Configuration (override config.py)
# MODEL_NAME=large-v3
# BEAM_SIZE=10
# FP16=true
"""

    env_path = Path(".env.template")
    with open(env_path, "w") as f:
        f.write(env_template)
    print(f"Generated: {env_path}")

def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket STT Server Setup")
    parser.add_argument("--generate-files", action="store_true",
                       help="Generate deployment configuration files")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check setup, don't generate files")

    args = parser.parse_args()

    # Always print the setup report
    print_setup_report()

    if args.generate_files:
        print("\\n📁 Generating deployment files...")
        setup_deployment_files()
        print("\\n✅ Deployment files generated successfully!")

    elif not args.check_only:
        # Default: ask user if they want to generate files
        response = input("\\n❓ Generate deployment files? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            print("\\n📁 Generating deployment files...")
            setup_deployment_files()
            print("\\n✅ Deployment files generated successfully!")

if __name__ == "__main__":
    main()