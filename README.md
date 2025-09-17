<div>
<h1>STT-SERVICE</h1>
<p>
	<img src="https://img.shields.io/github/license/Berkay2002/stt-service?style=flat-square&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Berkay2002/stt-service?style=flat-square&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Berkay2002/stt-service?style=flat-square&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Berkay2002/stt-service?style=flat-square&color=0080ff" alt="repo-language-count">
</p>
<p>Built with the tools and technologies:</p>
<p>
	<img src="https://img.shields.io/badge/Flask-000000.svg?style=flat-square&logo=Flask&logoColor=white" alt="Flask">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat-square&logo=Docker&logoColor=white" alt="Docker">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
</p>
</div>
<br clear="right">

<details><summary>Table of Contents</summary>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

</details>
<hr>

##  Overview

stt-service is a high-performance, real-time speech-to-text microservice built with Python and FastAPI. It provides **enhanced partial and final transcription capabilities** with sub-300ms latency for conversational AI applications. The service features dual-mode processing, intelligent voice activity detection, and comprehensive monitoring for production deployment. Designed for seamless integration with conversational AI pipelines, it enables natural real-time interactions with advanced interrupt handling (barge-in) support.

---

##  Features

- **Partial and Final Transcription**: Real-time partial results (~300ms) with high-quality final transcriptions (~600ms)
- **Enhanced Voice Activity Detection**: Smart triggering with utterance boundary detection and barge-in support
- **Dual Processing Modes**: Optimized pipelines for speed (partial) vs. accuracy (final) with overlapping audio windows
- **WebSocket Real-time API**: FastAPI-based server with concurrent connections and intelligent buffering
- **Conversational AI Optimized**: Purpose-built for STT→LLM→TTS pipelines with interrupt handling
- **GPU-Accelerated Processing**: Whisper model integration with smart caching and resource management
- **Production Monitoring**: Comprehensive metrics, health checks, and performance analysis tools
- **Backward Compatible**: Legacy transcription API support for existing integrations
- **Docker & GPU Ready**: Complete containerization with CUDA support and scaling configurations
- **Performance Testing**: Built-in benchmarking tools for latency validation and optimization

---

##  Project Structure

```sh
└── stt-service/
    ├── Dockerfile
    ├── Dockerfile.gpu
    ├── app
    │   ├── __init__.py
    │   ├── core
    │   │   ├── audio_stream_processor.py    # Enhanced dual-mode processing
    │   │   ├── websocket_server.py          # FastAPI WebSocket server
    │   │   ├── whisper_handler.py           # Partial/final transcription
    │   │   └── microphone_capture.py
    │   ├── client_examples
    │   │   └── websocket_client_example.py  # Enhanced client with partial/final
    │   ├── main.py                          # CLI with WebSocket mode
    │   ├── monitoring
    │   └── utils
    ├── assets
    │   └── harvard.wav
    ├── docs
    │   ├── MONITORING.md
    │   ├── WebSocket_STT_Architecture.md    # Complete architecture guide
    │   └── PRODUCTION_MONITORING.md
    ├── main.py                              # Service entry point
    ├── test_partial_final_performance.py    # Performance validation
    ├── requirements.txt
    ├── scripts
    │   └── test_endpoints.bat
    └── tests
        ├── __init__.py
        ├── test_error_handler.py
        ├── test_logging.py
        ├── test_monitoring_simple.py
        └── test_realtime.py
```


###  Project Index
<details open>
	<summary><b><code>STT-SERVICE/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>Python package dependencies for running stt-service (see `requirements.txt`).</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/dockerfile-gpu.txt'>dockerfile-gpu.txt</a></b></td>
				<td><code>Optional GPU-enabled Dockerfile snippet for building with CUDA / GPU support.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/Dockerfile'>Dockerfile</a></b></td>
				<td><code>Production-ready container build for the microservice.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- scripts Submodule -->
		<summary><b>scripts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/scripts/test_endpoints.bat'>test_endpoints.bat</a></b></td>
				<td><code>Windows batch script to exercise HTTP endpoints for quick local testing.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- app Submodule -->
		<summary><b>app</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/main.py'>main.py</a></b></td>
				<td><code>Flask application entrypoint and CLI for running the service.</code></td>
			</tr>
			</table>
			<details>
				<summary><b>core</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/core/whisper_handler.py'>whisper_handler.py</a></b></td>
						<td><code>Model adapter that wraps Whisper (or compatible) transcription models.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/core/realtime_transcription.py'>realtime_transcription.py</a></b></td>
						<td><code>Implements realtime transcription flow and request handling utilities.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/core/microphone_capture.py'>microphone_capture.py</a></b></td>
						<td><code>Helpers for capturing audio from a microphone or system input device.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/core/audio_processor.py'>audio_processor.py</a></b></td>
						<td><code>Audio preprocessing utilities (resampling, chunking, normalization).</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>monitoring</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/monitoring/monitoring_service.py'>monitoring_service.py</a></b></td>
						<td><code>Provides monitoring endpoints and routines used by health checks and probes.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/monitoring/service_monitor.py'>service_monitor.py</a></b></td>
						<td><code>Background routines to collect service metrics and write JSON test outputs.</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>utils</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/utils/logger.py'>logger.py</a></b></td>
						<td><code>Logging configuration and helpers used throughout the service.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/utils/config.py'>config.py</a></b></td>
						<td><code>Configuration helpers (environment variables, default settings).</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/utils/error_handler.py'>error_handler.py</a></b></td>
						<td><code>Centralized error handling utilities and testable helpers.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Berkay2002/stt-service/blob/master/app/utils/connection_manager.py'>connection_manager.py</a></b></td>
						<td><code>Network and dependency connection helpers (external model backends, telemetry sinks).</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with stt-service, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip
- **Container Runtime:** Docker


###  Installation

Install stt-service using one of the following methods:

**Build from source:**

1. Clone the stt-service repository:
```sh
❯ git clone https://github.com/Berkay2002/stt-service
```

2. Navigate to the project directory:
```sh
❯ cd stt-service
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker build -t Berkay2002/stt-service .
```




###  Usage

#### WebSocket Real-time Server (Recommended)
```sh
# Start WebSocket server for real-time transcription
❯ python main.py websocket --port 8000

# With custom configuration
❯ python main.py websocket --host 0.0.0.0 --port 8080 --max-connections 100
```

#### File Processing Mode
```sh
# Process single audio file
❯ python main.py file input.wav

# With word-level timestamps
❯ python main.py file input.wav --timestamps
```

#### Microphone Recording Mode
```sh
# Record and transcribe from microphone
❯ python main.py microphone --timestamps
```

#### Configuration Management
```sh
# Show current configuration
❯ python main.py config --show
```

#### Docker Deployment
**CPU Version:**
```sh
❯ docker build -t stt-service .
❯ docker run -p 8000:8000 -p 9091:9091 stt-service
```

**GPU Version:**
```sh
❯ docker build -f Dockerfile.gpu -t stt-service-gpu .
❯ docker run --gpus all -p 8000:8000 -p 9091:9091 stt-service-gpu
```

**Docker Compose with GPU:**
```sh
❯ docker-compose -f docker-compose.gpu.yml up
```


###  Testing

#### Unit Tests
```sh
# Run all unit tests
❯ pytest

# Run specific test modules
❯ pytest tests/test_error_handler.py
❯ pytest tests/test_realtime.py
```

#### Performance Testing
```sh
# Test partial/final transcription performance
❯ python test_partial_final_performance.py

# Expected output:
# ✅ EXCELLENT: Partial latency < 300ms  
# ✅ EXCELLENT: Final latency < 600ms
# ✅ CONFIRMED: Partial transcription working
# ✅ CONFIRMED: Final transcription working
```

#### WebSocket Client Testing
```sh
# Interactive client with partial/final support
❯ python app/client_examples/websocket_client_example.py

# Choose option 1 for microphone testing
# Choose option 2 for test audio file
# Choose option 3 for connection stress test
```

#### API Endpoint Testing
```sh
# Test HTTP endpoints (Windows)
❯ scripts/test_endpoints.bat

# Test WebSocket health
❯ curl http://localhost:8000/health
❯ curl http://localhost:8000/stats
```

---

##  License

This project is protected under the MIT License.

---

##  Acknowledgments

Thanks to the open-source projects and communities that make this work possible:

- **OpenAI and the Whisper model authors** — for the foundational speech-to-text models and faster-whisper optimizations
- **FastAPI community** — for the high-performance async web framework enabling real-time WebSocket processing
- **Python audio ecosystem** — NumPy, soundfile, pyaudio, scipy for comprehensive audio processing capabilities
- **GPU acceleration libraries** — PyTorch, CUDA toolkit for efficient real-time transcription processing
- **WebSocket and async communities** — for enabling seamless real-time communication protocols

---

## Quick Start Guide

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Start WebSocket server:**
   ```sh
   python main.py websocket
   ```

3. **Test with performance validator:**
   ```sh
   python test_partial_final_performance.py
   ```

4. **Try interactive client:**
   ```sh
   python app/client_examples/websocket_client_example.py
   ```

### WebSocket Endpoints
- **Transcription**: `ws://localhost:8000/ws/transcribe`
- **Health Check**: `http://localhost:8000/health`
- **Statistics**: `http://localhost:8000/stats`
- **Monitoring**: `http://localhost:9091/health`

### Performance Targets
- **Partial Results**: <300ms average latency
- **Final Results**: <600ms average latency  
- **Concurrent Connections**: 50+ streams
- **Real-time Factor**: <0.3 GPU utilization