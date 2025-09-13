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

stt-service is a lightweight, production-oriented speech-to-text microservice built with Python and Flask. It provides realtime and batch transcription primitives, a simple monitoring layer, and utilities for audio capture and processing. The project aims to make it easy to run transcription pipelines locally or inside a container and to extend the codebase with different speech models or runtime integrations.

---

##  Features

- Realtime transcription endpoint and helper classes for microphone capture and audio processing.
- Pluggable model handler (example: Whisper-based handler) so models can be swapped without changing the surface API.
- Simple service monitoring and health checks with JSON-formatted test results for easy observability.
- Small, well-tested core with unit tests for error handling, logging, and basic monitoring flows.
- Docker-ready: includes container configuration and optional GPU Dockerfile variant.

---

##  Project Structure

```sh
└── stt-service/
    ├── Dockerfile
    ├── app
    │   ├── __init__.py
    │   ├── core
    │   ├── main.py
    │   ├── monitoring
    │   └── utils
    ├── assets
    │   └── harvard.wav
    ├── dockerfile-gpu.txt
    ├── docs
    │   ├── MONITORING.md
    │   └── PRODUCTION_MONITORING.md
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
Run stt-service using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker run -it {image_name}
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```

---

##  License

This project is protected under the MIT License.

---

##  Acknowledgments

Thanks to the open-source projects and communities that make this work possible:

- OpenAI and the Whisper model authors — for inspiration and reference implementations used in the model adapter.
- The Flask community — for a lightweight web framework that makes building microservices straightforward.
- Contributors to many Python audio libraries (NumPy, soundfile, pyaudio) which make audio processing achievable.

---

Notes

- Replace `{entrypoint}` with `app.main` or run `python -m app.main` to start the service locally.
- Replace `{image_name}` with the name you used when building the Docker image (for example `Berkay2002/stt-service`).