# WebSocket STT Server Architecture

## Fasttalk System Integration

The WebSocket STT Server is a critical component of the **Fasttalk** conversational AI system—a comprehensive real-time voice interaction platform that orchestrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) services to deliver seamless conversational AI experiences.

### Fasttalk System Overview

Fasttalk is designed as a modular, GPU-accelerated conversational AI platform that enables real-time voice interactions with advanced interrupt handling (barge-in) capabilities. The STT service documented here serves as the entry point for audio processing in the complete conversational pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASTTALK SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Web Browser   │◄───┤   Mobile App    │────┤  Audio Player   │        │
│  │  (Microphone)   │    │  (Microphone)   │    │  (TTS Output)   │        │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘        │
│            │                      │                      │                │
│            └──────────────────────┼──────────────────────┘                │
│                                   │                                       │
│  ┌─────────────────────────────────▼─────────────────────────────────────┐  │
│  │                    CORE BACKEND SERVICE                               │  │
│  │                                                                       │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────────┐│  │
│  │  │   WebSocket      │  │ Core Interaction│  │  Configuration        ││  │
│  │  │  Communication  │  │     Logic       │  │   Management API      ││  │
│  │  │   Interface     │  │                 │  │                       ││  │
│  │  └─────────┬────────┘  └─────────┬───────┘  └───────────────────────┘│  │
│  │            │                     │                                   │  │
│  │            │     ┌───────────────▼──────────────┐                   │  │
│  │            │     │   BARGE-IN INTERRUPT CONTROL │                   │  │
│  │            │     │   (User Interrupt Handling)  │                   │  │
│  │            │     └─────────────┬──────────────────┘                 │  │
│  └────────────┼─────────────────────┼─────────────────────────────────────┘  │
│               │                     │                                        │
│  ┌────────────▼─────────────────────▼────────────────────────────────────┐  │
│  │                AI MODELS SERVICE PIPELINE                             │  │
│  │                  (GPU Docker Containers)                              │  │
│  │                                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐   │  │
│  │  │ STT MODULE  │───▶│ LLM MODULE  │───▶│      TTS MODULE         │   │  │
│  │  │             │    │             │    │                         │   │  │
│  │  │ • Voice     │    │ • Response  │    │ • Audio Synthesis       │   │  │
│  │  │   Activity  │    │   Generation│    │ • Streaming Buffer      │   │  │
│  │  │   Detection │    │ • Token     │    │ • Audio Splicing        │   │  │
│  │  │ • Real-time │    │   Streaming │    │ • Smooth Playback       │   │  │
│  │  │   Audio     │    │ • Context   │    │                         │   │  │
│  │  │   Transcr.  │    │   Awareness │    │                         │   │  │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │ Web Dashboard   │    │        Observability & Benchmarking             │ │
│  │                 │    │                                                 │ │
│  │ • System Config │    │ • Automated Testing    • Metrics Collection    │ │
│  │ • Debugging     │    │ • Performance Bench.   • Visualization         │ │
│  │ • Monitoring    │    │ • Health Monitoring    • Alert Management      │ │
│  └─────────────────┘    └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

The STT service participates in a sophisticated data flow pipeline that enables real-time conversational AI:

```
┌──────────────┐     Audio Stream     ┌──────────────┐     Audio Stream     ┌──────────────┐
│ User Client  │────────────────────▶ │ Core Backend │────────────────────▶ │ STT Service  │
│   (Browser)  │                      │   Service    │                      │              │
└──────────────┘                      └──────┬───────┘                      └──────┬───────┘
       ▲                                     │                                     │
       │                                     │                                     │
       │ Audio Stream                        │ Text Prompt                         │ Transcribed Text
       │                                     ▼                                     ▼
┌──────┴───────┐                      ┌──────────────┐     Text Prompt     ┌──────────────┐
│ TTS Service  │◄─────────────────────│ Core Backend │────────────────────▶│ LLM Service  │
│              │     Text Stream      │   Service    │                      │              │
└──────────────┘                      └──────▲───────┘                      └──────┬───────┘
                                             │                                     │
                                             │ LLM Token Stream                    │ Response Tokens
                                             └─────────────────────────────────────┘

                                    BARGE-IN INTERRUPT FLOW:
┌──────────────┐  User Interrupt   ┌──────────────┐  Stop Signal   ┌─────────────────────┐
│ User Client  │──────────────────▶│ Core Backend │───────────────▶│ All AI Services    │
│              │                   │   Service    │                │ (STT, LLM, TTS)     │
└──────────────┘                   └──────────────┘                └─────────────────────┘
```

### STT Service Role in Fasttalk

The WebSocket STT Server fulfills several critical roles within the Fasttalk ecosystem:

1. **Audio Gateway**: Primary entry point for all audio data from user clients
2. **Real-time Processing**: Converts speech to text with minimal latency for conversational flow
3. **Voice Activity Detection**: Intelligently detects speech boundaries and silence
4. **Interrupt Detection**: Supports barge-in functionality by detecting user interruptions
5. **Stream Coordination**: Coordinates with Core Backend Service for seamless pipeline integration

### Integration Points

#### Core Backend Service Integration
The STT service integrates directly with the Core Backend Service, which:
- Manages WebSocket connections to user clients
- Orchestrates the complete STT→LLM→TTS pipeline
- Implements barge-in interrupt control across all AI services
- Handles session management and conversation state
- Provides configuration management for all AI components

#### AI Models Service Pipeline Integration
As part of the GPU Docker container ecosystem, the STT service:
- Shares GPU resources with LLM and TTS services
- Participates in interrupt signaling for barge-in scenarios
- Contributes to system-wide performance metrics and monitoring
- Supports coordinated scaling and resource management

## STT Service Technical Overview

The WebSocket STT Server is a high-performance, real-time speech-to-text service built on FastAPI and designed to integrate with GPU-accelerated Whisper models. It provides concurrent WebSocket connections, intelligent audio buffering, comprehensive error handling, and seamless integration with the broader Fasttalk monitoring infrastructure.

## STT Service Architecture Components

### 1. Core Server Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Core Backend  │────│  WebSocket STT   │────│   Whisper GPU   │
│    Service      │    │     Server       │    │    Container    │
│   (Fasttalk)    │    │   (This Service) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │ Barge-in Control       │ Real-time Processing   │ GPU Inference
         │ Session Management     │ Voice Activity Det.    │ Model Loading
         │                        │                        │
                    ┌─────────────┼─────────────┐
                    │             │             │
            ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼──────┐
            │ Connection  │ │   Audio   │ │ Monitoring │
            │  Manager    │ │  Stream   │ │ & Health   │
            │             │ │ Processor │ │   Checks   │
            │• WebSocket  │ │• VAD      │ │• Metrics   │
            │• Sessions   │ │• Buffering│ │• Status    │
            │• Lifecycle  │ │• Chunking │ │• Error     │
            │• Interrupts │ │• Triggers │ │  Tracking  │
            └─────────────┘ └───────────┘ └────────────┘
```

#### **WebSocketSTTServer** (`app/core/websocket_server.py`)
- Main FastAPI application
- WebSocket endpoint management
- Health and statistics endpoints
- Integration with monitoring services

#### **ConnectionManager**
- Manages multiple concurrent WebSocket connections
- Connection lifecycle management
- Session state tracking
- Integration with error handling and monitoring

#### **AudioStreamProcessor** (`app/core/audio_stream_processor.py`)
- Real-time audio buffering and processing
- Voice Activity Detection (VAD)
- Intelligent chunk processing
- Background transcription triggering

#### **WebSocketErrorHandler** (`app/core/error_handler.py`)
- Comprehensive error categorization
- Circuit breaker patterns
- Retry logic with exponential backoff
- Error statistics and recovery strategies

### 2. Integration Components

#### **WhisperHandler Integration**
- GPU-optimized transcription processing
- Support for both simple and timestamp-enabled transcription
- Configurable model parameters
- Thread-pool executor for async execution

#### **ServiceMonitor Integration**
- Real-time performance metrics
- Connection statistics
- Health status monitoring
- Resource utilization tracking

## WebSocket Message Protocol

### Client to Server Messages

#### Audio Data (Binary)
```
Binary WebSocket frame containing:
- Raw PCM audio data (float32, 16kHz, mono)
- Expected chunk size: 512-2048 samples
- Continuous streaming supported
```

#### Control Messages (JSON)
```json
{
  "type": "start_session",
  "session_config": {
    "language": "en",           // Optional: Language hint
    "enable_timestamps": true,  // Include word-level timestamps
    "enable_vad": true,         // Voice Activity Detection
    "buffer_duration": 2.0      // Audio buffer duration (seconds)
  }
}

{
  "type": "end_session"
}

{
  "type": "flush_buffer"  // Force immediate transcription
}

{
  "type": "configure",
  "config": {
    "buffer_duration": 1.5
  }
}
```

### Server to Client Messages

#### Transcription Results
```json
{
  "type": "transcription",
  "data": {
    "text": "Hello world",
    "processing_time_ms": 250,
    "audio_duration_ms": 2000,
    "timestamps": [
      {
        "word": "Hello",
        "start_ms": 0,
        "end_ms": 500,
        "confidence": 0.95
      },
      {
        "word": "world",
        "start_ms": 600,
        "end_ms": 1100,
        "confidence": 0.92
      }
    ],
    "metadata": {
      "duration": 2.0,
      "chunks_count": 4,
      "reason": "buffer_full"
    }
  },
  "metadata": {
    "session_id": "uuid-here",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Status Messages
```json
{
  "type": "session_started",
  "session_id": "uuid-here",
  "timestamp": "2024-01-15T10:30:00Z"
}

{
  "type": "error",
  "error": {
    "code": "audio_processing_error",
    "message": "Invalid audio format",
    "category": "audio_format",
    "severity": "low",
    "recoverable": true,
    "retry_after": 1.0
  }
}
```

## Connection Lifecycle Management

### Connection States
- **connecting**: Initial connection establishment
- **active**: Ready to receive audio and control messages
- **processing**: Currently processing transcription
- **buffering**: Accumulating audio for processing
- **disconnecting**: Graceful disconnection in progress
- **error**: Error state requiring recovery

### Lifecycle Flow
1. **Connection**: Client connects to `/ws/transcribe`
2. **Session Start**: Client sends `start_session` with configuration
3. **Audio Streaming**: Client sends continuous audio chunks
4. **Processing**: Server processes audio when conditions are met
5. **Results**: Server sends transcription results
6. **Session End**: Client sends `end_session` or disconnects

## Error Handling Strategy

### Error Categories
- **Connection**: WebSocket connection issues
- **Audio Format**: Invalid audio data or format
- **Processing**: Transcription processing errors
- **GPU**: GPU memory or processing issues
- **Resource**: System resource exhaustion
- **Timeout**: Processing timeout errors

### Recovery Mechanisms

#### Circuit Breaker Pattern
```python
# GPU Circuit Breaker
- Failure Threshold: 3 consecutive GPU errors
- Timeout: 5 minutes
- States: closed → open → half-open → closed

# Transcription Circuit Breaker
- Failure Threshold: 5 consecutive failures
- Timeout: 2 minutes
- Protects against model loading issues
```

#### Retry Strategy
```python
# Connection Errors
- Max Retries: 3
- Base Delay: 0.5s
- Exponential backoff

# Processing Errors
- Max Retries: 2
- Base Delay: 1.0s
- With GPU resource awareness
```

## Performance Characteristics

### Concurrency
- **Max Connections**: 50 (configurable)
- **GPU Serialization**: Single semaphore for GPU access
- **Connection Isolation**: Each connection has independent audio processing

### Real-time Performance
- **Latency Target**: <500ms for 2-second audio chunks
- **Throughput**: 10+ concurrent real-time streams
- **Buffer Management**: Intelligent VAD-based processing

### Resource Management
- **Memory**: Automatic buffer trimming and cleanup
- **GPU**: Shared Whisper model with serialized access
- **CPU**: Async processing with thread pool executors

## Monitoring Integration

### Connection Statistics
```json
{
  "active_connections": 5,
  "total_connections": 127,
  "connections_by_state": {
    "active": 3,
    "processing": 2,
    "buffering": 0
  },
  "processing_stats": {
    "session-id-1": {
      "total_chunks": 45,
      "total_processing_time": 12.5,
      "average_latency": 0.28,
      "real_time_factor": 0.15
    }
  }
}
```

### Health Endpoints
- **`/health`**: Overall service health
- **`/stats`**: Detailed service statistics
- **`/`**: Service information and endpoints
- **`:9091/health`**: Monitoring service health

## Deployment Considerations

### Docker Integration

#### Environment Variables
```bash
# WebSocket Configuration
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8000
WEBSOCKET_MAX_CONNECTIONS=50

# Error Handling
TRANSCRIPTION_FAILURE_THRESHOLD=5
GPU_FAILURE_THRESHOLD=3
MAX_SESSION_ERRORS=10

# Processing Configuration
BUFFER_DURATION=2.0
MIN_AUDIO_LENGTH=0.5
MAX_SILENCE_DURATION=2.0
```

#### Docker Compose Integration
```yaml
services:
  stt-websocket:
    build: .
    ports:
      - "8000:8000"
      - "9091:9091"
    environment:
      - WEBSOCKET_HOST=0.0.0.0
      - WEBSOCKET_PORT=8000
    volumes:
      - ./logs:/app/logs
    depends_on:
      - whisper-gpu
```

### Production Deployment

#### Scaling Considerations
- **Horizontal Scaling**: Multiple container instances with load balancer
- **GPU Sharing**: Single GPU per container for optimal performance
- **Connection Stickiness**: WebSocket connections require sticky sessions

#### Security
- **CORS Configuration**: Restrict origins in production
- **Rate Limiting**: Implement connection and message rate limits
- **Authentication**: Add JWT or API key authentication if required

#### Monitoring
- **Prometheus Metrics**: Export connection and performance metrics
- **Log Aggregation**: Structured JSON logging for centralized collection
- **Health Checks**: Kubernetes/Docker health check endpoints

## Usage Examples

### Basic Client Connection
```python
import websockets
import asyncio
import json

async def connect_and_transcribe():
    uri = "ws://localhost:8000/ws/transcribe"

    async with websockets.connect(uri) as websocket:
        # Start session
        await websocket.send(json.dumps({
            "type": "start_session",
            "session_config": {
                "enable_timestamps": True,
                "buffer_duration": 2.0
            }
        }))

        # Send audio chunks
        # ... audio processing code ...

        # End session
        await websocket.send(json.dumps({"type": "end_session"}))
```

### Integration with Fasttalk Core Backend

The STT service integrates seamlessly with the Core Backend Service through a well-defined WebSocket protocol that supports the complete conversational AI pipeline.

#### Core Backend Integration Pattern
```python
# Fasttalk Core Backend Service Integration
class FasttalkSTTIntegration:
    def __init__(self, stt_endpoint="ws://stt-service:8000/ws/transcribe"):
        self.stt_endpoint = stt_endpoint
        self.current_session = None
        self.barge_in_handler = None

    async def start_conversation_session(self, user_id: str, config: dict):
        """Initialize STT for a new conversation session"""
        async with websockets.connect(self.stt_endpoint) as ws:
            self.current_session = ws

            # Configure for Fasttalk conversation mode
            await ws.send(json.dumps({
                "type": "start_session",
                "session_config": {
                    "enable_timestamps": False,  # Real-time mode
                    "enable_vad": True,         # Voice Activity Detection
                    "buffer_duration": 1.0,     # Low latency
                    "fasttalk_mode": True,      # Enable Fasttalk features
                    "user_id": user_id,
                    "conversation_id": config.get("conversation_id")
                }
            }))

            return await self._handle_conversation_loop(ws)

    async def _handle_conversation_loop(self, ws):
        """Main conversation loop with barge-in support"""
        async def audio_processor():
            async for message in ws:
                result = json.loads(message)

                if result["type"] == "transcription":
                    text = result["data"]["text"]

                    # Send to LLM service via Core Backend
                    await self.send_to_llm_pipeline(text, result["metadata"])

                elif result["type"] == "voice_activity_detected":
                    # Handle barge-in: user starting to speak
                    await self.handle_user_interrupt(result["data"])

        return audio_processor()

    async def handle_user_interrupt(self, vad_data):
        """Handle barge-in interruption"""
        # Signal all AI services to stop current processing
        await self.signal_barge_in_interrupt({
            "type": "user_interrupt",
            "confidence": vad_data.get("confidence", 0.8),
            "timestamp": vad_data.get("timestamp"),
            "session_id": self.current_session.session_id
        })

    async def send_audio_stream(self, audio_stream):
        """Stream audio to STT service"""
        if not self.current_session:
            raise RuntimeError("No active STT session")

        async for chunk in audio_stream:
            # Send raw audio data to STT service
            await self.current_session.send(chunk.tobytes())

    async def signal_barge_in_interrupt(self, interrupt_data):
        """Send interrupt signal to all AI services"""
        # Notify LLM service to stop generation
        await self.llm_service.interrupt_generation(interrupt_data)

        # Notify TTS service to stop synthesis
        await self.tts_service.interrupt_synthesis(interrupt_data)

        # Clear any pending audio buffers
        await self.audio_buffer_manager.clear_buffers()
```

#### Fasttalk-Specific WebSocket Extensions

The STT service supports additional message types for Fasttalk integration:

```json
// Barge-in detection message
{
  "type": "voice_activity_detected",
  "data": {
    "confidence": 0.85,
    "duration_ms": 200,
    "energy_level": 0.7,
    "is_speech": true
  },
  "metadata": {
    "session_id": "uuid-here",
    "timestamp": "2024-01-15T10:30:00Z",
    "conversation_id": "conv-uuid"
  }
}

// Interrupt acknowledgment
{
  "type": "interrupt_acknowledged",
  "data": {
    "interrupted_at": "2024-01-15T10:30:00Z",
    "buffer_cleared": true,
    "processing_stopped": true
  }
}

// Conversation context update
{
  "type": "context_update",
  "data": {
    "conversation_state": "listening",  // listening, processing, responding
    "expected_response_type": "question", // question, command, casual
    "language_context": "en-US"
  }
}
```

## Performance Tuning

### Audio Processing
- **Chunk Size**: 512-2048 samples optimal for real-time
- **Buffer Duration**: 1.0-3.0 seconds based on latency requirements
- **VAD Sensitivity**: Tune for environment and use case

### GPU Optimization
- **Model Size**: Balance accuracy vs. speed (base.en vs. large-v3)
- **Batch Size**: Single sample processing for real-time
- **Memory Management**: Monitor VRAM usage under load

### Connection Management
- **Max Connections**: Scale based on GPU capacity and RAM
- **Timeout Settings**: Balance responsiveness with stability
- **Error Thresholds**: Adjust based on expected error rates

## Core Backend Service Integration Guide

This section provides specific guidance for backend teams integrating with the STT service within the Fasttalk ecosystem.

### Integration Checklist for Backend Teams

#### Prerequisites
- [ ] Fasttalk Core Backend Service deployed and accessible
- [ ] WebSocket client library with async support
- [ ] GPU-enabled STT service containers
- [ ] Redis/shared state store for session management
- [ ] Monitoring and alerting infrastructure

#### Integration Steps

1. **Establish Service Discovery**
```python
# Service discovery for STT integration
class FasttalkServiceDiscovery:
    def __init__(self):
        self.service_registry = {
            "stt-service": "ws://stt-service:8000/ws/transcribe",
            "llm-service": "http://llm-service:8001/generate",
            "tts-service": "http://tts-service:8002/synthesize"
        }

    async def get_stt_endpoint(self, load_balance=True):
        if load_balance:
            return await self.get_least_loaded_stt_instance()
        return self.service_registry["stt-service"]
```

2. **Implement Conversation Session Management**
```python
class ConversationSessionManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.active_sessions = {}

    async def create_conversation_session(self, user_id, config):
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            stt_config=config.get("stt", {}),
            llm_config=config.get("llm", {}),
            tts_config=config.get("tts", {}),
            created_at=datetime.utcnow()
        )

        # Store session state
        await self.redis.setex(
            f"conversation:{session_id}",
            3600,  # 1 hour TTL
            session.to_json()
        )

        self.active_sessions[session_id] = session
        return session
```

3. **Handle Real-time Audio Streaming**
```python
async def handle_user_audio_stream(self, session_id, audio_websocket):
    """Handle incoming audio from user client"""
    session = await self.get_session(session_id)
    stt_connection = await self.connect_to_stt_service(session)

    try:
        # Start STT session
        await stt_connection.send(json.dumps({
            "type": "start_session",
            "session_config": {
                "fasttalk_mode": True,
                "session_id": session_id,
                "enable_vad": True,
                "barge_in_enabled": True,
                "buffer_duration": 1.0
            }
        }))

        # Stream audio to STT service
        audio_task = asyncio.create_task(
            self.forward_audio_to_stt(audio_websocket, stt_connection)
        )

        # Process STT results
        result_task = asyncio.create_task(
            self.process_stt_results(stt_connection, session)
        )

        # Handle both simultaneously
        await asyncio.gather(audio_task, result_task)

    except Exception as e:
        await self.handle_conversation_error(session_id, e)
    finally:
        await stt_connection.close()
```

4. **Implement Barge-in Interrupt Handling**
```python
async def handle_interrupt_signal(self, session_id, interrupt_data):
    """Handle user barge-in interruption"""
    session = self.active_sessions.get(session_id)
    if not session:
        return

    # Stop all active AI processing
    tasks_to_cancel = [
        session.llm_task,
        session.tts_task,
        session.audio_playback_task
    ]

    for task in tasks_to_cancel:
        if task and not task.done():
            task.cancel()

    # Clear audio buffers
    await session.audio_buffer.clear()

    # Reset conversation state
    session.state = "listening"
    session.last_interrupt = datetime.utcnow()

    # Log interrupt event
    await self.log_interrupt_event(session_id, interrupt_data)

    # Notify user client of interrupt acknowledgment
    await session.user_websocket.send(json.dumps({
        "type": "interrupt_acknowledged",
        "timestamp": datetime.utcnow().isoformat()
    }))
```

### Error Handling Best Practices

```python
class FasttalkErrorHandler:
    async def handle_stt_service_error(self, error, session_id):
        """Handle STT service specific errors"""
        if error.code == "gpu_memory_exhaustion":
            # Try to restart with smaller model
            await self.restart_stt_with_fallback_model(session_id)
        elif error.code == "websocket_disconnected":
            # Attempt reconnection with exponential backoff
            await self.reconnect_to_stt_service(session_id)
        elif error.code == "transcription_timeout":
            # Skip current audio chunk and continue
            await self.skip_current_audio_chunk(session_id)

    async def handle_conversation_pipeline_error(self, error, session_id):
        """Handle errors in the complete conversation pipeline"""
        session = await self.get_session(session_id)

        # Graceful degradation strategies
        if error.affects_services(["stt", "llm"]):
            # Fall back to text-only mode
            await self.enable_text_only_mode(session)
        elif error.affects_service("tts"):
            # Continue with text responses only
            await self.disable_voice_responses(session)
```

### Performance Optimization for Backend Teams

```python
# Connection pooling for STT services
class STTConnectionPool:
    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        self.available_connections = asyncio.Queue(maxsize=pool_size)
        self.total_connections = 0

    async def get_connection(self):
        """Get a connection from the pool"""
        try:
            return await asyncio.wait_for(
                self.available_connections.get(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # Create new connection if pool is full
            return await self.create_new_connection()

    async def return_connection(self, connection):
        """Return connection to pool"""
        if connection.is_healthy():
            await self.available_connections.put(connection)
        else:
            # Replace unhealthy connection
            await connection.close()
            new_conn = await self.create_new_connection()
            await self.available_connections.put(new_conn)
```

### Testing and Validation

```python
# Integration tests for STT service
class TestSTTIntegration:
    async def test_basic_transcription(self):
        """Test basic transcription flow"""
        session = await self.create_test_session()
        audio_data = self.load_test_audio("hello_world.wav")

        result = await self.send_audio_to_stt(session, audio_data)

        assert result["type"] == "transcription"
        assert "hello world" in result["data"]["text"].lower()
        assert result["data"]["processing_time_ms"] < 500

    async def test_barge_in_functionality(self):
        """Test interrupt handling"""
        session = await self.create_test_session()

        # Start long audio processing
        long_audio = self.load_test_audio("long_speech.wav")
        processing_task = asyncio.create_task(
            self.send_audio_to_stt(session, long_audio)
        )

        # Send interrupt signal
        await asyncio.sleep(1.0)  # Let processing start
        interrupt_audio = self.load_test_audio("interrupt.wav")
        await self.send_interrupt_audio(session, interrupt_audio)

        # Verify interrupt was handled
        result = await processing_task
        assert result["type"] == "voice_activity_detected"
        assert result["data"]["is_speech"] == True
```

## Conclusion

This WebSocket STT Server serves as a crucial component in the Fasttalk conversational AI ecosystem, providing high-performance, real-time speech-to-text capabilities with sophisticated barge-in interrupt handling. The architecture is designed to seamlessly integrate with the broader Fasttalk system while maintaining the flexibility to operate as a standalone service.

### Key Features Summary

- **Real-time Processing**: Sub-500ms latency for conversational AI applications
- **Barge-in Support**: Advanced interrupt detection and handling for natural conversations
- **Fasttalk Integration**: Purpose-built for the complete STT→LLM→TTS pipeline
- **Scalable Architecture**: Support for 100+ concurrent conversations with GPU optimization
- **Production Ready**: Comprehensive error handling, monitoring, and high availability support
- **Developer Friendly**: Complete integration guide and testing framework for backend teams

The service provides a robust, scalable foundation for real-time speech-to-text capabilities while maintaining seamless integration with the complete Fasttalk conversational AI ecosystem.

This comprehensive architecture documentation provides backend teams with all the information needed to successfully integrate with and deploy the STT service within the Fasttalk ecosystem, ensuring optimal performance for real-time conversational AI applications.