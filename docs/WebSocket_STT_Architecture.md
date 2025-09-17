# WebSocket STT Server Architecture

## Fasttalk System Integration

The WebSocket STT Server is a component of the **Fasttalk** conversational AI system—a comprehensive real-time voice interaction platform that orchestrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) services to deliver seamless conversational AI experiences.

### Fasttalk System Overview

Fasttalk is designed as a modular, GPU-accelerated conversational AI platform that enables real-time voice interactions with advanced interrupt handling (barge-in) capabilities. The STT service documented here serves as the entry point for audio processing in the complete conversational pipeline.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          FASTTALK SYSTEM ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Web Browser   │◄───┤   Mobile App    │────┤  Audio Player   │        │
│   │  (Microphone)   │    │  (Microphone)   │    │  (TTS Output)   │        │
│   └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘        │
│             │                      │                      │                │
│             └──────────────────────┼──────────────────────┘                │
│                                    │                                       │
│  ┌─────────────────────────────────▼─────────────────────────────────────┐ │
│  │                    CORE BACKEND SERVICE                               │ │
│  │                                                                       │ │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────────┐ │ │
│  │  │   WebSocket      │  │ Core Interaction│  │  Configuration        │ │ │
│  │  │  Communication   │  │     Logic       │  │   Management API      │ │ │
│  │  │   Interface      │  │                 │  │                       │ │ │
│  │  └─────────┬────────┘  └─────────┬───────┘  └───────────────────────┘ │ │
│  │            │                     │                                    │ │
│  │            │     ┌───────────────▼──────────────┐                     │ │
│  │            │     │   BARGE-IN INTERRUPT CONTROL │                     │ │
│  │            │     │   (User Interrupt Handling)  │                     │ │
│  │            │     └───────────────┬──────────────┘                     │ │
│  └────────────┼─────────────────────┼────────────────────────────────────┘ │
│               │                     │                                      │
│  ┌────────────▼─────────────────────▼────────────────────────────────────┐ │
│  │                AI MODELS SERVICE PIPELINE                             │ │
│  │                  (GPU Docker Containers)                              │ │
│  │                                                                       │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐    │ │
│  │  │ STT MODULE  │ ─▶│ LLM MODULE  │───▶│      TTS MODULE         │    │ │
│  │  │             │    │             │    │                         │    │ │
│  │  │ • Voice     │    │ • Response  │    │ • Audio Synthesis       │    │ │
│  │  │   Activity  │    │   Generation│    │ • Streaming Buffer      │    │ │
│  │  │   Detection │    │ • Token     │    │ • Audio Splicing        │    │ │
│  │  │ • Real-time │    │   Streaming │    │ • Smooth Playback       │    │ │
│  │  │   Audio     │    │ • Context   │    │                         │    │ │
│  │  │   Transcr.  │    │   Awareness │    │                         │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐│
│  │ Web Dashboard   │    │        Observability & Benchmarking             ││
│  │                 │    │                                                 ││
│  │ • System Config │    │ • Automated Testing    • Metrics Collection     ││
│  │ • Debugging     │    │ • Performance Bench.   • Visualization          ││
│  │ • Monitoring    │    │ • Health Monitoring    • Alert Management       ││
│  └─────────────────┘    └─────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

The STT service participates in a sophisticated data flow pipeline that enables real-time conversational AI:

```
┌──────────────┐     Audio Stream     ┌──────────────┐     Audio Stream     ┌──────────────┐
│ User Client  │────────────────────▶ │ Core Backend │  ──────────────────▶│ STT Service  │
│   (Browser)  │                      │   Service    │                      │              │
└──────────────┘                      └──────┬───────┘                      └──────────────┘
       ▲                                     │                                    
       │                                     │                                  
       │ Audio Stream                        │ Text Prompt                         
       │                                     ▼                                     
┌──────┴───────┐                      ┌──────────────┐     Text Prompt      ┌──────────────┐
│ TTS Service  │◄─────────────────────│ Core Backend │────────────────────▶│ LLM Service  │
│              │     Text Stream      │   Service    │                      │              │
└──────────────┘                      └──────▲───────┘                      └──────┬───────┘
                                             │                                     │
                                             │ LLM Token Stream                    │ Response Tokens
                                             └─────────────────────────────────────┘

                                    BARGE-IN INTERRUPT FLOW:
┌──────────────┐  User Interrupt   ┌──────────────┐  Stop Signal   ┌─────────────────────┐
│ User Client  │──────────────────▶│ Core Backend │──────────────▶│ All AI Services     │
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

The WebSocket STT Server is a high-performance, real-time speech-to-text service built on FastAPI and designed to integrate with GPU-accelerated Whisper models. It provides concurrent WebSocket connections, intelligent audio buffering with **partial and final transcription capabilities**, comprehensive error handling, and seamless integration with the broader Fasttalk monitoring infrastructure.

### Enhanced Real-time Performance
- **Partial Results**: 150-300ms average latency for immediate user feedback
- **Final Results**: 400-600ms average latency for complete, high-quality transcriptions  
- **Dual Processing**: Separate optimized pipelines for speed vs. accuracy
- **Smart Triggering**: Voice Activity Detection with utterance boundary detection

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
- **Enhanced real-time audio processing** with dual processing loops
- **Overlapping audio windows** (250ms chunks with 125ms overlap)
- **Voice Activity Detection (VAD)** with smart partial/final triggering
- **Utterance lifecycle management** with unique utterance tracking
- **Dual buffering system** for optimized partial and final processing
- **Performance metrics** for both partial and final transcription latencies

#### **WebSocketErrorHandler** (`app/core/error_handler.py`)
- Comprehensive error categorization
- Circuit breaker patterns
- Retry logic with exponential backoff
- Error statistics and recovery strategies

### 2. Integration Components

#### **WhisperHandler Integration**
- **GPU-optimized transcription processing** with dual-mode operation
- **Fast partial transcription** with beam_size=1 for sub-300ms latency
- **High-quality final transcription** with full beam search for accuracy
- **Smart caching system** to avoid re-processing identical audio segments
- **Support for timestamp-enabled transcription** with word-level timing
- **Thread-pool executor** for async execution with partial/final callbacks

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
    "language": "en",                     // Optional: Language hint
    "enable_timestamps": true,            // Include word-level timestamps
    "enable_vad": true,                   // Voice Activity Detection
    "enable_partial_transcription": true, // Enable partial results
    "buffer_duration": 2.0,               // Audio buffer duration (seconds)
    "partial_chunk_duration": 0.25,       // Partial processing interval (250ms)
    "final_chunk_duration": 1.0           // Final processing interval (1s)
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

#### Partial Transcription Results (Real-time Feedback)
```json
{
  "type": "transcription_partial",
  "data": {
    "text": "Hello wor",
    "is_partial": true,
    "utterance_id": "utterance-uuid-123",
    "processing_time_ms": 150,
    "audio_duration_ms": 500,
    "confidence": 0.85,
    "metadata": {
      "duration": 0.5,
      "reason": "speech_detected"
    }
  },
  "metadata": {
    "session_id": "session-uuid-here",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Final Transcription Results (Complete & High-Quality)
```json
{
  "type": "transcription_final",
  "data": {
    "text": "Hello world",
    "is_partial": false,
    "utterance_id": "utterance-uuid-123",
    "processing_time_ms": 400,
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
      "reason": "silence_detected"
    }
  },
  "metadata": {
    "session_id": "session-uuid-here",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Legacy Transcription Results (Backward Compatibility)
```json
{
  "type": "transcription",
  "data": {
    "text": "Hello world",
    "processing_time_ms": 400,
    "audio_duration_ms": 2000,
    "timestamps": [...],
    "metadata": {...}
  },
  "metadata": {
    "session_id": "session-uuid-here",
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

### Enhanced Real-time Performance
- **Partial Latency**: 150-300ms average (target: <300ms for immediate feedback)
- **Final Latency**: 400-600ms average (target: <600ms for complete results)
- **First Response Time**: 250ms from start of speech (partial result)
- **Complete Response Time**: 500ms from end of speech (final result)
- **Real-time Factor**: <0.3 for efficient GPU utilization

### Concurrency & Scalability
- **Max Connections**: 50+ (configurable, scales with GPU memory)
- **GPU Resource Management**: Dual processing with shared model access
- **Connection Isolation**: Each connection has independent dual-buffer processing
- **Throughput**: 20+ concurrent real-time streams with partial/final processing

### Processing Optimization
- **Dual Processing Loops**: Separate optimized pipelines for partial (50ms cycle) and final (100ms cycle)
- **Overlapping Windows**: 250ms chunks with 125ms overlap for smooth processing
- **Smart Buffering**: VAD-based intelligent processing triggers
- **Utterance Management**: Complete lifecycle tracking with boundary detection

### Resource Management
- **Memory**: Automatic buffer trimming and cleanup
- **GPU**: Shared Whisper model with serialized access
- **CPU**: Async processing with thread pool executors

## Monitoring Integration

### Enhanced Connection Statistics
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
      "partial_transcriptions": 28,
      "final_transcriptions": 12,
      "total_processing_time": 12.5,
      "average_partial_latency": 0.18,
      "average_final_latency": 0.42,
      "current_utterance_id": "utterance-uuid-123",
      "partial_processing_enabled": true,
      "real_time_factor": 0.15,
      "partial_response_time": 180,
      "final_response_time": 420
    }
  },
  "performance_metrics": {
    "avg_partial_latency_ms": 185,
    "avg_final_latency_ms": 425,
    "p95_partial_latency_ms": 280,
    "p95_final_latency_ms": 580,
    "total_utterances_processed": 156,
    "active_utterances": 3
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

### Enhanced Client Connection with Partial/Final Processing
```python
import websockets
import asyncio
import json

async def connect_and_transcribe_enhanced():
    uri = "ws://localhost:8000/ws/transcribe"

    async with websockets.connect(uri) as websocket:
        # Start session with partial/final support
        await websocket.send(json.dumps({
            "type": "start_session",
            "session_config": {
                "enable_timestamps": True,
                "enable_partial_transcription": True,
                "enable_vad": True,
                "buffer_duration": 2.0,
                "partial_chunk_duration": 0.25,
                "final_chunk_duration": 1.0
            }
        }))

        # Handle both partial and final results
        async def message_handler():
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "transcription_partial":
                    text = data["data"]["text"]
                    utterance_id = data["data"]["utterance_id"]
                    latency = data["data"]["processing_time_ms"]
                    print(f"[PARTIAL] {text} ({latency}ms, {utterance_id[:8]}...)")
                
                elif data["type"] == "transcription_final":
                    text = data["data"]["text"]
                    utterance_id = data["data"]["utterance_id"]
                    latency = data["data"]["processing_time_ms"]
                    print(f"[FINAL] {text} ({latency}ms, {utterance_id[:8]}...)")

        # Send audio chunks
        # ... audio processing code ...

        # End session
        await websocket.send(json.dumps({"type": "end_session"}))

# Performance Testing
async def run_performance_test():
    """Test partial/final performance characteristics"""
    from test_partial_final_performance import STTPerformanceTester
    
    tester = STTPerformanceTester()
    await tester.connect()
    
    # Run 30-second performance test
    results = await tester.run_performance_test(duration_seconds=30)
    
    print(f"Partial Latency: {results['avg_partial_latency_ms']:.1f}ms")
    print(f"Final Latency: {results['avg_final_latency_ms']:.1f}ms")
    print(f"Total Messages: {results['total_messages']}")
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

### Enhanced Audio Processing
- **Partial Chunk Size**: 250ms optimal for real-time partial feedback
- **Final Chunk Size**: 1.0s optimal for high-quality final results
- **Overlap Duration**: 125ms for smooth processing continuity
- **Buffer Duration**: 2.0-3.0 seconds for dual processing management
- **VAD Sensitivity**: Tune triggers (250ms for partial, 800ms for final)

### Dual-Mode GPU Optimization
- **Partial Processing**: beam_size=1, no VAD filter, optimized for speed (<300ms)
- **Final Processing**: beam_size=5, full processing, optimized for accuracy (<600ms)
- **Model Sharing**: Single model instance with efficient dual-mode access
- **Cache Management**: Smart caching to avoid re-processing identical segments
- **Memory Management**: Monitor VRAM usage with dual processing loads

### Advanced Connection Management
- **Concurrent Streams**: Scale partial/final processing based on GPU memory
- **Processing Loop Timing**: 50ms for partial, 100ms for final processing cycles
- **Utterance Management**: Efficient tracking and cleanup of utterance state
- **Buffer Optimization**: Separate partial and main buffers for optimal performance
- **Error Recovery**: Enhanced thresholds for partial vs. final processing failures

### Real-time Optimization
- **Target Latencies**: <300ms partial, <600ms final for conversational AI
- **Processing Priorities**: Partial results prioritized for immediate feedback
- **Resource Allocation**: Balance between partial responsiveness and final quality
- **Performance Monitoring**: Track both partial and final latency percentiles

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

- **Enhanced Real-time Processing**: Dual-mode with <300ms partial and <600ms final latency
- **Partial & Final Transcription**: Immediate feedback with high-quality complete results
- **Advanced Barge-in Support**: Utterance-aware interrupt detection with smooth conversation flow
- **Optimized Performance**: Overlapping audio windows with smart VAD triggering
- **Fasttalk Integration**: Purpose-built for complete STT→LLM→TTS conversational AI pipeline
- **Scalable Architecture**: Support for 50+ concurrent partial/final processing streams
- **Production Ready**: Comprehensive error handling, enhanced monitoring, and high availability
- **Developer Friendly**: Complete integration guide, performance testing, and client examples
- **Backward Compatible**: Legacy transcription message support for existing integrations

The service provides a robust, scalable foundation for real-time speech-to-text capabilities while maintaining seamless integration with the complete Fasttalk conversational AI ecosystem.

This comprehensive architecture documentation provides backend teams with all the information needed to successfully integrate with and deploy the STT service within the Fasttalk ecosystem, ensuring optimal performance for real-time conversational AI applications.