# app/core/websocket_server.py

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Literal, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.whisper_handler import WhisperHandler
from app.core.audio_processor import AudioProcessor
from app.core.audio_stream_processor import AudioStreamProcessor
from app.core.error_handler import WebSocketErrorHandler, create_error_response, ErrorCategory
from app.monitoring.service_monitor import ServiceMonitor
from app.utils.config import load_config
from app.utils.logger import get_logger


# Type definitions for better code clarity
ConnectionState = Literal["connecting", "active", "buffering", "processing", "disconnecting", "error"]
MessageType = Literal["audio_chunk", "start_session", "end_session", "flush_buffer", "configure"]

@dataclass
class SessionConfig:
    """Configuration for a transcription session"""
    language: Optional[str] = None
    enable_timestamps: bool = False
    enable_vad: bool = True
    buffer_duration: float = 2.0
    chunk_size: int = 1024  # Audio chunk size in samples

@dataclass
class ConnectionInfo:
    """Information about an active WebSocket connection"""
    websocket: WebSocket
    session_id: str
    state: ConnectionState
    config: SessionConfig
    stream_processor: Optional[AudioStreamProcessor]
    last_activity: float
    stats: Dict[str, Any]

@dataclass
class TranscriptionResult:
    """Result of a transcription operation"""
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    processing_time_ms: float = 0
    audio_duration_ms: float = 0
    sequence_id: Optional[int] = None
    chunk_id: Optional[str] = None
    timestamps: Optional[List[Dict[str, Any]]] = None


class ConnectionManager:
    """Manages WebSocket connections and coordinates with Whisper processing"""

    def __init__(self, whisper_handler: WhisperHandler, audio_processor: AudioProcessor,
                 monitor: ServiceMonitor, config: Dict[str, Any], max_connections: int = 50):
        self.whisper_handler = whisper_handler
        self.audio_processor = audio_processor
        self.monitor = monitor
        self.config = config
        self.max_connections = max_connections

        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.processing_lock = asyncio.Semaphore(1)  # Serialize GPU access

        # Error handling
        self.error_handler = WebSocketErrorHandler(config)

        # Statistics
        self.total_connections = 0
        self.active_transcriptions = 0

        # Setup logging
        self.logger = get_logger('WebSocket_Manager')

    async def connect(self, websocket: WebSocket) -> str:
        """Handle new WebSocket connection"""
        # Check if connections should be rejected due to error conditions
        if await self.error_handler.should_reject_connection():
            await websocket.close(code=1013, reason="Service temporarily unavailable")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")

        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Maximum connections exceeded")
            raise HTTPException(status_code=503, detail="Server at capacity")

        session_id = str(uuid.uuid4())
        await websocket.accept()

        # Initialize connection info
        connection_info = ConnectionInfo(
            websocket=websocket,
            session_id=session_id,
            state="active",
            config=SessionConfig(),
            stream_processor=None,
            last_activity=time.time(),
            stats={
                "chunks_received": 0,
                "transcriptions_completed": 0,
                "total_audio_duration": 0.0,
                "total_processing_time": 0.0,
                "errors": 0
            }
        )

        self.connections[session_id] = connection_info
        self.total_connections += 1

        # Send session started message
        await self._send_message(websocket, {
            "type": "session_started",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        self.logger.info(f"New WebSocket connection: {session_id} (total: {len(self.connections)})")
        return session_id

    async def disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection"""
        if session_id in self.connections:
            connection = self.connections[session_id]
            connection.state = "disconnecting"

            # Stop stream processor if active
            if connection.stream_processor:
                await connection.stream_processor.stop_processing()

            # Send final statistics
            try:
                await self._send_message(connection.websocket, {
                    "type": "session_ended",
                    "session_id": session_id,
                    "stats": connection.stats,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception:
                pass  # Connection might already be closed

            # Clean up
            del self.connections[session_id]
            self.logger.info(f"WebSocket disconnected: {session_id} (remaining: {len(self.connections)})")

    async def handle_message(self, session_id: str, message: Union[str, bytes]) -> None:
        """Handle incoming WebSocket message"""
        if session_id not in self.connections:
            return

        connection = self.connections[session_id]
        connection.last_activity = time.time()

        try:
            if isinstance(message, bytes):
                # Binary audio data
                await self._handle_audio_chunk(session_id, message)
            else:
                # JSON control message
                await self._handle_control_message(session_id, json.loads(message))

        except Exception as e:
            connection.stats["errors"] += 1
            error_info = await self.error_handler.handle_error(e, "message_processing", session_id)
            error_response = create_error_response(error_info)
            await self._send_message(connection.websocket, error_response)

    async def _handle_audio_chunk(self, session_id: str, audio_data: bytes) -> None:
        """Process incoming audio chunk"""
        connection = self.connections[session_id]
        connection.stats["chunks_received"] += 1

        try:
            # Convert bytes to numpy array (assuming float32 PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            # Initialize stream processor if not already created
            if not connection.stream_processor:
                # Create transcription callback
                async def transcription_callback(session_id: str, audio_data: np.ndarray, metadata: Dict[str, Any]):
                    await self._process_transcription(session_id, audio_data, metadata)

                # Initialize stream processor with configuration
                stream_config = {
                    "sample_rate": 16000,
                    "channels": 1,
                    "chunk_duration": connection.config.buffer_duration,
                    "buffer_duration": connection.config.buffer_duration * 2,
                    "enable_vad": connection.config.enable_vad,
                    "min_audio_length": 0.5,
                    "max_silence_duration": 2.0
                }

                connection.stream_processor = AudioStreamProcessor(
                    session_id=session_id,
                    config=stream_config,
                    transcription_callback=transcription_callback
                )
                await connection.stream_processor.start_processing()

            # Add audio chunk to stream processor
            await connection.stream_processor.add_audio_chunk(audio_array)

            chunk_duration = len(audio_array) / 16000  # Assuming 16kHz
            connection.stats["total_audio_duration"] += chunk_duration

        except Exception as e:
            connection.stats["errors"] += 1
            error_info = await self.error_handler.handle_error(e, "audio_processing", session_id)
            error_response = create_error_response(error_info)
            await self._send_message(connection.websocket, error_response)

    async def _handle_control_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle control messages (JSON)"""
        connection = self.connections[session_id]
        msg_type = message.get("type")

        if msg_type == "start_session":
            # Update session configuration
            if "session_config" in message:
                config_data = message["session_config"]
                connection.config = SessionConfig(**config_data)

            await self._send_message(connection.websocket, {
                "type": "session_configured",
                "config": asdict(connection.config)
            })

        elif msg_type == "end_session":
            await self.disconnect(session_id)

        elif msg_type == "flush_buffer":
            # Force transcription of current buffer
            if connection.stream_processor:
                await connection.stream_processor.flush_buffer()

        elif msg_type == "configure":
            # Update configuration
            config_updates = message.get("config", {})
            for key, value in config_updates.items():
                if hasattr(connection.config, key):
                    setattr(connection.config, key, value)

        else:
            error_info = await self.error_handler.handle_error(
                ValueError(f"Unknown message type: {msg_type}"),
                "control_message",
                session_id
            )
            error_response = create_error_response(error_info)
            await self._send_message(connection.websocket, error_response)

    async def _process_transcription(self, session_id: str, audio_data: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Process transcription request from audio stream processor"""
        if session_id not in self.connections:
            return

        connection = self.connections[session_id]
        connection.state = "processing"
        start_time = time.time()

        try:
            async with self.processing_lock:
                # Perform transcription
                if connection.config.enable_timestamps:
                    segments = await asyncio.get_event_loop().run_in_executor(
                        None, self.whisper_handler.transcribe_with_timestamps, audio_data
                    )

                    # Extract text and prepare timestamps
                    text = " ".join(segment["text"] for segment in segments)
                    timestamps = [
                        {
                            "start_ms": int(word["start"] * 1000),
                            "end_ms": int(word["end"] * 1000),
                            "word": word["word"],
                            "confidence": word.get("probability")
                        }
                        for segment in segments
                        for word in segment.get("words", [])
                    ]
                else:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, self.whisper_handler.transcribe, audio_data
                    )
                    timestamps = None

                processing_time = time.time() - start_time
                audio_duration = metadata.get("duration", len(audio_data) / 16000)

                # Update statistics
                connection.stats["transcriptions_completed"] += 1
                connection.stats["total_processing_time"] += processing_time

                # Record in monitor
                self.monitor.record_transcription(audio_duration, processing_time)

                # Send result
                result = {
                    "type": "transcription",
                    "data": {
                        "text": text.strip(),
                        "processing_time_ms": int(processing_time * 1000),
                        "audio_duration_ms": int(audio_duration * 1000),
                        "timestamps": timestamps,
                        "metadata": metadata
                    },
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

                await self._send_message(connection.websocket, result)

        except Exception as e:
            connection.stats["errors"] += 1
            error_info = await self.error_handler.handle_error(e, "transcription", session_id)
            error_response = create_error_response(error_info)
            await self._send_message(connection.websocket, error_response)
            self.monitor.record_error(e, f"transcription_{session_id}")

        finally:
            # Update connection state
            if session_id in self.connections:
                self.connections[session_id].state = "active"

    async def _send_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send JSON message to WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        processing_stats = {}
        for session_id, connection in self.connections.items():
            if connection.stream_processor:
                processing_stats[session_id] = connection.stream_processor.get_processing_stats()

        return {
            "active_connections": len(self.connections),
            "total_connections": self.total_connections,
            "max_connections": self.max_connections,
            "connections_by_state": {
                state: sum(1 for conn in self.connections.values() if conn.state == state)
                for state in ["active", "processing", "buffering", "error"]
            },
            "processing_stats": processing_stats,
            "error_statistics": self.error_handler.get_error_statistics()
        }

class WebSocketSTTServer:
    """Main WebSocket STT Server using FastAPI"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.app = FastAPI(title="WebSocket STT Service", version="1.0.0")

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self.whisper_handler = WhisperHandler(self.config)
        self.audio_processor = AudioProcessor(self.config)
        self.monitor = ServiceMonitor(self.config)
        self.connection_manager = ConnectionManager(
            self.whisper_handler, self.audio_processor, self.monitor, self.config
        )

        self.logger = get_logger('WebSocket_Server')
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""

        @self.app.websocket("/ws/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for transcription"""
            session_id = None
            try:
                session_id = await self.connection_manager.connect(websocket)

                while True:
                    # Receive message (can be text or binary)
                    try:
                        message = await websocket.receive()

                        if "text" in message:
                            await self.connection_manager.handle_message(session_id, message["text"])
                        elif "bytes" in message:
                            await self.connection_manager.handle_message(session_id, message["bytes"])

                    except WebSocketDisconnect:
                        break

            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")

            finally:
                if session_id:
                    await self.connection_manager.disconnect(session_id)

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = self.monitor.health_check()
            return health_status

        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics"""
            return {
                "service": self.monitor.get_metrics(),
                "connections": self.connection_manager.get_connection_stats(),
                "whisper": self.whisper_handler.get_model_info()
            }

        @self.app.get("/")
        async def root():
            """Root endpoint with service information"""
            return {
                "service": "WebSocket STT Service",
                "version": "1.0.0",
                "endpoints": {
                    "websocket": "/ws/transcribe",
                    "health": "/health",
                    "stats": "/stats"
                },
                "status": "ready" if self.monitor.model_loaded else "loading"
            }

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the WebSocket server"""
        self.logger.info(f"Starting WebSocket STT Server on {host}:{port}")

        # Start monitoring server in background
        from app.monitoring.service_monitor import MonitoringServer
        monitoring_server = MonitoringServer(self.monitor, port=9091)
        monitoring_server.run_in_thread()

        # Start main server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()


# Entry point for running the server
async def main():
    """Main entry point"""
    config = load_config()
    server = WebSocketSTTServer(config)
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())