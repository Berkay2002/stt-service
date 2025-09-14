# app/client_examples/websocket_client_example.py

"""
Example WebSocket client for testing the STT service
Demonstrates how to connect, send audio, and receive transcriptions
"""

import asyncio
import json
import numpy as np
import websockets
import pyaudio
import time
from typing import Optional, Dict, Any


class STTWebSocketClient:
    """Example WebSocket client for the STT service"""

    def __init__(self, uri: str = "ws://localhost:8000/ws/transcribe"):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_id: Optional[str] = None

        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paFloat32

    async def connect(self) -> None:
        """Connect to the WebSocket STT server"""
        print(f"Connecting to {self.uri}...")

        try:
            self.websocket = await websockets.connect(self.uri)
            print("Connected to WebSocket STT server")

            # Start listening for messages
            asyncio.create_task(self._message_listener())

        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")

    async def start_session(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start a new transcription session"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        session_config = config or {
            "enable_timestamps": True,
            "enable_vad": True,
            "buffer_duration": 2.0
        }

        message = {
            "type": "start_session",
            "session_config": session_config
        }

        await self.websocket.send(json.dumps(message))
        print("Started transcription session")

    async def end_session(self) -> None:
        """End the current session"""
        if not self.websocket:
            return

        message = {"type": "end_session"}
        await self.websocket.send(json.dumps(message))
        print("Ended transcription session")

    async def send_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Send audio chunk to the server"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        # Convert to float32 bytes
        audio_bytes = audio_data.astype(np.float32).tobytes()
        await self.websocket.send(audio_bytes)

    async def flush_buffer(self) -> None:
        """Request immediate transcription of buffered audio"""
        if not self.websocket:
            return

        message = {"type": "flush_buffer"}
        await self.websocket.send(json.dumps(message))

    async def _message_listener(self) -> None:
        """Listen for incoming messages from the server"""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
        except Exception as e:
            print(f"Error in message listener: {e}")

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming messages from the server"""
        msg_type = message.get("type")

        if msg_type == "session_started":
            self.session_id = message.get("session_id")
            print(f"Session started: {self.session_id}")

        elif msg_type == "session_configured":
            config = message.get("config", {})
            print(f"Session configured: {config}")

        elif msg_type == "transcription":
            data = message.get("data", {})
            text = data.get("text", "")
            processing_time = data.get("processing_time_ms", 0)
            audio_duration = data.get("audio_duration_ms", 0)

            print(f"Transcription: '{text}'")
            print(f"  Processing time: {processing_time}ms")
            print(f"  Audio duration: {audio_duration}ms")

            # Print timestamps if available
            timestamps = data.get("timestamps")
            if timestamps:
                print("  Word timestamps:")
                for word_info in timestamps[:5]:  # Show first 5 words
                    word = word_info.get("word", "")
                    start = word_info.get("start_ms", 0)
                    end = word_info.get("end_ms", 0)
                    print(f"    {word}: {start}-{end}ms")

        elif msg_type == "session_ended":
            stats = message.get("stats", {})
            print(f"Session ended. Stats: {stats}")
            self.session_id = None

        elif msg_type == "error":
            error = message.get("error", {})
            error_code = error.get("code", "unknown")
            error_message = error.get("message", "Unknown error")
            recoverable = error.get("recoverable", False)

            print(f"Error [{error_code}]: {error_message}")
            if not recoverable:
                print("Error is not recoverable - disconnecting")
                await self.disconnect()

        elif msg_type == "heartbeat":
            # Server heartbeat - no action needed
            pass

        else:
            print(f"Unknown message type: {msg_type}")


async def record_and_transcribe_microphone():
    """Example: Record from microphone and transcribe in real-time"""
    client = STTWebSocketClient()

    try:
        await client.connect()
        await client.start_session({
            "enable_timestamps": True,
            "enable_vad": True,
            "buffer_duration": 1.5
        })

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open microphone stream
        stream = p.open(
            format=client.audio_format,
            channels=client.channels,
            rate=client.sample_rate,
            input=True,
            frames_per_buffer=client.chunk_size
        )

        print("Recording from microphone... Press Ctrl+C to stop")
        print("Speak now!")

        while True:
            # Read audio chunk
            audio_data = stream.read(client.chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            # Send to server
            await client.send_audio_chunk(audio_array)

            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping recording...")

    finally:
        # Cleanup
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()

        await client.end_session()
        await client.disconnect()


async def send_test_audio_file():
    """Example: Send a test audio file for transcription"""
    client = STTWebSocketClient()

    try:
        await client.connect()
        await client.start_session({
            "enable_timestamps": False,
            "enable_vad": False,
            "buffer_duration": 1.0
        })

        # Generate test audio (sine wave)
        duration = 3.0  # 3 seconds
        frequency = 440  # A4 note
        sample_rate = 16000

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        print(f"Sending {duration} seconds of test audio...")

        # Send audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await client.send_audio_chunk(chunk)
            await asyncio.sleep(0.1)  # 100ms delay between chunks

        # Wait a moment for processing
        await asyncio.sleep(2.0)

        # Flush any remaining audio
        await client.flush_buffer()
        await asyncio.sleep(1.0)

    finally:
        await client.end_session()
        await client.disconnect()


async def connection_stress_test():
    """Example: Test multiple concurrent connections"""
    clients = []
    num_clients = 5

    try:
        print(f"Creating {num_clients} concurrent connections...")

        # Create and connect clients
        for i in range(num_clients):
            client = STTWebSocketClient()
            await client.connect()
            await client.start_session({
                "enable_timestamps": False,
                "enable_vad": True,
                "buffer_duration": 2.0
            })
            clients.append(client)
            print(f"Client {i+1} connected")

        # Generate and send test audio from each client
        print("Sending test audio from all clients...")

        async def send_from_client(client_idx: int, client: STTWebSocketClient):
            # Generate unique test audio for each client
            duration = 2.0
            frequency = 440 + (client_idx * 100)  # Different frequency per client
            sample_rate = 16000

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

            # Send in chunks
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await client.send_audio_chunk(chunk)
                await asyncio.sleep(0.05)

        # Send audio from all clients concurrently
        tasks = [
            send_from_client(i, client)
            for i, client in enumerate(clients)
        ]
        await asyncio.gather(*tasks)

        # Wait for processing
        print("Waiting for transcriptions...")
        await asyncio.sleep(3.0)

    finally:
        # Clean up all clients
        for i, client in enumerate(clients):
            await client.end_session()
            await client.disconnect()
            print(f"Client {i+1} disconnected")


if __name__ == "__main__":
    print("WebSocket STT Client Examples")
    print("1. Record from microphone")
    print("2. Send test audio file")
    print("3. Connection stress test")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        asyncio.run(record_and_transcribe_microphone())
    elif choice == "2":
        asyncio.run(send_test_audio_file())
    elif choice == "3":
        asyncio.run(connection_stress_test())
    else:
        print("Invalid choice")