# test_partial_final_performance.py

"""
Performance test script for partial/final STT implementation
Tests latency improvements and validates the new functionality
"""

import asyncio
import json
import time
import numpy as np
import websockets
from typing import List, Dict, Any
import statistics


class PerformanceTestResults:
    """Container for performance test results"""
    
    def __init__(self):
        self.partial_latencies: List[float] = []
        self.final_latencies: List[float] = []
        self.total_test_time = 0.0
        self.messages_received = 0
        self.partial_messages = 0
        self.final_messages = 0
        self.legacy_messages = 0
        self.errors = 0

    def add_partial_latency(self, latency: float):
        self.partial_latencies.append(latency)
        self.partial_messages += 1

    def add_final_latency(self, latency: float):
        self.final_latencies.append(latency)
        self.final_messages += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_messages": self.messages_received,
            "partial_messages": self.partial_messages,
            "final_messages": self.final_messages,
            "legacy_messages": self.legacy_messages,
            "errors": self.errors,
            "avg_partial_latency_ms": statistics.mean(self.partial_latencies) if self.partial_latencies else 0,
            "avg_final_latency_ms": statistics.mean(self.final_latencies) if self.final_latencies else 0,
            "median_partial_latency_ms": statistics.median(self.partial_latencies) if self.partial_latencies else 0,
            "median_final_latency_ms": statistics.median(self.final_latencies) if self.final_latencies else 0,
            "p95_partial_latency_ms": np.percentile(self.partial_latencies, 95) if self.partial_latencies else 0,
            "p95_final_latency_ms": np.percentile(self.final_latencies, 95) if self.final_latencies else 0,
            "test_duration_s": self.total_test_time
        }


class STTPerformanceTester:
    """WebSocket STT Performance Tester"""

    def __init__(self, uri: str = "ws://localhost:8000/ws/transcribe"):
        self.uri = uri
        self.websocket = None
        self.session_id = None
        self.results = PerformanceTestResults()
        self.audio_send_times = {}  # Track when each audio chunk was sent

    async def connect(self) -> None:
        """Connect to the WebSocket STT server"""
        print(f"Connecting to {self.uri}...")
        try:
            self.websocket = await websockets.connect(self.uri)
            print("Connected to WebSocket STT server")
            asyncio.create_task(self._message_listener())
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

    async def start_session(self) -> None:
        """Start a transcription session with partial/final enabled"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        config = {
            "enable_timestamps": False,  # Disable for faster processing
            "enable_vad": True,
            "enable_partial_transcription": True,
            "buffer_duration": 2.0,
            "partial_chunk_duration": 0.25,
            "final_chunk_duration": 1.0
        }

        message = {
            "type": "start_session",
            "session_config": config
        }

        await self.websocket.send(json.dumps(message))
        print("Started enhanced transcription session")

    async def run_performance_test(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive performance test
        
        Args:
            duration_seconds: How long to run the test
            
        Returns:
            Performance test results
        """
        print(f"Starting {duration_seconds}s performance test...")
        
        await self.start_session()
        
        # Generate and send test audio for the specified duration
        start_time = time.time()
        chunk_counter = 0
        
        while time.time() - start_time < duration_seconds:
            # Generate 250ms of test audio (16kHz, mono)
            chunk_duration = 0.25
            sample_rate = 16000
            samples = int(chunk_duration * sample_rate)
            
            # Generate speech-like audio (mixed frequencies)
            t = np.linspace(0, chunk_duration, samples, False)
            audio_data = (
                0.3 * np.sin(2 * np.pi * 300 * t) +  # Fundamental frequency
                0.2 * np.sin(2 * np.pi * 600 * t) +  # First harmonic
                0.1 * np.sin(2 * np.pi * 900 * t) +  # Second harmonic
                0.05 * np.random.normal(0, 1, samples)  # Add some noise
            ).astype(np.float32)
            
            # Normalize
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Send audio chunk
            chunk_id = f"chunk_{chunk_counter}"
            self.audio_send_times[chunk_id] = time.time()
            
            await self.websocket.send(audio_data.tobytes())
            chunk_counter += 1
            
            # Wait for next chunk (simulate real-time audio)
            await asyncio.sleep(chunk_duration)
        
        # Wait for final processing
        print("Waiting for final processing...")
        await asyncio.sleep(3.0)
        
        # Flush any remaining audio
        await self.websocket.send(json.dumps({"type": "flush_buffer"}))
        await asyncio.sleep(2.0)
        
        self.results.total_test_time = time.time() - start_time
        
        return self.results.get_summary()

    async def _message_listener(self) -> None:
        """Listen for incoming messages and track performance"""
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
            self.results.errors += 1

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming messages and calculate latencies"""
        msg_type = message.get("type")
        current_time = time.time()
        
        self.results.messages_received += 1

        if msg_type == "session_started":
            self.session_id = message.get("session_id")
            print(f"Session started: {self.session_id}")

        elif msg_type == "transcription_partial":
            data = message.get("data", {})
            processing_time = data.get("processing_time_ms", 0)
            text = data.get("text", "")
            
            self.results.add_partial_latency(processing_time)
            print(f"[PARTIAL] '{text[:30]}...' ({processing_time}ms)")

        elif msg_type == "transcription_final":
            data = message.get("data", {})
            processing_time = data.get("processing_time_ms", 0)
            text = data.get("text", "")
            
            self.results.add_final_latency(processing_time)
            print(f"[FINAL] '{text[:50]}...' ({processing_time}ms)")

        elif msg_type == "transcription":
            # Legacy format
            data = message.get("data", {})
            processing_time = data.get("processing_time_ms", 0)
            self.results.legacy_messages += 1
            print(f"[LEGACY] Processing time: {processing_time}ms")

        elif msg_type == "error":
            error = message.get("error", {})
            print(f"Error: {error}")
            self.results.errors += 1

    async def disconnect(self) -> None:
        """End session and disconnect"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "end_session"}))
                await self.websocket.close()
                print("Disconnected from server")
            except Exception as e:
                print(f"Error during disconnect: {e}")


async def run_performance_tests():
    """Run comprehensive performance tests"""
    
    print("=" * 70)
    print("STT PARTIAL/FINAL PERFORMANCE TESTS")
    print("=" * 70)
    
    tester = STTPerformanceTester()
    
    try:
        await tester.connect()
        
        # Run 30-second performance test
        results = await tester.run_performance_test(duration_seconds=30)
        
        print("\n" + "=" * 70)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 70)
        
        print(f"Test Duration: {results['test_duration_s']:.1f}s")
        print(f"Total Messages: {results['total_messages']}")
        print(f"Partial Messages: {results['partial_messages']}")
        print(f"Final Messages: {results['final_messages']}")
        print(f"Legacy Messages: {results['legacy_messages']}")
        print(f"Errors: {results['errors']}")
        
        print("\nLatency Metrics:")
        print(f"Average Partial Latency: {results['avg_partial_latency_ms']:.1f}ms")
        print(f"Average Final Latency: {results['avg_final_latency_ms']:.1f}ms")
        print(f"Median Partial Latency: {results['median_partial_latency_ms']:.1f}ms")
        print(f"Median Final Latency: {results['median_final_latency_ms']:.1f}ms")
        print(f"95th Percentile Partial: {results['p95_partial_latency_ms']:.1f}ms")
        print(f"95th Percentile Final: {results['p95_final_latency_ms']:.1f}ms")
        
        print("\nPerformance Assessment:")
        if results['avg_partial_latency_ms'] < 300:
            print("✅ EXCELLENT: Partial latency < 300ms")
        elif results['avg_partial_latency_ms'] < 500:
            print("✅ GOOD: Partial latency < 500ms")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Partial latency > 500ms")
            
        if results['avg_final_latency_ms'] < 600:
            print("✅ EXCELLENT: Final latency < 600ms")
        elif results['avg_final_latency_ms'] < 800:
            print("✅ GOOD: Final latency < 800ms")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Final latency > 800ms")
            
        if results['partial_messages'] > 0:
            print("✅ CONFIRMED: Partial transcription working")
        else:
            print("❌ ISSUE: No partial transcriptions received")
            
        if results['final_messages'] > 0:
            print("✅ CONFIRMED: Final transcription working")
        else:
            print("❌ ISSUE: No final transcriptions received")
        
    except Exception as e:
        print(f"Test failed: {e}")
        
    finally:
        await tester.disconnect()


if __name__ == "__main__":
    print("Starting STT Performance Tests...")
    print("Make sure the STT WebSocket server is running on localhost:8000")
    print("")
    
    asyncio.run(run_performance_tests())