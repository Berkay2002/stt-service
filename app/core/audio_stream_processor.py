# app/core/audio_stream_processor.py

import asyncio
import numpy as np
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from collections import deque
import threading

from app.utils.logger import get_logger


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    data: np.ndarray
    timestamp: float
    sequence_id: int
    session_id: str
    sample_rate: int = 16000
    channels: int = 1

@dataclass
class ProcessingStats:
    """Statistics for audio processing performance"""
    total_chunks: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    buffer_overruns: int = 0
    buffer_underruns: int = 0
    average_latency: float = 0.0

class VoiceActivityDetector:
    """Simple voice activity detection based on energy and zero crossing rate"""

    def __init__(self, config: Dict[str, Any]):
        self.energy_threshold = config.get("energy_threshold", 0.01)
        self.zcr_threshold = config.get("zcr_threshold", 0.1)
        self.min_speech_duration = config.get("min_speech_duration", 0.1)  # seconds
        self.min_silence_duration = config.get("min_silence_duration", 0.3)  # seconds

        # State tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.current_state = "silence"  # "speech" or "silence"

        self.logger = get_logger('VAD')

    def detect_voice_activity(self, audio_chunk: AudioChunk) -> bool:
        """
        Detect voice activity in audio chunk

        Args:
            audio_chunk: AudioChunk containing audio data

        Returns:
            bool: True if speech detected, False if silence
        """
        audio_data = audio_chunk.data

        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_data ** 2))

        # Calculate zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))

        # Simple decision logic
        has_speech = energy > self.energy_threshold and zcr > self.zcr_threshold

        # Apply temporal smoothing
        frame_duration = len(audio_data) / audio_chunk.sample_rate
        min_speech_frames = int(self.min_speech_duration / frame_duration)
        min_silence_frames = int(self.min_silence_duration / frame_duration)

        if has_speech:
            self.speech_frames += 1
            self.silence_frames = 0

            if self.speech_frames >= min_speech_frames:
                self.current_state = "speech"

        else:
            self.silence_frames += 1
            self.speech_frames = 0

            if self.silence_frames >= min_silence_frames:
                self.current_state = "silence"

        return self.current_state == "speech"

class AudioStreamProcessor:
    """
    Processes streaming audio for real-time transcription
    Handles buffering, VAD, and audio preprocessing
    """

    def __init__(self, session_id: str, config: Dict[str, Any],
                 transcription_callback: Optional[Callable] = None):
        self.session_id = session_id
        self.config = config
        self.transcription_callback = transcription_callback

        # Audio configuration
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_duration = config.get("chunk_duration", 0.5)  # seconds
        self.buffer_duration = config.get("buffer_duration", 2.0)  # seconds

        # Processing configuration
        self.enable_vad = config.get("enable_vad", True)
        self.min_audio_length = config.get("min_audio_length", 0.5)
        self.max_silence_duration = config.get("max_silence_duration", 2.0)

        # Buffers and state
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.current_buffer_duration = 0.0
        self.max_buffer_size = int(self.buffer_duration * self.sample_rate)

        # Sequence tracking
        self.sequence_counter = 0
        self.last_transcription_time = time.time()

        # Voice activity detection
        if self.enable_vad:
            self.vad = VoiceActivityDetector(config.get("vad", {}))
        else:
            self.vad = None

        # Statistics
        self.stats = ProcessingStats()

        # Processing task
        self.processing_task: Optional[asyncio.Task] = None
        self.is_processing = False

        self.logger = get_logger('AudioStreamProcessor')
        self.logger.info(f"Initialized audio stream processor for session {session_id}")

    async def start_processing(self) -> None:
        """Start the background audio processing task"""
        if not self.processing_task or self.processing_task.done():
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._processing_loop())
            self.logger.info(f"Started audio processing for session {self.session_id}")

    async def stop_processing(self) -> None:
        """Stop the background audio processing task"""
        self.is_processing = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Stopped audio processing for session {self.session_id}")

    async def add_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        Add audio chunk to processing buffer

        Args:
            audio_data: Raw audio data as numpy array
        """
        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            session_id=self.session_id,
            sample_rate=self.sample_rate,
            channels=self.channels
        )

        self.sequence_counter += 1

        with self.buffer_lock:
            # Add to buffer
            self.audio_buffer.append(chunk)
            chunk_duration = len(audio_data) / self.sample_rate
            self.current_buffer_duration += chunk_duration

            # Update statistics
            self.stats.total_chunks += 1
            self.stats.total_audio_duration += chunk_duration

            # Trim buffer if it's too long
            while (self.current_buffer_duration > self.buffer_duration and
                   len(self.audio_buffer) > 1):
                removed_chunk = self.audio_buffer.popleft()
                removed_duration = len(removed_chunk.data) / self.sample_rate
                self.current_buffer_duration -= removed_duration
                self.stats.buffer_overruns += 1

    async def _processing_loop(self) -> None:
        """Main processing loop for audio stream"""
        while self.is_processing:
            try:
                await self._process_audio_buffer()
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exception=e)
                await asyncio.sleep(1.0)  # Longer delay on error

    async def _process_audio_buffer(self) -> None:
        """Process audio buffer for transcription"""
        with self.buffer_lock:
            if not self.audio_buffer:
                return

            # Check if we should process based on duration or silence
            should_process = False

            # Process if buffer is full
            if self.current_buffer_duration >= self.buffer_duration:
                should_process = True
                reason = "buffer_full"

            # Process if we haven't processed in a while and have minimum audio
            elif (self.current_buffer_duration >= self.min_audio_length and
                  time.time() - self.last_transcription_time > self.max_silence_duration):
                should_process = True
                reason = "timeout"

            # Process based on VAD if enabled
            elif self.enable_vad and self.vad and self.current_buffer_duration >= self.min_audio_length:
                # Check last few chunks for speech activity
                recent_chunks = list(self.audio_buffer)[-3:]  # Last 3 chunks
                speech_detected = False

                for chunk in recent_chunks:
                    if self.vad.detect_voice_activity(chunk):
                        speech_detected = True
                        break

                # If no recent speech detected and we have sufficient silence
                if not speech_detected and len(recent_chunks) >= 2:
                    should_process = True
                    reason = "vad_silence"

            if not should_process:
                return

            # Extract audio for processing
            audio_chunks = list(self.audio_buffer)
            self.audio_buffer.clear()
            self.current_buffer_duration = 0.0

        if not audio_chunks:
            return

        # Concatenate audio chunks
        audio_data = np.concatenate([chunk.data for chunk in audio_chunks])
        total_duration = len(audio_data) / self.sample_rate

        if total_duration < self.min_audio_length:
            self.logger.debug(f"Skipping transcription: audio too short ({total_duration:.2f}s)")
            return

        # Update processing statistics
        processing_start = time.time()
        self.last_transcription_time = processing_start

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_data)

        # Trigger transcription callback if provided
        if self.transcription_callback:
            try:
                await self.transcription_callback(
                    session_id=self.session_id,
                    audio_data=processed_audio,
                    metadata={
                        "duration": total_duration,
                        "chunks_count": len(audio_chunks),
                        "reason": reason,
                        "sequence_start": audio_chunks[0].sequence_id,
                        "sequence_end": audio_chunks[-1].sequence_id
                    }
                )

                # Update statistics
                processing_time = time.time() - processing_start
                self.stats.total_processing_time += processing_time
                self.stats.average_latency = (
                    self.stats.average_latency * (self.stats.total_chunks - len(audio_chunks)) +
                    processing_time * len(audio_chunks)
                ) / self.stats.total_chunks

            except Exception as e:
                self.logger.error(f"Error in transcription callback: {e}", exception=e)

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for transcription

        Args:
            audio_data: Raw audio data

        Returns:
            Preprocessed audio data
        """
        # Ensure audio is float32
        audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        # Apply gentle high-pass filter to remove DC bias
        if len(audio_data) > 1:
            audio_data = audio_data - np.mean(audio_data)

        # Optional: Apply gentle noise reduction
        # This is a simple spectral subtraction approach
        if self.config.get("noise_reduction", False):
            audio_data = self._apply_noise_reduction(audio_data)

        return audio_data

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply simple noise reduction to audio

        Args:
            audio_data: Input audio data

        Returns:
            Audio data with reduced noise
        """
        # Simple noise gate - suppress very quiet signals
        noise_floor = np.percentile(np.abs(audio_data), 10)  # 10th percentile as noise floor
        noise_threshold = noise_floor * 2  # Threshold above noise floor

        # Apply soft gating
        mask = np.abs(audio_data) > noise_threshold
        audio_data = audio_data * mask

        return audio_data

    async def flush_buffer(self) -> None:
        """Force processing of current buffer contents"""
        with self.buffer_lock:
            if self.audio_buffer and self.current_buffer_duration > 0:
                # Set a flag to force processing
                self.last_transcription_time = 0  # Force timeout condition

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        with self.buffer_lock:
            current_buffer_chunks = len(self.audio_buffer)
            current_buffer_duration = self.current_buffer_duration

        return {
            "session_id": self.session_id,
            "total_chunks": self.stats.total_chunks,
            "total_audio_duration": self.stats.total_audio_duration,
            "total_processing_time": self.stats.total_processing_time,
            "buffer_overruns": self.stats.buffer_overruns,
            "buffer_underruns": self.stats.buffer_underruns,
            "average_latency": self.stats.average_latency,
            "current_buffer_chunks": current_buffer_chunks,
            "current_buffer_duration": current_buffer_duration,
            "processing_active": self.is_processing,
            "real_time_factor": (
                self.stats.total_processing_time / max(self.stats.total_audio_duration, 0.001)
                if self.stats.total_audio_duration > 0 else 0.0
            )
        }

    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.stats = ProcessingStats()
        self.sequence_counter = 0
        self.logger.info(f"Reset statistics for session {self.session_id}")