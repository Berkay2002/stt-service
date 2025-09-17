# app/core/audio_stream_processor.py

import asyncio
import numpy as np
import time
import uuid
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
    partial_transcriptions: int = 0
    final_transcriptions: int = 0
    average_partial_latency: float = 0.0
    average_final_latency: float = 0.0

@dataclass
class UtteranceState:
    """Tracks the state of an ongoing utterance"""
    utterance_id: str
    start_time: float
    last_partial_text: str = ""
    accumulated_audio: List[AudioChunk] = None
    speech_detected: bool = False
    silence_duration: float = 0.0
    
    def __post_init__(self):
        if self.accumulated_audio is None:
            self.accumulated_audio = []

class VoiceActivityDetector:
    """Enhanced voice activity detection for partial/final transcription timing"""

    def __init__(self, config: Dict[str, Any]):
        self.energy_threshold = config.get("energy_threshold", 0.01)
        self.zcr_threshold = config.get("zcr_threshold", 0.1)
        self.min_speech_duration = config.get("min_speech_duration", 0.1)  # seconds
        self.min_silence_duration = config.get("min_silence_duration", 0.3)  # seconds
        
        # Enhanced thresholds for partial/final detection
        self.partial_trigger_duration = config.get("partial_trigger_duration", 0.25)  # 250ms
        self.final_trigger_silence = config.get("final_trigger_silence", 0.8)  # 800ms silence for final

        # State tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.current_state = "silence"  # "speech" or "silence"
        self.total_speech_duration = 0.0
        self.total_silence_duration = 0.0

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
            self.total_speech_duration += frame_duration
            self.total_silence_duration = 0.0  # Reset silence duration

            if self.speech_frames >= min_speech_frames:
                self.current_state = "speech"

        else:
            self.silence_frames += 1
            self.speech_frames = 0
            self.total_silence_duration += frame_duration

            if self.silence_frames >= min_silence_frames:
                self.current_state = "silence"

        return self.current_state == "speech"
    
    def should_trigger_partial(self) -> bool:
        """
        Determine if we should trigger a partial transcription
        
        Returns:
            bool: True if partial transcription should be triggered
        """
        return (self.current_state == "speech" and 
                self.total_speech_duration >= self.partial_trigger_duration)
    
    def should_trigger_final(self) -> bool:
        """
        Determine if we should trigger a final transcription
        
        Returns:
            bool: True if final transcription should be triggered
        """
        return (self.current_state == "silence" and 
                self.total_silence_duration >= self.final_trigger_silence)
    
    def reset_speech_timing(self):
        """Reset speech timing counters"""
        self.total_speech_duration = 0.0
        self.total_silence_duration = 0.0

class AudioStreamProcessor:
    """
    Enhanced audio stream processor with partial/final transcription support
    Handles buffering, VAD, and optimized audio preprocessing for real-time performance
    """

    def __init__(self, session_id: str, config: Dict[str, Any],
                 transcription_callback: Optional[Callable] = None,
                 partial_callback: Optional[Callable] = None):
        self.session_id = session_id
        self.config = config
        self.transcription_callback = transcription_callback
        self.partial_callback = partial_callback

        # Audio configuration - optimized for partial/final processing
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.partial_chunk_duration = config.get("partial_chunk_duration", 0.25)  # 250ms for partial
        self.final_chunk_duration = config.get("final_chunk_duration", 1.0)  # 1s for final
        self.overlap_duration = config.get("overlap_duration", 0.125)  # 125ms overlap
        self.buffer_duration = config.get("buffer_duration", 3.0)  # Extended buffer

        # Processing configuration
        self.enable_vad = config.get("enable_vad", True)
        self.enable_partial_transcription = config.get("enable_partial_transcription", True)
        self.min_audio_length = config.get("min_audio_length", 0.3)  # Reduced for faster response
        self.max_silence_duration = config.get("max_silence_duration", 2.0)

        # Enhanced buffers and state
        self.audio_buffer = deque()  # Main audio buffer
        self.partial_buffer = deque()  # Buffer for partial processing
        self.overlap_buffer = np.array([])  # Overlap buffer for smooth processing
        self.buffer_lock = threading.Lock()
        self.current_buffer_duration = 0.0
        self.max_buffer_size = int(self.buffer_duration * self.sample_rate)

        # Utterance tracking
        self.current_utterance: Optional[UtteranceState] = None
        self.last_partial_time = 0.0
        self.last_final_time = 0.0

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

        # Processing tasks
        self.processing_task: Optional[asyncio.Task] = None
        self.partial_task: Optional[asyncio.Task] = None
        self.is_processing = False

        self.logger = get_logger('AudioStreamProcessor')
        self.logger.info(f"Initialized enhanced audio stream processor for session {session_id}")
        self.logger.info(f"Partial processing: {self.enable_partial_transcription}, VAD: {self.enable_vad}")

    async def start_processing(self) -> None:
        """Start the background audio processing tasks"""
        if not self.processing_task or self.processing_task.done():
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._processing_loop())
            self.logger.info(f"Started main audio processing for session {self.session_id}")
            
            # Start partial processing task if enabled
            if self.enable_partial_transcription and (not self.partial_task or self.partial_task.done()):
                self.partial_task = asyncio.create_task(self._partial_processing_loop())
                self.logger.info(f"Started partial processing for session {self.session_id}")

    async def stop_processing(self) -> None:
        """Stop the background audio processing tasks"""
        self.is_processing = False
        
        # Stop main processing task
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Stop partial processing task
        if self.partial_task and not self.partial_task.done():
            self.partial_task.cancel()
            try:
                await self.partial_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Stopped audio processing for session {self.session_id}")

    async def add_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        Add audio chunk to processing buffers with overlap handling

        Args:
            audio_data: Raw audio data as numpy array
        """
        # Combine with overlap buffer for smoother processing
        if len(self.overlap_buffer) > 0:
            combined_audio = np.concatenate([self.overlap_buffer, audio_data])
        else:
            combined_audio = audio_data
            
        # Update overlap buffer (keep last 125ms for next chunk)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        if len(audio_data) >= overlap_samples:
            self.overlap_buffer = audio_data[-overlap_samples:]
        else:
            self.overlap_buffer = audio_data

        chunk = AudioChunk(
            data=combined_audio,
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            session_id=self.session_id,
            sample_rate=self.sample_rate,
            channels=self.channels
        )

        self.sequence_counter += 1

        with self.buffer_lock:
            # Add to main buffer
            self.audio_buffer.append(chunk)
            
            # Add to partial buffer for quick processing if enabled
            if self.enable_partial_transcription:
                self.partial_buffer.append(chunk)
                
            chunk_duration = len(combined_audio) / self.sample_rate
            self.current_buffer_duration += chunk_duration

            # Update statistics
            self.stats.total_chunks += 1
            self.stats.total_audio_duration += chunk_duration

            # Trim main buffer if it's too long
            while (self.current_buffer_duration > self.buffer_duration and
                   len(self.audio_buffer) > 1):
                removed_chunk = self.audio_buffer.popleft()
                removed_duration = len(removed_chunk.data) / self.sample_rate
                self.current_buffer_duration -= removed_duration
                self.stats.buffer_overruns += 1
                
            # Trim partial buffer (keep it smaller)
            partial_max_duration = self.partial_chunk_duration * 3  # Keep 3x partial duration
            current_partial_duration = sum(len(c.data) / self.sample_rate for c in self.partial_buffer)
            while (current_partial_duration > partial_max_duration and
                   len(self.partial_buffer) > 1):
                removed_chunk = self.partial_buffer.popleft()
                current_partial_duration -= len(removed_chunk.data) / self.sample_rate

        # Update VAD if enabled
        if self.vad:
            is_speech = self.vad.detect_voice_activity(chunk)
            
            # Handle utterance state
            if is_speech and not self.current_utterance:
                # Start new utterance
                self.current_utterance = UtteranceState(
                    utterance_id=str(uuid.uuid4()),
                    start_time=chunk.timestamp
                )
                self.logger.debug(f"Started new utterance: {self.current_utterance.utterance_id}")
            elif self.current_utterance:
                self.current_utterance.speech_detected = is_speech

    async def _processing_loop(self) -> None:
        """Main processing loop for final transcriptions"""
        while self.is_processing:
            try:
                await self._process_final_transcription()
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop

            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}", exception=e)
                await asyncio.sleep(1.0)  # Longer delay on error

    async def _partial_processing_loop(self) -> None:
        """Dedicated loop for partial transcriptions"""
        while self.is_processing:
            try:
                await self._process_partial_transcription()
                await asyncio.sleep(0.05)  # Faster loop for partial results

            except Exception as e:
                self.logger.error(f"Error in partial processing loop: {e}", exception=e)
                await asyncio.sleep(0.5)  # Shorter delay on error

    async def _process_partial_transcription(self) -> None:
        """Process partial buffer for quick transcription feedback"""
        if not self.enable_partial_transcription or not self.partial_callback:
            return
            
        with self.buffer_lock:
            if not self.partial_buffer or not self.current_utterance:
                return
                
            # Check if we should trigger partial transcription
            should_process_partial = False
            
            # Check VAD trigger for partial results
            if self.vad and self.vad.should_trigger_partial():
                should_process_partial = True
                reason = "speech_detected"
                
            # Time-based trigger for partial results
            elif time.time() - self.last_partial_time >= self.partial_chunk_duration:
                current_partial_duration = sum(len(c.data) / self.sample_rate for c in self.partial_buffer)
                if current_partial_duration >= self.min_audio_length:
                    should_process_partial = True
                    reason = "time_trigger"
                    
            if not should_process_partial:
                return
                
            # Extract recent audio for partial processing
            partial_chunks = list(self.partial_buffer)[-2:]  # Use last 2 chunks for partial
            
        if not partial_chunks:
            return
            
        # Concatenate partial audio
        partial_audio = np.concatenate([chunk.data for chunk in partial_chunks])
        total_duration = len(partial_audio) / self.sample_rate
        
        if total_duration < 0.2:  # Skip very short partial audio
            return
            
        processing_start = time.time()
        self.last_partial_time = processing_start
        
        # Preprocess audio
        processed_audio = self._preprocess_audio(partial_audio)
        
        # Trigger partial callback
        try:
            await self.partial_callback(
                session_id=self.session_id,
                audio_data=processed_audio,
                metadata={
                    "duration": total_duration,
                    "is_partial": True,
                    "utterance_id": self.current_utterance.utterance_id,
                    "reason": reason,
                    "sequence_start": partial_chunks[0].sequence_id,
                    "sequence_end": partial_chunks[-1].sequence_id
                }
            )
            
            # Update statistics
            processing_time = time.time() - processing_start
            self.stats.partial_transcriptions += 1
            self.stats.average_partial_latency = (
                (self.stats.average_partial_latency * (self.stats.partial_transcriptions - 1) + processing_time) 
                / self.stats.partial_transcriptions
            )
            
        except Exception as e:
            self.logger.error(f"Error in partial transcription callback: {e}", exception=e)

    async def _process_final_transcription(self) -> None:
        """Process main buffer for final transcription results"""
        with self.buffer_lock:
            if not self.audio_buffer:
                return

            # Check if we should process final transcription
            should_process = False

            # Process if buffer is full
            if self.current_buffer_duration >= self.final_chunk_duration:
                should_process = True
                reason = "buffer_full"

            # Process based on VAD silence detection for final results
            elif self.vad and self.vad.should_trigger_final() and self.current_utterance:
                should_process = True
                reason = "silence_detected"

            # Fallback timeout processing
            elif (self.current_buffer_duration >= self.min_audio_length and
                  time.time() - self.last_final_time > self.max_silence_duration):
                should_process = True
                reason = "timeout"

            if not should_process:
                return

            # Extract audio for final processing
            audio_chunks = list(self.audio_buffer)
            
            # For final processing, keep some buffer to avoid cutting speech
            if reason != "buffer_full" and len(audio_chunks) > 2:
                # Keep last chunk in buffer for continuity
                audio_chunks = audio_chunks[:-1]
                kept_chunk = self.audio_buffer[-1]
                self.audio_buffer.clear()
                self.audio_buffer.append(kept_chunk)
                self.current_buffer_duration = len(kept_chunk.data) / self.sample_rate
            else:
                self.audio_buffer.clear()
                self.current_buffer_duration = 0.0

        if not audio_chunks:
            return

        # Concatenate audio chunks
        audio_data = np.concatenate([chunk.data for chunk in audio_chunks])
        total_duration = len(audio_data) / self.sample_rate

        if total_duration < self.min_audio_length:
            self.logger.debug(f"Skipping final transcription: audio too short ({total_duration:.2f}s)")
            return

        # Update processing statistics
        processing_start = time.time()
        self.last_final_time = processing_start

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_data)

        # Trigger final transcription callback
        if self.transcription_callback:
            try:
                await self.transcription_callback(
                    session_id=self.session_id,
                    audio_data=processed_audio,
                    metadata={
                        "duration": total_duration,
                        "is_partial": False,
                        "utterance_id": self.current_utterance.utterance_id if self.current_utterance else None,
                        "chunks_count": len(audio_chunks),
                        "reason": reason,
                        "sequence_start": audio_chunks[0].sequence_id,
                        "sequence_end": audio_chunks[-1].sequence_id
                    }
                )

                # Update statistics
                processing_time = time.time() - processing_start
                self.stats.final_transcriptions += 1
                self.stats.total_processing_time += processing_time
                self.stats.average_final_latency = (
                    (self.stats.average_final_latency * (self.stats.final_transcriptions - 1) + processing_time) 
                    / self.stats.final_transcriptions
                )

                # End current utterance if this was triggered by silence
                if reason == "silence_detected" and self.current_utterance:
                    self.logger.debug(f"Ended utterance: {self.current_utterance.utterance_id}")
                    self.current_utterance = None
                    if self.vad:
                        self.vad.reset_speech_timing()

            except Exception as e:
                self.logger.error(f"Error in final transcription callback: {e}", exception=e)

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
        """Get enhanced processing statistics including partial/final metrics"""
        with self.buffer_lock:
            current_buffer_chunks = len(self.audio_buffer)
            current_buffer_duration = self.current_buffer_duration
            current_partial_chunks = len(self.partial_buffer) if self.enable_partial_transcription else 0

        return {
            "session_id": self.session_id,
            "total_chunks": self.stats.total_chunks,
            "total_audio_duration": self.stats.total_audio_duration,
            "total_processing_time": self.stats.total_processing_time,
            "buffer_overruns": self.stats.buffer_overruns,
            "buffer_underruns": self.stats.buffer_underruns,
            "average_latency": self.stats.average_latency,
            
            # Enhanced stats for partial/final processing
            "partial_transcriptions": self.stats.partial_transcriptions,
            "final_transcriptions": self.stats.final_transcriptions,
            "average_partial_latency": self.stats.average_partial_latency,
            "average_final_latency": self.stats.average_final_latency,
            
            # Current buffer status
            "current_buffer_chunks": current_buffer_chunks,
            "current_buffer_duration": current_buffer_duration,
            "current_partial_chunks": current_partial_chunks,
            
            # Processing status
            "processing_active": self.is_processing,
            "partial_processing_enabled": self.enable_partial_transcription,
            "current_utterance_id": self.current_utterance.utterance_id if self.current_utterance else None,
            
            # Performance metrics
            "real_time_factor": (
                self.stats.total_processing_time / max(self.stats.total_audio_duration, 0.001)
                if self.stats.total_audio_duration > 0 else 0.0
            ),
            "partial_response_time": self.stats.average_partial_latency * 1000,  # ms
            "final_response_time": self.stats.average_final_latency * 1000,  # ms
        }

    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.stats = ProcessingStats()
        self.sequence_counter = 0
        self.logger.info(f"Reset statistics for session {self.session_id}")