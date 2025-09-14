# app/core/realtime_transcription.py

import threading
import queue
import time
import numpy as np
import pyaudio
import logging
import signal
import sys
from typing import Optional, Tuple
from app.utils.config import load_config
from app.core.whisper_handler import WhisperHandler
from app.core.audio_processor import AudioProcessor

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    print("Warning: webrtcvad not available. Install with 'pip install webrtcvad' for better voice activity detection.")


def find_microphone_by_name(name_pattern: str) -> Optional[int]:
    """
    Find a microphone device by name pattern.
    
    Args:
        name_pattern (str): Pattern to search for in microphone names (case-insensitive)
        
    Returns:
        Optional[int]: Device ID if found, None otherwise
    """
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if (device_info['maxInputChannels'] > 0 and 
                    name_pattern.lower() in device_info['name'].lower()):
                    return i
            except Exception:
                continue
    finally:
        p.terminate()
    return None


def list_microphones() -> list:
    """
    List all available microphone devices.
    
    Returns:
        list: List of tuples (device_id, device_name, channels)
    """
    p = pyaudio.PyAudio()
    microphones = []
    try:
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    microphones.append((i, device_info['name'], device_info['maxInputChannels']))
            except Exception:
                continue
    finally:
        p.terminate()
    return microphones

class RealTimeTranscriber:
    """
    Real-time speech-to-text transcription system using threading and audio buffering.
    Continuously captures audio from microphone and transcribes it using Whisper.
    """
    
    def __init__(self, config: dict, chunk_duration: float = 1.0, buffer_duration: float = 30.0, microphone_device_id: Optional[int] = None):
        """
        Initialize the real-time transcriber.

        Args:
            config (dict): Configuration for Whisper and audio processing (includes realtime, microphone, and VAD settings)
            chunk_duration (float): Audio chunk duration in seconds (fallback if not in config)
            buffer_duration (float): Maximum audio buffer duration in seconds (fallback if not in config)
            microphone_device_id (Optional[int]): Specific microphone device ID to use (None for default)

        Note:
            Most settings are now loaded from config['realtime'], config['microphone'], and config['vad_realtime'].
            Parameters chunk_duration and buffer_duration are only used as fallbacks if not specified in config.
        """
        self.config = config

        # Get real-time settings from config or use provided parameters as fallback
        rt_config = config.get("realtime", {})
        self.chunk_duration = chunk_duration if chunk_duration != 1.0 else rt_config.get("chunk_duration", 1.0)
        self.buffer_duration = buffer_duration if buffer_duration != 30.0 else rt_config.get("buffer_duration", 30.0)
        self.microphone_device_id = microphone_device_id

        # Audio configuration from config
        self.sample_rate = rt_config.get("sample_rate", 16000)  # Whisper expects 16kHz
        self.channels = rt_config.get("channels", 1)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Convert format string to pyaudio format
        format_str = rt_config.get("format", "float32")
        self.format = pyaudio.paFloat32 if format_str == "float32" else pyaudio.paInt16
        
        # Threading and queue setup
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_running = False
        self.threads = []
        
        # Audio processing components
        self.audio_processor = AudioProcessor(config)
        self.whisper_handler = WhisperHandler(config)
        
        # Voice Activity Detection
        vad_config = config.get("vad_realtime", {})
        self.use_vad = WEBRTC_VAD_AVAILABLE and vad_config.get("enable", True)
        if self.use_vad:
            aggressiveness = vad_config.get("aggressiveness", 2)
            self.vad = webrtcvad.Vad(aggressiveness)  # Aggressiveness mode from config

        # Audio buffer for accumulating speech
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

        # Silence detection settings from config
        self.silence_threshold = rt_config.get("silence_threshold", 0.01)
        self.min_audio_length = rt_config.get("min_audio_length", 1.0)
        self.max_silence_duration = rt_config.get("max_silence_duration", 2.0)
        self.silence_start_time = None
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for real-time transcription."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal, stopping transcription...")
        self.stop()
    
    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech using VAD or simple volume threshold.
        
        Args:
            audio_chunk (np.ndarray): Audio data chunk
            
        Returns:
            bool: True if speech detected, False otherwise
        """
        if self.use_vad:
            try:
                # Convert float32 to int16 for webrtcvad
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # WebRTC VAD requires specific frame sizes (10, 20, or 30 ms)
                vad_config = self.config.get("vad_realtime", {})
                frame_duration = vad_config.get("frame_duration_ms", 30)  # ms from config
                frame_size = int(self.sample_rate * frame_duration / 1000)
                
                if len(audio_int16) >= frame_size:
                    frame = audio_bytes[:frame_size * 2]  # 2 bytes per int16
                    return self.vad.is_speech(frame, self.sample_rate)
                else:
                    return False
            except Exception as e:
                self.logger.warning(f"VAD failed, falling back to volume threshold: {e}")
        
        # Fallback to simple volume-based detection
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.silence_threshold
    
    def _audio_capture_thread(self):
        """Thread function for capturing audio from microphone."""
        self.logger.info("Starting audio capture thread...")
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        try:
            # Open audio stream
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.microphone_device_id,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info(f"Audio stream opened: {self.sample_rate}Hz, chunk size: {self.chunk_size}")
            
            while self.is_running:
                try:
                    # Read audio data
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Put audio in queue for processing
                    self.audio_queue.put(audio_array)
                    
                except Exception as e:
                    if self.is_running:  # Only log if we're still supposed to be running
                        self.logger.warning(f"Error reading audio: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            p.terminate()
            self.logger.info("Audio capture thread stopped")
    
    def _audio_processing_thread(self):
        """Thread function for processing audio and managing speech buffer."""
        self.logger.info("Starting audio processing thread...")
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Check if this chunk contains speech
                is_speech = self._is_speech(audio_chunk)
                
                with self.buffer_lock:
                    if is_speech:
                        # Add speech audio to buffer
                        self.audio_buffer.append(audio_chunk)
                        self.silence_start_time = None
                        
                        # Check if buffer is getting too long
                        buffer_duration = len(self.audio_buffer) * self.chunk_duration
                        if buffer_duration > self.buffer_duration:
                            self.logger.info(f"Buffer full ({buffer_duration:.1f}s), processing...")
                            self._process_buffer()
                    
                    else:
                        # Handle silence
                        if len(self.audio_buffer) > 0:
                            if self.silence_start_time is None:
                                self.silence_start_time = time.time()
                            
                            # Check if we've had enough silence to process the buffer
                            silence_duration = time.time() - self.silence_start_time
                            if silence_duration > self.max_silence_duration:
                                buffer_duration = len(self.audio_buffer) * self.chunk_duration
                                if buffer_duration >= self.min_audio_length:
                                    self.logger.info(f"Silence detected, processing buffer ({buffer_duration:.1f}s)...")
                                    self._process_buffer()
                                else:
                                    # Buffer too short, clear it
                                    self.audio_buffer = []
                                    self.silence_start_time = None
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing thread: {e}")
        
        # Process any remaining buffer when shutting down
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                buffer_duration = len(self.audio_buffer) * self.chunk_duration
                if buffer_duration >= self.min_audio_length:
                    self.logger.info("Processing final buffer...")
                    self._process_buffer()
        
        self.logger.info("Audio processing thread stopped")
    
    def _process_buffer(self):
        """Process accumulated audio buffer for transcription."""
        if not self.audio_buffer:
            return
        
        try:
            # Concatenate all audio chunks
            combined_audio = np.concatenate(self.audio_buffer)
            buffer_duration = len(combined_audio) / self.sample_rate
            
            self.logger.info(f"Processing {buffer_duration:.1f}s of audio...")
            
            # Add to transcription queue
            self.transcription_queue.put(combined_audio.copy())
            
            # Clear the buffer
            self.audio_buffer = []
            self.silence_start_time = None
            
        except Exception as e:
            self.logger.error(f"Error processing audio buffer: {e}")
    
    def _transcription_thread(self):
        """Thread function for transcribing audio using Whisper."""
        self.logger.info("Starting transcription thread...")
        
        while self.is_running:
            try:
                # Get audio from transcription queue
                audio_data = self.transcription_queue.get(timeout=1.0)
                
                # Transcribe audio
                start_time = time.time()
                text = self.whisper_handler.transcribe(audio_data)
                transcription_time = time.time() - start_time
                
                # Output the transcription
                if text.strip():
                    audio_duration = len(audio_data) / self.sample_rate
                    print(f"\\n[{time.strftime('%H:%M:%S')}] ({audio_duration:.1f}s, {transcription_time:.1f}s): {text}")
                    sys.stdout.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in transcription thread: {e}")
        
        # Process any remaining transcriptions when shutting down
        while not self.transcription_queue.empty():
            try:
                audio_data = self.transcription_queue.get_nowait()
                text = self.whisper_handler.transcribe(audio_data)
                if text.strip():
                    audio_duration = len(audio_data) / self.sample_rate
                    print(f"\\n[{time.strftime('%H:%M:%S')}] (final, {audio_duration:.1f}s): {text}")
                    sys.stdout.flush()
            except:
                break
        
        self.logger.info("Transcription thread stopped")
    
    def start(self):
        """Start the real-time transcription system."""
        if self.is_running:
            self.logger.warning("Transcription is already running")
            return
        
        self.is_running = True
        
        # Start threads
        self.threads = [
            threading.Thread(target=self._audio_capture_thread, name="AudioCapture"),
            threading.Thread(target=self._audio_processing_thread, name="AudioProcessing"),
            threading.Thread(target=self._transcription_thread, name="Transcription")
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        self.logger.info("Real-time transcription started!")
        print("\\n" + "="*80)
        print("🚀 REAL-TIME TRANSCRIPTION ACTIVE")
        print("Speak into your microphone - transcriptions will appear below")
        print("Press Ctrl+C to stop")
        print("="*80 + "\\n")
    
    def stop(self):
        """Stop the real-time transcription system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time transcription...")
        self.is_running = False
        
        # Wait for threads to complete
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        print("\\n" + "="*80)
        print("🛑 REAL-TIME TRANSCRIPTION STOPPED")
        print("="*80)
    
    def run(self):
        """Run the real-time transcription system until interrupted."""
        self.start()
        
        try:
            # Keep the main thread alive
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Main function to run real-time transcription."""
    config = load_config()
    
    # Create and run real-time transcriber - settings now come from config
    transcriber = RealTimeTranscriber(
        config=config
    )
    
    transcriber.run()


if __name__ == "__main__":
    main()