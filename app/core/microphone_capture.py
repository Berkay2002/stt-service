# app/core/microphone_capture.py
import pyaudio
import numpy as np
import logging
import time
from typing import Optional


class SimpleMicrophoneCapture:
    """
    Simple microphone capture for speech-to-text transcription.
    Records audio for a specified duration without complex VAD.
    """
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        """
        Initialize microphone capture.
        
        Args:
            sample_rate (int): Sample rate for recording (16kHz for Whisper)
            chunk_size (int): Audio chunk size for recording buffer
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = 1
        
        self.audio = pyaudio.PyAudio()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for microphone capture."""
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
    
    def list_microphones(self):
        """List available microphone devices."""
        self.logger.info("Available microphone devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            max_channels = info['maxInputChannels']
            if isinstance(max_channels, (int, float)) and max_channels > 0:
                self.logger.info(f"  Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
    
    def record_audio(self, duration=5.0, device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio for specified duration.
        
        Args:
            duration (float): Recording duration in seconds
            device_index (int, optional): Specific microphone device to use
            
        Returns:
            np.ndarray: Recorded audio data as float32 array
        """
        self.logger.info(f"🎤 Recording for {duration} seconds...")
        
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            total_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            # Record audio chunks
            for i in range(total_chunks):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.int16))
                    
                    # Progress indicator
                    if i % (total_chunks // 10) == 0:
                        progress = (i / total_chunks) * 100
                        print(f"Recording... {progress:.0f}%", end='\r')
                        
                except Exception as e:
                    self.logger.warning(f"Error reading audio chunk: {e}")
                    continue
            
            # Clean up stream
            stream.stop_stream()
            stream.close()
            
            print()  # New line after progress indicator
            
            if not frames:
                self.logger.error("No audio data recorded")
                return np.array([], dtype=np.float32)
            
            # Combine frames and convert to float32 for Whisper
            audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
            
            # Basic audio validation
            if np.all(audio_data == 0):
                self.logger.warning("Recorded audio appears to be silent")
            
            actual_duration = len(audio_data) / self.sample_rate
            self.logger.info(f"✅ Successfully recorded {actual_duration:.2f}s of audio")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to record audio: {e}")
            raise
    
    def record_with_countdown(self, duration=10.0, countdown=3, device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio with a countdown before starting.
        
        Args:
            duration (float): Recording duration in seconds
            countdown (int): Countdown seconds before recording starts
            device_index (int, optional): Specific microphone device to use
            
        Returns:
            np.ndarray: Recorded audio data
        """
        self.logger.info(f"Get ready to speak! Recording will start in {countdown} seconds...")
        
        # Countdown
        for i in range(countdown, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
        
        print("🔴 RECORDING NOW! Speak clearly...")
        return self.record_audio(duration, device_index)
    
    def test_microphone(self, duration=2.0, device_index: Optional[int] = None) -> bool:
        """
        Test microphone by recording a short sample.
        
        Args:
            duration (float): Test recording duration
            device_index (int, optional): Specific microphone device to use
            
        Returns:
            bool: True if microphone works, False otherwise
        """
        try:
            self.logger.info("Testing microphone...")
            audio_data = self.record_audio(duration, device_index)
            
            if len(audio_data) > 0:
                max_amplitude = np.max(np.abs(audio_data))
                rms_level = np.sqrt(np.mean(audio_data**2))
                
                self.logger.info(f"✅ Microphone test successful!")
                self.logger.info(f"   Max amplitude: {max_amplitude:.4f}")
                self.logger.info(f"   RMS level: {rms_level:.4f}")
                
                if max_amplitude < 0.001:
                    self.logger.warning("⚠️  Audio level very low - check microphone volume")
                
                return True
            else:
                self.logger.error("Microphone test failed - no audio recorded")
                return False
                
        except Exception as e:
            self.logger.error(f"Microphone test failed: {e}")
            return False
    
    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        Get information about recorded audio.
        
        Args:
            audio_data (np.ndarray): Audio data
            
        Returns:
            dict: Audio information
        """
        if len(audio_data) == 0:
            return {"status": "empty"}
        
        return {
            "duration": len(audio_data) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "data_type": str(audio_data.dtype),
            "shape": audio_data.shape,
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "rms_level": float(np.sqrt(np.mean(audio_data**2))),
            "is_silent": float(np.max(np.abs(audio_data))) < 0.001
        }
    
    def cleanup(self):
        """Clean up PyAudio resources."""
        if hasattr(self, 'audio'):
            self.audio.terminate()
            self.logger.info("Microphone resources cleaned up")
