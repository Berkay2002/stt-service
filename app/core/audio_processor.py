# app/core/audio_processor.py
import os
import logging
import numpy as np
import torch
import torchaudio
from typing import Union, Optional, Dict, Any


class AudioProcessor:
    """
    Audio processing class for loading and preprocessing audio files
    using torchaudio for speech-to-text transcription.
    """
    
    def __init__(self, config: dict):
        """
        Initialize AudioProcessor with configuration.
        
        Args:
            config (dict): Configuration dictionary containing processing parameters
        """
        self.config = config
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.logger = self._setup_logger()
        
        # Set torch device based on config (CPU/GPU)
        device_name = self.config.get("device", "cpu")
        self.device = torch.device(device_name)
        self.logger.info(f"AudioProcessor using device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the audio processor."""
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
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file using torchaudio and convert to format expected by Whisper.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Audio data as numpy array (float32, mono, 16kHz)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If audio loading fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.logger.info(f"Loading audio file with torchaudio: {audio_path}")
        
        try:
            # Load audio with torchaudio
            waveform, original_sample_rate = torchaudio.load(audio_path)
            
            self.logger.info(f"Original audio: shape={waveform.shape}, sr={original_sample_rate}")
            
            # Move to configured device (CPU or GPU)
            waveform = waveform.to(self.device)
            
            # Convert to mono if stereo (torchaudio loads as [channels, samples])
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                self.logger.info("Converted stereo to mono")
            
            # Resample to 16kHz if needed
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=self.sample_rate
                ).to(self.device)
                waveform = resampler(waveform)
                self.logger.info(f"Resampled from {original_sample_rate}Hz to {self.sample_rate}Hz")
            
            # Remove channel dimension and convert to numpy array
            # faster-whisper expects numpy.ndarray, not torch.Tensor
            audio_data = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            
            # Basic validation
            if len(audio_data) == 0:
                raise ValueError("Audio file appears to be empty")
            
            self.logger.info(f"Audio loading complete: {len(audio_data)/self.sample_rate:.2f}s duration")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply additional preprocessing to audio data if needed.
        
        Args:
            audio_data (np.ndarray): Raw audio data
            
        Returns:
            np.ndarray: Preprocessed audio data
        """
        # Convert to torch tensor for processing
        audio_tensor = torch.from_numpy(audio_data).to(self.device)
        
        # Normalize audio to [-1, 1] range if needed
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 1.0:
            audio_tensor = audio_tensor / max_val
            self.logger.info("Audio normalized to [-1, 1] range")
        
        # Apply additional preprocessing if needed
        audio_tensor = self._apply_audio_effects(audio_tensor)
        
        # Convert back to numpy
        return audio_tensor.detach().cpu().numpy().astype(np.float32)
    
    def _apply_audio_effects(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply audio effects using torchaudio transforms.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor
            
        Returns:
            torch.Tensor: Processed audio tensor
        """
        # Example: Apply a high-pass filter to remove low-frequency noise
        # You can uncomment and customize as needed
        
        # highpass = torchaudio.transforms.Highpass(
        #     sample_rate=self.sample_rate, 
        #     cutoff_freq=80.0
        # ).to(self.device)
        # audio_tensor = highpass(audio_tensor.unsqueeze(0)).squeeze(0)
        
        # Example: Apply volume normalization
        # vol_transform = torchaudio.transforms.Vol(gain=1.0).to(self.device)
        # audio_tensor = vol_transform(audio_tensor.unsqueeze(0)).squeeze(0)
        
        return audio_tensor
    
    def get_audio_info(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Get information about loaded audio data.
        
        Args:
            audio_data (np.ndarray): Audio data
            
        Returns:
            Dict[str, Any]: Audio information
        """
        return {
            "duration": len(audio_data) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "channels": 1,  # Always mono for Whisper
            "data_type": str(audio_data.dtype),
            "shape": audio_data.shape,
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "min_amplitude": float(np.min(np.abs(audio_data))),
            "rms_level": float(np.sqrt(np.mean(audio_data**2))),
            "backend": "torchaudio"
        }
    
    def save_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save audio data to file using torchaudio.
        
        Args:
            audio_data (np.ndarray): Audio data to save
            output_path (str): Path where to save the audio file
        """
        try:
            # Convert numpy to torch tensor and add channel dimension
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dim
            
            # Save using torchaudio
            torchaudio.save(output_path, audio_tensor, self.sample_rate)
            self.logger.info(f"Audio saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio to {output_path}: {str(e)}")
            raise
    
    def load_audio_segment(self, audio_path: str, start_time: float, duration: float) -> np.ndarray:
        """
        Load a specific segment of audio file.
        
        Args:
            audio_path (str): Path to audio file
            start_time (float): Start time in seconds
            duration (float): Duration in seconds
            
        Returns:
            np.ndarray: Audio segment as numpy array
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Get file info first
            info = torchaudio.info(audio_path)
            original_sample_rate = info.sample_rate
            
            # Calculate frame offsets
            frame_offset = int(start_time * original_sample_rate)
            num_frames = int(duration * original_sample_rate)
            
            # Load specific segment
            waveform, _ = torchaudio.load(
                audio_path, 
                frame_offset=frame_offset, 
                num_frames=num_frames
            )
            
            self.logger.info(f"Loaded segment: {start_time}s-{start_time+duration}s")
            
            # Process same as regular load_audio
            waveform = waveform.to(self.device)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=self.sample_rate
                ).to(self.device)
                waveform = resampler(waveform)
            
            return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to load audio segment: {str(e)}")
            raise
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported audio formats.
        
        Returns:
            list: List of supported file extensions
        """
        # Common formats supported by torchaudio
        return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma']
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate if audio file can be loaded.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            info = torchaudio.info(audio_path)
            self.logger.info(f"Audio validation - Duration: {info.num_frames/info.sample_rate:.2f}s, "
                           f"Sample Rate: {info.sample_rate}Hz, Channels: {info.num_channels}")
            return True
        except Exception as e:
            self.logger.error(f"Audio validation failed for {audio_path}: {str(e)}")
            return False
