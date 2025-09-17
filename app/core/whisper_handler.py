# app/core/whisper_handler.py
import logging
from typing import Union, List, Optional, Dict, Any
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class WhisperHandler:
    """
    Enhanced Whisper model handler with partial/final transcription support using faster-whisper.
    """
    
    def __init__(self, config: dict):
        """
        Initialize WhisperHandler with configuration.
        
        Args:
            config (dict): Configuration dictionary with model parameters
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            )
        
        self.config = config
        self.logger = self._setup_logger()
        self.model = None
        
        # Partial transcription configuration
        self.enable_partial_transcription = config.get("enable_partial_transcription", True)
        self.partial_beam_size = config.get("partial_beam_size", 1)  # Faster for partial
        self.final_beam_size = config.get("beam_size", 5)  # Higher quality for final
        
        # Cache for repeated audio segments (optimization)
        self.audio_cache = {}
        self.cache_max_size = config.get("cache_max_size", 10)
        
        self._load_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the whisper handler."""
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
    
    def _load_model(self):
        """Load the Whisper model based on configuration."""
        try:
            device = self.config.get("device", "cpu")
            model_name = self.config.get("model_name", "base.en")

            # Determine compute type based on device and config
            if device == "cuda":
                compute_type = self.config.get("compute_type", "float16" if self.config.get("fp16", False) else "float32")
                cuda_device_index = self.config.get("cuda_device_index", 0)
                cpu_threads = self.config.get("cpu_threads", 8)
                num_workers = self.config.get("num_workers", 4)

                self.logger.info(f"Loading Whisper model '{model_name}' on GPU {cuda_device_index} with {compute_type}")
                self.logger.info(f"Using {cpu_threads} CPU threads and {num_workers} workers for high-end RTX optimization")

                self.model = WhisperModel(
                    model_name,
                    device=device,
                    device_index=cuda_device_index,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=num_workers
                )
            else:
                compute_type = "int8" if not self.config.get("fp16", False) else "float32"
                cpu_threads = self.config.get("cpu_threads", 4)

                self.logger.info(f"Loading Whisper model '{model_name}' on CPU with {compute_type}")
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads
                )

            self.logger.info("Whisper model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Get transcription parameters from config
            beam_size = self.config.get("beam_size", 5)
            temperature = self.config.get("temperature", 0.0)
            condition_on_previous_text = self.config.get("condition_on_previous_text", True)
            compression_ratio_threshold = self.config.get("compression_ratio_threshold", 2.4)
            log_prob_threshold = self.config.get("log_prob_threshold", -1.0)
            no_speech_threshold = self.config.get("no_speech_threshold", 0.6)
            vad_filter = self.config.get("vad_filter", True)
            vad_parameters = self.config.get("vad_parameters", {})

            self.logger.info(f"Starting high-performance transcription with beam_size={beam_size}, temperature={temperature}")

            # Transcribe audio with optimized parameters for high-end GPUs
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=beam_size,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
            # Log detected language
            self.logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Collect all text segments
            transcribed_text = ""
            segment_count = 0
            
            for segment in segments:
                transcribed_text += segment.text + " "
                segment_count += 1
                self.logger.debug(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            
            # Clean up the text
            transcribed_text = transcribed_text.strip()
            
            self.logger.info(f"Transcription completed: {segment_count} segments, {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def transcribe_partial(self, audio_data: np.ndarray) -> str:
        """
        Fast transcription for partial results with optimized parameters.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text (partial result)
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Generate cache key for audio data
            audio_hash = hash(audio_data.tobytes())
            
            # Check cache first
            if audio_hash in self.audio_cache:
                cached_result = self.audio_cache[audio_hash]
                self.logger.debug(f"Using cached partial result: {len(cached_result)} chars")
                return cached_result
            
            # Use optimized parameters for fast partial transcription
            self.logger.debug(f"Starting partial transcription with beam_size={self.partial_beam_size}")

            # Optimized parameters for speed over accuracy
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=self.partial_beam_size,  # Lower beam size for speed
                temperature=0.0,  # No temperature sampling for consistency
                condition_on_previous_text=False,  # Disable for independence
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=False,  # Disable VAD for partial (already done upstream)
                language=self.config.get("language", None),  # Use specific language if set
                task="transcribe"  # Explicit transcribe task
            )
            
            # Collect text quickly - only extract text, skip detailed analysis
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + " "
                # Break early for partial results to save time
                if len(transcribed_text) > 50:  # Stop after reasonable partial text
                    break
            
            # Clean up the text
            transcribed_text = transcribed_text.strip()
            
            # Cache the result for potential reuse
            self._update_cache(audio_hash, transcribed_text)
            
            self.logger.debug(f"Partial transcription completed: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            self.logger.error(f"Partial transcription failed: {str(e)}")
            # Return empty string for partial failures to avoid breaking flow
            return ""
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Transcribe audio data with detailed timestamp information.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            
        Returns:
            List[Dict]: List of segments with timestamps and text
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Get transcription parameters from config
            beam_size = self.config.get("beam_size", 5)
            temperature = self.config.get("temperature", 0.0)
            condition_on_previous_text = self.config.get("condition_on_previous_text", True)
            compression_ratio_threshold = self.config.get("compression_ratio_threshold", 2.4)
            log_prob_threshold = self.config.get("log_prob_threshold", -1.0)
            no_speech_threshold = self.config.get("no_speech_threshold", 0.6)
            vad_filter = self.config.get("vad_filter", True)
            vad_parameters = self.config.get("vad_parameters", {})

            segments, info = self.model.transcribe(
                audio_data,
                beam_size=beam_size,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                word_timestamps=True,  # Enable word-level timestamps
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
            results = []
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_data = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": getattr(word, 'probability', None)
                        }
                        segment_data["words"].append(word_data)
                
                results.append(segment_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Transcription with timestamps failed: {str(e)}")
            raise
    
    def _update_cache(self, audio_hash: int, result: str) -> None:
        """
        Update the audio result cache with size management.
        
        Args:
            audio_hash: Hash of the audio data
            result: Transcription result to cache
        """
        if len(self.audio_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        
        self.audio_cache[audio_hash] = result
    
    def clear_cache(self) -> None:
        """Clear the audio result cache."""
        self.audio_cache.clear()
        self.logger.info("Cleared audio transcription cache")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information including partial processing capabilities
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.config.get("model_name", "unknown"),
            "device": self.config.get("device", "unknown"),
            "beam_size": self.config.get("beam_size", 5),
            "partial_beam_size": self.partial_beam_size,
            "fp16": self.config.get("fp16", False),
            "partial_transcription_enabled": self.enable_partial_transcription,
            "cache_size": len(self.audio_cache),
            "cache_max_size": self.cache_max_size
        }
