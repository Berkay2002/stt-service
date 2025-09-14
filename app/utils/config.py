# app/utils/config.py

def load_config():
    """
    Loads system configuration for the STT pipeline.
    Specifically optimized for RTX 3090 (24GB VRAM, 10496 CUDA cores) with:
    - Large-v3 model for maximum accuracy utilizing 24GB VRAM
    - Aggressive batch processing (24 batch size) for compute saturation
    - Enhanced parallel processing (12 CPU threads, 6 workers)
    - Ultra-responsive real-time processing (0.25s chunks)
    - Maximum VAD aggressiveness for quality speech detection
    - Float16 precision leveraging RTX 3090's tensor cores
    """
    # CPU Setup (Commented out for GPU usage)
    '''
    config = {
        "device": "cpu",           # Use "cuda" for GPU, "cpu" for CPU inference
        "model_name": "base.en",   # Change to e.g., "medium.en" or other Whisper variant as needed
        "beam_size": 5,            # Decoding beam size for Whisper
        "fp16": False              # Use FP16 if supported (set True for GPU, False for CPU)
    }
    '''

    # GPU configuration (CUDA) - Optimized for high-end RTX cards (3090, 4090, 5090)
    config = {
        # Core Whisper Model Settings
        "device": "cuda",              # Use GPU via CUDA
        "cuda_device_index": 0,        # Which GPU to use (0 for default)
        "model_name": "large-v3",      # RTX 3090 can handle large model with 24GB VRAM
        "beam_size": 10,               # Higher beam size for maximum quality on 3090
        "fp16": True,                  # Enable FP16 for faster GPU inference
        "compute_type": "float16",     # Explicit compute type for consistency
        "cpu_threads": 12,             # More CPU threads for better preprocessing (3090 benefits from more CPU power)
        "num_workers": 6,              # More parallel workers to saturate 3090's massive compute
        "batch_size": 24,              # Larger batch size to utilize 24GB VRAM effectively

        # Transcription Quality Settings
        "temperature": 0.0,                  # Use greedy decoding for consistency
        "condition_on_previous_text": True,  # Better context handling
        "compression_ratio_threshold": 2.4,  # Quality threshold
        "log_prob_threshold": -1.0,          # Log probability threshold
        "no_speech_threshold": 0.6,          # No speech detection threshold
        "vad_filter": True,                  # Enable voice activity detection
        "vad_parameters": {
            "min_silence_duration_ms": 500         # Minimum silence duration between speech segments
        },

        # Real-Time Audio Processing Settings (RTX 3090 Optimized)
        "realtime": {
            "chunk_duration": 0.25,      # Smaller chunks for ultra-responsive processing on 3090
            "buffer_duration": 60.0,     # Larger buffer to utilize massive VRAM (24GB)
            "silence_threshold": 0.003,  # Lower threshold for better speech detection
            "min_audio_length": 0.6,     # Reduced minimum for faster response times
            "max_silence_duration": 1.2, # Faster silence processing on powerful hardware
            "sample_rate": 16000,        # Audio sample rate (Whisper expects 16kHz)
            "channels": 1,               # Audio channels (mono)
            "format": "float32"          # Audio format
        },

        # Microphone Settings
        "microphone": {
            "preferred_device": "hyperx",    # Preferred microphone name pattern (case-insensitive)
            "test_duration": 2.0,            # Microphone test duration in seconds
            "record_countdown": 3,           # Countdown before recording starts
            "default_record_duration": 10.0  # Default recording duration for mic mode
        },

        # Voice Activity Detection Settings (RTX 3090 Optimized)
        "vad_realtime": {
            "aggressiveness": 3,       # Maximum aggressiveness for best quality on powerful hardware
            "frame_duration_ms": 20,   # Smaller frame duration for more responsive VAD processing
            "enable": True             # Enable/disable VAD for real-time processing
        }
    }

    return config


def get_gpu_info():
    """
    Get detailed GPU information for display purposes.

    Returns:
        dict: GPU information including name, memory, compute capability
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)

            # Get memory info
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            total_memory_gb = total_memory / (1024**3)

            # Get compute capability
            major, minor = torch.cuda.get_device_capability(device_id)
            compute_capability = f"{major}.{minor}"

            # Get current memory usage
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_id) / (1024**3)

            return {
                "available": True,
                "device_id": device_id,
                "name": gpu_name,
                "total_memory_gb": round(total_memory_gb, 1),
                "allocated_memory_gb": round(allocated, 2),
                "reserved_memory_gb": round(reserved, 2),
                "compute_capability": compute_capability,
                "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            }
        else:
            return {"available": False, "reason": "CUDA not available"}

    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "reason": f"Error: {str(e)}"}


def get_microphone_info(device_id=None):
    """
    Get detailed microphone information.

    Args:
        device_id (int, optional): Specific device ID to get info for

    Returns:
        dict: Microphone information
    """
    try:
        import pyaudio

        p = pyaudio.PyAudio()
        try:
            if device_id is not None:
                # Get specific device info
                device_info = p.get_device_info_by_index(device_id)
                return {
                    "available": True,
                    "device_id": device_id,
                    "name": device_info['name'],
                    "max_input_channels": device_info['maxInputChannels'],
                    "default_sample_rate": int(device_info['defaultSampleRate']),
                    "host_api": p.get_host_api_info_by_index(device_info['hostApi'])['name']
                }
            else:
                # Get default input device info
                default_device = p.get_default_input_device_info()
                return {
                    "available": True,
                    "device_id": default_device['index'],
                    "name": default_device['name'],
                    "max_input_channels": default_device['maxInputChannels'],
                    "default_sample_rate": int(default_device['defaultSampleRate']),
                    "host_api": p.get_host_api_info_by_index(default_device['hostApi'])['name']
                }
        finally:
            p.terminate()

    except ImportError:
        return {"available": False, "reason": "PyAudio not installed"}
    except Exception as e:
        return {"available": False, "reason": f"Error: {str(e)}"}


def get_model_presets():
    """
    Get predefined model configurations for different use cases.
    Users can easily switch between these presets.
    """
    return {
        "fast": {
            "model_name": "base.en",
            "beam_size": 5,
            "temperature": 0.0,
            "chunk_duration": 0.5,
            "buffer_duration": 30.0,
        },
        "balanced": {
            "model_name": "medium.en",
            "beam_size": 8,
            "temperature": 0.0,
            "chunk_duration": 0.3,
            "buffer_duration": 45.0,
        },
        "quality": {
            "model_name": "large-v2",
            "beam_size": 10,
            "temperature": 0.2,
            "chunk_duration": 0.2,
            "buffer_duration": 60.0,
        }
    }


def get_gpu_presets():
    """
    Get predefined GPU configurations for different hardware tiers.
    """
    return {
        "low_end": {  # GTX 1650, RTX 2060, etc.
            "model_name": "base.en",
            "batch_size": 8,
            "cpu_threads": 4,
            "num_workers": 2,
        },
        "mid_range": {  # RTX 3070, RTX 4060, etc.
            "model_name": "medium.en",
            "batch_size": 12,
            "cpu_threads": 6,
            "num_workers": 3,
        },
        "high_end": {  # RTX 3090, RTX 4090, RTX 5090, etc.
            "model_name": "large-v3",
            "batch_size": 24,
            "cpu_threads": 12,
            "num_workers": 6,
        }
    }


def validate_config(config):
    """
    Validate configuration settings and provide helpful warnings.

    Args:
        config (dict): Configuration dictionary to validate

    Returns:
        tuple: (is_valid, warnings_list)
    """
    warnings = []
    is_valid = True

    # Validate beam_size
    beam_size = config.get("beam_size", 5)
    if beam_size < 1 or beam_size > 20:
        warnings.append(f"beam_size ({beam_size}) should be between 1-20 for optimal performance")

    # Validate temperature
    temperature = config.get("temperature", 0.0)
    if temperature < 0.0 or temperature > 1.0:
        warnings.append(f"temperature ({temperature}) should be between 0.0-1.0")

    # Validate realtime settings
    rt_config = config.get("realtime", {})
    chunk_duration = rt_config.get("chunk_duration", 0.3)
    if chunk_duration < 0.1 or chunk_duration > 5.0:
        warnings.append(f"chunk_duration ({chunk_duration}) should be between 0.1-5.0 seconds")

    # Validate VAD settings
    vad_config = config.get("vad_realtime", {})
    aggressiveness = vad_config.get("aggressiveness", 2)
    if aggressiveness < 0 or aggressiveness > 3:
        warnings.append(f"VAD aggressiveness ({aggressiveness}) should be between 0-3")

    frame_duration = vad_config.get("frame_duration_ms", 30)
    if frame_duration not in [10, 20, 30]:
        warnings.append(f"VAD frame_duration_ms ({frame_duration}) must be 10, 20, or 30")

    return is_valid, warnings


def print_config_summary(config):
    """
    Print a user-friendly summary of the current configuration.

    Args:
        config (dict): Configuration dictionary
    """
    print("\\n" + "="*60)
    print("STT SERVICE CONFIGURATION SUMMARY")
    print("="*60)

    print(f"Model: {config.get('model_name', 'unknown')}")
    print(f"Device: {config.get('device', 'unknown')}")
    print(f"Compute Type: {config.get('compute_type', 'unknown')}")
    print(f"Beam Size: {config.get('beam_size', 'unknown')}")

    rt_config = config.get("realtime", {})
    print(f"\\nReal-time Settings:")
    print(f"  Chunk Duration: {rt_config.get('chunk_duration', 'unknown')}s")
    print(f"  Buffer Duration: {rt_config.get('buffer_duration', 'unknown')}s")
    print(f"  Silence Threshold: {rt_config.get('silence_threshold', 'unknown')}")

    mic_config = config.get("microphone", {})
    print(f"\\nMicrophone Settings:")
    print(f"  Preferred Device: {mic_config.get('preferred_device', 'unknown')}")
    print(f"  Test Duration: {mic_config.get('test_duration', 'unknown')}s")

    vad_config = config.get("vad_realtime", {})
    print(f"\\nVAD Settings:")
    print(f"  Enabled: {vad_config.get('enable', 'unknown')}")
    print(f"  Aggressiveness: {vad_config.get('aggressiveness', 'unknown')}")

    print("="*60)
