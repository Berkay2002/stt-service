# app/utils/config.py

def load_config():
    """
    Loads system configuration for the STT pipeline.
    Currently configured for GPU (CUDA); see CPU section for CPU-only options.
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

    # GPU configuration (CUDA) - Enabled for GTX 1650
    config = {
        "device": "cuda",          # Use GPU via CUDA
        "cuda_device_index": 0,    # Which GPU to use (0 for default)
        "model_name": "base.en",   # Or "large-v2", "medium.en", etc.
        "beam_size": 5,
        "fp16": True               # Enable FP16 for faster GPU inference
    }

    return config
