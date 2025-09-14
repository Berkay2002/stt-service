# app/main.py

import os
import sys
import time
from app.utils.config import load_config, validate_config, print_config_summary, get_gpu_info, get_microphone_info
from app.core.audio_processor import AudioProcessor
from app.core.whisper_handler import WhisperHandler
from app.core.microphone_capture import SimpleMicrophoneCapture
from app.core.realtime_transcription import RealTimeTranscriber, find_microphone_by_name, list_microphones
from app.monitoring.service_monitor import ServiceMonitor, MonitoringServer

def transcribe_from_file():
    """Original file-based transcription."""
    config = load_config()
    audio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'harvard.wav')
    print(f"Audio file path: {audio_path}")
    
    audio_processor = AudioProcessor(config)
    whisper = WhisperHandler(config)
    
    audio_data = audio_processor.load_audio(audio_path)
    recognized_text = whisper.transcribe(audio_data)
    print("Transcribed text:", recognized_text)

def transcribe_from_microphone():
    """Microphone-based transcription (10-second recording)."""
    config = load_config()
    mic_capture = SimpleMicrophoneCapture()
    whisper = WhisperHandler(config)
    
    try:
        # List available microphones
        mic_capture.list_microphones()
        
        # Get microphone settings from config
        mic_config = config.get("microphone", {})
        test_duration = mic_config.get("test_duration", 2.0)
        record_duration = mic_config.get("default_record_duration", 10.0)
        countdown = mic_config.get("record_countdown", 3)

        # Test microphone
        print(f"\\nTesting microphone for {test_duration} seconds...")
        if not mic_capture.test_microphone(duration=test_duration):
            print("Microphone test failed. Please check your microphone.")
            return

        print("\\nMicrophone working!")

        # Record with countdown
        audio_data = mic_capture.record_with_countdown(
            duration=record_duration,  # Record duration from config
            countdown=countdown        # Countdown from config
        )
        
        if len(audio_data) > 0:
            # Show audio info
            audio_info = mic_capture.get_audio_info(audio_data)
            print(f"Audio info: {audio_info['duration']:.2f}s, RMS: {audio_info['rms_level']:.4f}")
            
            if audio_info['is_silent']:
                print("Warning: Audio appears to be silent")
            
            # Transcribe
            print("Transcribing...")
            recognized_text = whisper.transcribe(audio_data)
            print("Transcribed text:", recognized_text)
        else:
            print("No audio was recorded")
            
    except KeyboardInterrupt:
        print("\\nRecording interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mic_capture.cleanup()

def transcribe_realtime():
    """Real-time continuous transcription."""
    config = load_config()
    
    print("\\n" + "="*80)
    print("STARTING REAL-TIME TRANSCRIPTION")
    print("="*80)

    # Display GPU Information
    print("\\n🖥️  GPU INFORMATION:")
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"   Name: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['total_memory_gb']} GB total")
        print(f"   Compute Capability: {gpu_info['compute_capability']}")
        print(f"   CUDA Version: {gpu_info['cuda_version']}")
        print(f"   Current Usage: {gpu_info['allocated_memory_gb']} GB allocated, {gpu_info['reserved_memory_gb']} GB reserved")
    else:
        print(f"   ❌ GPU not available: {gpu_info['reason']}")

    # Find preferred microphone from config
    mic_config = config.get("microphone", {})
    preferred_device = mic_config.get("preferred_device", "hyperx")

    preferred_device_id = find_microphone_by_name(preferred_device)

    # Display Microphone Information
    print("\\n🎤 MICROPHONE INFORMATION:")
    if preferred_device_id is not None:
        print(f"   Using preferred device: {preferred_device}")
        mic_info = get_microphone_info(preferred_device_id)
        if mic_info["available"]:
            print(f"   Device ID: {mic_info['device_id']}")
            print(f"   Name: {mic_info['name']}")
            print(f"   Channels: {mic_info['max_input_channels']}")
            print(f"   Sample Rate: {mic_info['default_sample_rate']} Hz")
            print(f"   Host API: {mic_info['host_api']}")
        else:
            print(f"   ❌ Error getting mic info: {mic_info['reason']}")
    else:
        print(f"   {preferred_device} microphone not found, using default microphone")
        default_mic_info = get_microphone_info()
        if default_mic_info["available"]:
            print(f"   Device ID: {default_mic_info['device_id']}")
            print(f"   Name: {default_mic_info['name']}")
            print(f"   Channels: {default_mic_info['max_input_channels']}")
            print(f"   Sample Rate: {default_mic_info['default_sample_rate']} Hz")
            print(f"   Host API: {default_mic_info['host_api']}")
        else:
            print(f"   ❌ Error getting default mic info: {default_mic_info['reason']}")

        print("\\n   Available microphones:")
        mics = list_microphones()
        for device_id, name, channels in mics[:10]:  # Show first 10 to avoid clutter
            print(f"     Device {device_id}: {name} (Channels: {channels})")

    # Display Model and Configuration Information
    print("\\n⚙️  MODEL CONFIGURATION:")
    print(f"   Model: {config.get('model_name', 'unknown')}")
    print(f"   Compute Type: {config.get('compute_type', 'unknown')}")
    print(f"   Beam Size: {config.get('beam_size', 'unknown')}")
    print(f"   Batch Size: {config.get('batch_size', 'unknown')}")

    rt_config = config.get('realtime', {})
    print(f"   Chunk Duration: {rt_config.get('chunk_duration', 'unknown')}s")
    print(f"   Buffer Duration: {rt_config.get('buffer_duration', 'unknown')}s")

    vad_config = config.get('vad_realtime', {})
    print(f"   VAD Enabled: {vad_config.get('enable', 'unknown')}")
    print(f"   VAD Aggressiveness: {vad_config.get('aggressiveness', 'unknown')}")

    print("\\nThis will continuously transcribe everything you say.")
    print("Speech will be detected automatically and transcribed in real-time.")
    print("Press Ctrl+C at any time to stop.")
    print("="*80)
    
    # Wait a moment for user to read
    print("\\nStarting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    try:
        # Create and run real-time transcriber - settings now come from config
        transcriber = RealTimeTranscriber(
            config=config,
            microphone_device_id=preferred_device_id  # Use preferred microphone if found
        )
        
        transcriber.run()
        
    except KeyboardInterrupt:
        print("\\n\\nReal-time transcription stopped by user")
    except Exception as e:
        print(f"\\nError during real-time transcription: {e}")

def show_config():
    """Show current configuration and validate it."""
    config = load_config()

    # Print configuration summary
    print_config_summary(config)

    # Show hardware information
    print("\\n" + "="*60)
    print("HARDWARE DETECTION")
    print("="*60)

    # GPU Info
    gpu_info = get_gpu_info()
    print("\\n🖥️  GPU Information:")
    if gpu_info["available"]:
        print(f"   ✅ {gpu_info['name']}")
        print(f"   Memory: {gpu_info['total_memory_gb']} GB")
        print(f"   Compute: {gpu_info['compute_capability']}")
        print(f"   CUDA: {gpu_info['cuda_version']}")
    else:
        print(f"   ❌ {gpu_info['reason']}")

    # Microphone Info
    print("\\n🎤 Microphone Information:")
    mic_config = config.get("microphone", {})
    preferred_device = mic_config.get("preferred_device", "hyperx")
    preferred_device_id = find_microphone_by_name(preferred_device)

    if preferred_device_id is not None:
        mic_info = get_microphone_info(preferred_device_id)
        if mic_info["available"]:
            print(f"   ✅ Found preferred: {mic_info['name']}")
            print(f"   Device ID: {mic_info['device_id']}")
        else:
            print(f"   ❌ {mic_info['reason']}")
    else:
        print(f"   ⚠️  Preferred '{preferred_device}' not found")
        default_mic = get_microphone_info()
        if default_mic["available"]:
            print(f"   ✅ Using default: {default_mic['name']}")

    # Validate configuration
    is_valid, warnings = validate_config(config)

    if warnings:
        print("\\n" + "="*60)
        print("CONFIGURATION WARNINGS")
        print("="*60)
        for warning in warnings:
            print(f"⚠️  {warning}")
        print("="*60)

    if is_valid and not warnings:
        print("\\n✅ Configuration is valid and optimized!")


def show_help():
    """Show usage instructions."""
    print("\\n" + "="*60)
    print("WHISPER TRANSCRIPTION TOOL")
    print("="*60)
    print("Usage:")
    print("  python main.py                    - Transcribe from audio file")
    print("  python main.py --mic             - Record from microphone (duration from config)")
    print("  python main.py --realtime        - Real-time continuous transcription")
    print("  python main.py --monitor         - Run monitoring server only")
    print("  python main.py --config          - Show current configuration")
    print("  python main.py --help            - Show this help message")
    print()
    print("Real-time mode features:")
    print("  [*] Continuous microphone monitoring")
    print("  [*] Automatic speech detection")
    print("  [*] Voice Activity Detection (VAD)")
    print("  [*] Smart audio buffering")
    print("  [*] Live transcription output")
    print("  [*] All settings configurable in app/utils/config.py")
    print()
    print("Configuration:")
    print("  [*] All settings are now configurable in app/utils/config.py")
    print("  [*] Optimized presets available for different GPU tiers")
    print("  [*] Microphone, audio, and VAD settings fully customizable")
    print("  [*] Use --config to see current settings")
    print()
    print("Monitoring endpoints (when --monitor is used):")
    print("  Health check: http://localhost:9091/health")
    print("  Metrics: http://localhost:9091/metrics")
    print("  Service info: http://localhost:9091/info")
    print("="*60)

def run_monitoring_only():
    """Run only the monitoring server without transcription."""
    config = load_config()
    
    print("\\n" + "="*60)
    print("STARTING STT SERVICE MONITORING")
    print("="*60)
    
    try:
        # Initialize monitor
        print("Initializing service monitor...")
        monitor = ServiceMonitor(config)
        
        # Create monitoring server
        monitoring_server = MonitoringServer(monitor, port=9091)
        
        print("Monitoring server starting on port 9091")
        print("Health check: http://localhost:9091/health")
        print("Metrics: http://localhost:9091/metrics")
        print("Service info: http://localhost:9091/info")
        print("Press Ctrl+C to stop...")
        print("="*60)
        
        # Run server
        monitoring_server.run(debug=False)
        
    except KeyboardInterrupt:
        print("\\nMonitoring server stopped by user")
    except Exception as e:
        print(f"Error running monitoring server: {e}")

def main():
    """Main entry point with improved command line handling."""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            show_help()
        elif arg in ['--config', '-c', 'config']:
            show_config()
        elif arg in ['--mic', '-m', 'mic']:
            transcribe_from_microphone()
        elif arg in ['--realtime', '-r', 'realtime', 'real-time']:
            transcribe_realtime()
        elif arg in ['--file', '-f', 'file']:
            transcribe_from_file()
        elif arg in ['--monitor', '-mon', 'monitor']:
            run_monitoring_only()
        else:
            print(f"Unknown argument: {arg}")
            show_help()
    else:
        # Default behavior - transcribe from file
        transcribe_from_file()

if __name__ == '__main__':
    main()