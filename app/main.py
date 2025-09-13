# app/main.py

import os
import sys
import time
from app.utils.config import load_config
from app.core.audio_processor import AudioProcessor
from app.core.whisper_handler import WhisperHandler
from app.core.microphone_capture import SimpleMicrophoneCapture
from app.core.realtime_transcription import RealTimeTranscriber
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
        
        # Test microphone
        print("\\nTesting microphone...")
        if not mic_capture.test_microphone(duration=2.0):
            print("Microphone test failed. Please check your microphone.")
            return
        
        print("\\n✅ Microphone working!")
        
        # Record with countdown
        audio_data = mic_capture.record_with_countdown(
            duration=10.0,  # Record for 10 seconds
            countdown=3     # 3 second countdown
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
        print("\\n🛑 Recording interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mic_capture.cleanup()

def transcribe_realtime():
    """Real-time continuous transcription."""
    config = load_config()
    
    print("\\n" + "="*60)
    print("🎤 STARTING REAL-TIME TRANSCRIPTION")
    print("="*60)
    print("This will continuously transcribe everything you say.")
    print("Speech will be detected automatically and transcribed in real-time.")
    print("Press Ctrl+C at any time to stop.")
    print("="*60)
    
    # Wait a moment for user to read
    print("\\nStarting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    try:
        # Create and run real-time transcriber
        transcriber = RealTimeTranscriber(
            config=config,
            chunk_duration=0.5,      # Process audio in 0.5 second chunks
            buffer_duration=30.0     # Maximum 30 seconds of audio in buffer
        )
        
        transcriber.run()
        
    except KeyboardInterrupt:
        print("\\n\\n🛑 Real-time transcription stopped by user")
    except Exception as e:
        print(f"\\nError during real-time transcription: {e}")

def show_help():
    """Show usage instructions."""
    print("\\n" + "="*60)
    print("🎙️  WHISPER TRANSCRIPTION TOOL")
    print("="*60)
    print("Usage:")
    print("  python main.py                    - Transcribe from audio file")
    print("  python main.py --mic             - Record 10s from microphone")
    print("  python main.py --realtime        - Real-time continuous transcription")
    print("  python main.py --monitor         - Run monitoring server only")
    print("  python main.py --help            - Show this help message")
    print()
    print("Real-time mode features:")
    print("  ✅ Continuous microphone monitoring")
    print("  ✅ Automatic speech detection")
    print("  ✅ Voice Activity Detection (VAD)")
    print("  ✅ Smart audio buffering")
    print("  ✅ Live transcription output")
    print()
    print("Monitoring endpoints (when --monitor is used):")
    print("  📊 Health check: http://localhost:9091/health")
    print("  📈 Metrics: http://localhost:9091/metrics")
    print("  ℹ️  Service info: http://localhost:9091/info")
    print("="*60)

def run_monitoring_only():
    """Run only the monitoring server without transcription."""
    config = load_config()
    
    print("\\n" + "="*60)
    print("🔍 STARTING STT SERVICE MONITORING")
    print("="*60)
    
    try:
        # Initialize monitor
        print("Initializing service monitor...")
        monitor = ServiceMonitor(config)
        
        # Create monitoring server
        monitoring_server = MonitoringServer(monitor, port=9091)
        
        print("🌐 Monitoring server starting on port 9091")
        print("📊 Health check: http://localhost:9091/health")
        print("📈 Metrics: http://localhost:9091/metrics")
        print("ℹ️  Service info: http://localhost:9091/info")
        print("Press Ctrl+C to stop...")
        print("="*60)
        
        # Run server
        monitoring_server.run(debug=False)
        
    except KeyboardInterrupt:
        print("\\n🛑 Monitoring server stopped by user")
    except Exception as e:
        print(f"❌ Error running monitoring server: {e}")

def main():
    """Main entry point with improved command line handling."""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            show_help()
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