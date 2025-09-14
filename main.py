# main.py

"""
Main entry point for the STT Service
Supports both traditional file-based processing and WebSocket real-time streaming
"""

import argparse
import asyncio
import sys
from typing import Optional

from app.utils.config import load_config, print_config_summary
from app.utils.logger import setup_logging, get_logger


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="STT Service - Speech-to-Text with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py websocket                    # Start WebSocket server
  python main.py websocket --port 8080       # WebSocket server on custom port
  python main.py file input.wav              # Process single file
  python main.py microphone                  # Record from microphone
  python main.py config --show               # Show current configuration

WebSocket Endpoints:
  ws://localhost:8000/ws/transcribe          # WebSocket transcription
  http://localhost:8000/health               # Health check
  http://localhost:8000/stats                # Service statistics
  http://localhost:9091/health               # Monitoring service
        """
    )

    parser.add_argument(
        "mode",
        choices=["websocket", "file", "microphone", "config"],
        help="Operating mode"
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path (for file mode) or audio file (for processing)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket server port (default: 8000)"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="WebSocket server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--max-connections",
        type=int,
        default=50,
        help="Maximum WebSocket connections (default: 50)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show configuration and exit (for config mode)"
    )

    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable word-level timestamps (for file/microphone modes)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = get_logger("STT_Main")

    # Load configuration
    config = load_config()

    try:
        if args.mode == "websocket":
            run_websocket_server(config, args, logger)

        elif args.mode == "file":
            if not args.input:
                logger.error("File path required for file mode")
                sys.exit(1)
            run_file_processing(config, args.input, args.timestamps, logger)

        elif args.mode == "microphone":
            run_microphone_recording(config, args.timestamps, logger)

        elif args.mode == "config":
            if args.show:
                print_config_summary(config)
            else:
                logger.info("Configuration loaded successfully")
                print_config_summary(config)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def run_websocket_server(config: dict, args: argparse.Namespace, logger) -> None:
    """Run the WebSocket STT server"""
    logger.info("Starting WebSocket STT Server...")

    # Update config with command line arguments
    config.setdefault("websocket", {})
    config["websocket"]["host"] = args.host
    config["websocket"]["port"] = args.port
    config["websocket"]["max_connections"] = args.max_connections

    # Import and run WebSocket server
    from app.core.websocket_launcher import WebSocketLauncher

    launcher = WebSocketLauncher(config)
    asyncio.run(launcher.run())


def run_file_processing(config: dict, file_path: str, enable_timestamps: bool, logger) -> None:
    """Process a single audio file"""
    logger.info(f"Processing audio file: {file_path}")

    from app.core.whisper_handler import WhisperHandler
    from app.core.audio_processor import AudioProcessor

    # Initialize components
    whisper_handler = WhisperHandler(config)
    audio_processor = AudioProcessor(config)

    try:
        # Load and process audio
        audio_data = audio_processor.load_audio(file_path)
        audio_info = audio_processor.get_audio_info(audio_data)

        logger.info(f"Audio info: {audio_info}")

        # Transcribe
        if enable_timestamps:
            logger.info("Transcribing with timestamps...")
            segments = whisper_handler.transcribe_with_timestamps(audio_data)

            print("\\nTranscription with timestamps:")
            print("-" * 50)
            for segment in segments:
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")

                # Show word-level timestamps if available
                if 'words' in segment and segment['words']:
                    for word in segment['words']:
                        word_text = word.get('word', '')
                        start_time = word.get('start', 0)
                        end_time = word.get('end', 0)
                        confidence = word.get('probability', 0)
                        print(f"  {word_text}: {start_time:.2f}s-{end_time:.2f}s (conf: {confidence:.2f})")
        else:
            logger.info("Transcribing...")
            text = whisper_handler.transcribe(audio_data)

            print("\\nTranscription:")
            print("-" * 50)
            print(text)

        print("-" * 50)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


def run_microphone_recording(config: dict, enable_timestamps: bool, logger) -> None:
    """Record from microphone and transcribe"""
    logger.info("Starting microphone recording...")

    try:
        import pyaudio
        import numpy as np
        import time

        from app.core.whisper_handler import WhisperHandler
        from app.core.audio_processor import AudioProcessor

        # Initialize components
        whisper_handler = WhisperHandler(config)
        audio_processor = AudioProcessor(config)

        # Audio configuration
        SAMPLE_RATE = 16000
        CHANNELS = 1
        CHUNK_SIZE = 1024
        RECORD_DURATION = 5  # seconds

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Find preferred microphone
        preferred_device = config.get("microphone", {}).get("preferred_device", "").lower()
        device_index = None

        if preferred_device:
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if preferred_device in device_info['name'].lower():
                    device_index = i
                    logger.info(f"Using preferred microphone: {device_info['name']}")
                    break

        if device_index is None:
            device_info = p.get_default_input_device_info()
            device_index = device_info['index']
            logger.info(f"Using default microphone: {device_info['name']}")

        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )

        print(f"Recording for {RECORD_DURATION} seconds... Speak now!")

        # Record audio
        audio_chunks = []
        for _ in range(int(SAMPLE_RATE * RECORD_DURATION / CHUNK_SIZE)):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_chunks.append(audio_chunk)

        # Stop recording
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Recording finished. Processing...")

        # Concatenate audio chunks
        audio_data = np.concatenate(audio_chunks)

        # Transcribe
        if enable_timestamps:
            segments = whisper_handler.transcribe_with_timestamps(audio_data)

            print("\\nTranscription with timestamps:")
            print("-" * 50)
            for segment in segments:
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
        else:
            text = whisper_handler.transcribe(audio_data)

            print("\\nTranscription:")
            print("-" * 50)
            print(text)

        print("-" * 50)

    except ImportError:
        logger.error("PyAudio not installed. Install with: pip install pyaudio")
        raise
    except Exception as e:
        logger.error(f"Error with microphone recording: {e}")
        raise


if __name__ == "__main__":
    main()