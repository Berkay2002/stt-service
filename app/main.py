import os
from app.config import load_config
from app.audio_processor import AudioProcessor
from app.whisper_handler import WhisperHandler

def main():
    # Step 1: Load configuration
    config = load_config()

    # Step 2: Build absolute path to harvard.wav (in project root)
    audio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harvard.wav')
    print(f"Audio file path: {audio_path}")  # For debugging

    # Step 3: Initialize audio processor and Whisper handler
    audio_processor = AudioProcessor(config)
    whisper = WhisperHandler(config)

    # Step 4: Load audio data using the audio processor
    audio_data = audio_processor.load_audio(audio_path)

    # Step 5: Transcribe audio to text
    recognized_text = whisper.transcribe(audio_data)

    # Step 6: Output the recognized text
    print("Transcribed text:", recognized_text)

if __name__ == '__main__':
    main()