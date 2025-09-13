# tests/test_realtime.py

"""
Test script for real-time transcription system.
This script helps debug and verify the real-time transcription setup.
"""

import sys
import time
import numpy as np
import pyaudio
from app.utils.config import load_config

def test_audio_devices():
    """Test and list available audio devices."""
    print("🎤 Testing Audio Devices")
    print("=" * 50)
    
    p = pyaudio.PyAudio()
    
    print("Available audio devices:")
    device_count = p.get_device_count()
    input_devices = []
    
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        if int(info['maxInputChannels']) > 0:
            input_devices.append((i, info))
            status = "✅" if info['defaultSampleRate'] == 16000.0 else "⚠️"
            print(f"  {status} Device {i}: {info['name']}")
            print(f"      Channels: {info['maxInputChannels']}, Sample Rate: {info['defaultSampleRate']}")
    
    if not input_devices:
        print("❌ No input devices found!")
        p.terminate()
        return False
    
    p.terminate()
    print(f"✅ Found {len(input_devices)} input device(s)")
    return True

def test_vad_import():
    """Test Voice Activity Detection import."""
    print("\\n🔍 Testing Voice Activity Detection")
    print("=" * 50)
    
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        print("✅ WebRTC VAD is available")
        
        # Test VAD with proper dummy data
        sample_rate = 16000
        frame_duration = 30  # ms - WebRTC VAD supports 10, 20, or 30ms frames
        frame_size = int(sample_rate * frame_duration / 1000)  # Number of samples
        
        # Create proper silence frame (int16 format, not bytes)
        # WebRTC VAD expects 16-bit PCM audio data
        dummy_audio = np.zeros(frame_size, dtype=np.int16)
        dummy_frame = dummy_audio.tobytes()
        
        result = vad.is_speech(dummy_frame, sample_rate)
        print(f"✅ VAD test successful (silence detected: {not result})")
        
        # Test with some "fake speech" (random noise)
        noise_audio = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
        noise_frame = noise_audio.tobytes()
        noise_result = vad.is_speech(noise_frame, sample_rate)
        print(f"✅ VAD noise test: {'speech detected' if noise_result else 'no speech detected'}")
        
        return True
        
    except ImportError:
        print("⚠️  WebRTC VAD not available - will use volume-based detection")
        print("   Install with: pip install webrtcvad")
        return False
    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False

    """Test Voice Activity Detection import."""
    print("\\n🔍 Testing Voice Activity Detection")
    print("=" * 50)
    
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        print("✅ WebRTC VAD is available")
        
        # Test VAD with dummy data
        sample_rate = 16000
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        dummy_frame = b'\\x00\\x00' * frame_size
        
        result = vad.is_speech(dummy_frame, sample_rate)
        print(f"✅ VAD test successful (silence detected: {not result})")
        return True
        
    except ImportError:
        print("⚠️  WebRTC VAD not available - will use volume-based detection")
        print("   Install with: pip install webrtcvad")
        return False
    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False

def test_whisper_loading():
    """Test Whisper model loading."""
    print("\\n🤖 Testing Whisper Model Loading")
    print("=" * 50)
    
    try:
        config = load_config()
        print(f"Config: {config}")
        
        from app.core.whisper_handler import WhisperHandler
        whisper = WhisperHandler(config)
        
        # Test with dummy audio
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        text = whisper.transcribe(dummy_audio)
        
        print("✅ Whisper model loaded successfully")
        print(f"✅ Transcription test: '{text}' (should be empty or minimal)")
        return True
        
    except Exception as e:
        print(f"❌ Whisper loading failed: {e}")
        return False

def test_microphone_recording():
    """Test microphone recording capability."""
    print("\\n🎙️  Testing Microphone Recording")
    print("=" * 50)
    
    try:
        p = pyaudio.PyAudio()
        
        # Try to open a stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Recording 2 seconds of audio to test microphone...")
        frames = []
        
        for _ in range(0, int(16000 / 1024 * 2)):  # 2 seconds
            data = stream.read(1024, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Analyze the recorded audio
        combined_audio = np.concatenate(frames)
        max_amplitude = np.max(np.abs(combined_audio))
        rms_level = np.sqrt(np.mean(combined_audio ** 2))
        
        print(f"✅ Microphone recording successful")
        print(f"   Max amplitude: {max_amplitude:.6f}")
        print(f"   RMS level: {rms_level:.6f}")
        
        if max_amplitude < 0.001:
            print("⚠️  Audio level is very low - check microphone volume")
            return False
        
        print("✅ Audio levels look good")
        return True
        
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False

def test_threading():
    """Test threading functionality."""
    print("\\n🧵 Testing Threading")
    print("=" * 50)
    
    try:
        import threading
        import queue
        
        test_queue = queue.Queue()
        results = []
        
        def test_thread(name, delay):
            time.sleep(delay)
            test_queue.put(f"Thread {name} completed")
            results.append(name)
        
        # Start test threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=test_thread, args=(i, 0.1 * i))
            thread.start()
            threads.append(thread)
        
        # Wait for threads
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Check results
        while not test_queue.empty():
            print(f"✅ {test_queue.get()}")
        
        if len(results) == 3:
            print("✅ Threading test successful")
            return True
        else:
            print(f"❌ Only {len(results)}/3 threads completed")
            return False
            
    except Exception as e:
        print(f"❌ Threading test failed: {e}")
        return False

def run_quick_realtime_test():
    """Run a quick real-time transcription test."""
    print("\\n🚀 Quick Real-Time Test")
    print("=" * 50)
    
    try:
        from app.core.realtime_transcription import RealTimeTranscriber
        config = load_config()
        
        print("Creating transcriber (this may take a moment to load the model)...")
        transcriber = RealTimeTranscriber(
            config=config,
            chunk_duration=0.5,
            buffer_duration=10.0
        )
        
        print("✅ Real-time transcriber created successfully")
        print("\\n🎤 Starting 10-second real-time test...")
        print("Speak into your microphone now - transcriptions will appear below:")
        print("-" * 50)
        
        transcriber.start()
        
        # Run for 10 seconds
        start_time = time.time()
        try:
            while time.time() - start_time < 10.0:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        transcriber.stop()
        print("-" * 50)
        print("✅ Real-time test completed")
        return True
        
    except Exception as e:
        print(f"❌ Real-time test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 REAL-TIME TRANSCRIPTION SYSTEM TEST")
    print("=" * 60)
    print("This will test all components of your real-time transcription system.")
    print("=" * 60)
    
    tests = [
        ("Audio Devices", test_audio_devices),
        ("Voice Activity Detection", test_vad_import),
        ("Whisper Model", test_whisper_loading),
        ("Microphone Recording", test_microphone_recording),
        ("Threading", test_threading),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\\n📊 TEST RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your system is ready for real-time transcription.")
        
        response = input("\\nRun a 10-second real-time transcription test? (y/n): ")
        if response.lower().startswith('y'):
            run_quick_realtime_test()
    else:
        print("⚠️  Some tests failed. Check the output above for issues to resolve.")
        print("\\nCommon solutions:")
        print("- Install webrtcvad: pip install webrtcvad")
        print("- Check microphone permissions")
        print("- Ensure microphone is not in use by other applications")
    
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    main()