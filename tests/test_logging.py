# tests/test_logging.py

"""
Test script for the enhanced logging system.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import get_logger, log_execution_time, log_info, log_error, log_warning

def test_basic_logging():
    """Test basic logging functionality."""
    logger = get_logger("Test_Logger")
    
    print("🧪 Testing Basic Logging")
    print("=" * 40)
    
    # Test different log levels
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    # Test structured logging
    logger.info("Structured log test", {
        "user_id": "user123",
        "action": "test_action",
        "timestamp": time.time()
    })
    
    print("✅ Basic logging test completed")

def test_request_context():
    """Test request context logging."""
    logger = get_logger("Request_Logger")
    
    print("\\n🧪 Testing Request Context Logging")
    print("=" * 40)
    
    with logger.request_context() as request_id:
        logger.info("Processing request", {"step": "validation"})
        time.sleep(0.1)  # Simulate processing
        logger.info("Request processing", {"step": "execution"})
        time.sleep(0.1)  # Simulate more processing
        logger.info("Request finalizing", {"step": "response"})
    
    print("✅ Request context test completed")

def test_transcription_logging():
    """Test transcription-specific logging."""
    logger = get_logger("Transcription_Logger")
    
    print("\\n🧪 Testing Transcription Logging")
    print("=" * 40)
    
    with logger.request_context() as request_id:
        # Simulate successful transcription
        logger.log_transcription(
            request_id=request_id,
            audio_duration=5.0,
            processing_time=0.8,
            text_length=120,
            model_used="whisper-base.en",
            success=True
        )
        
        # Simulate failed transcription
        logger.log_transcription(
            request_id=request_id,
            audio_duration=3.0,
            processing_time=0.5,
            text_length=0,
            model_used="whisper-base.en",
            success=False
        )
    
    print("✅ Transcription logging test completed")

@log_execution_time(get_logger("Performance_Logger"))
def slow_function():
    """Simulate a slow function for performance logging."""
    time.sleep(0.2)
    return "Function completed"

def test_performance_logging():
    """Test performance logging."""
    print("\\n🧪 Testing Performance Logging")
    print("=" * 40)
    
    # Test decorator
    result = slow_function()
    
    # Test manual performance logging
    logger = get_logger("Performance_Logger")
    start_time = time.time()
    time.sleep(0.1)
    duration = time.time() - start_time
    
    logger.log_performance(
        component="audio_processor",
        operation="load_audio",
        duration=duration,
        success=True,
        extra={"file_size_mb": 2.5, "sample_rate": 16000}
    )
    
    print("✅ Performance logging test completed")

def test_error_logging():
    """Test error logging with exceptions."""
    logger = get_logger("Error_Logger")
    
    print("\\n🧪 Testing Error Logging")
    print("=" * 40)
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        logger.error("Division by zero error occurred", 
                    extra={"operation": "mathematical_calculation"},
                    exception=e)
    
    # Test convenience function
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        log_error("Test error using convenience function", exception=e, 
                 context="test_function", user_id="test_user")
    
    print("✅ Error logging test completed")

def test_convenience_functions():
    """Test convenience logging functions."""
    print("\\n🧪 Testing Convenience Functions")
    print("=" * 40)
    
    log_info("This is a convenience info message", module="test", action="demo")
    log_warning("This is a convenience warning", severity="medium")
    
    print("✅ Convenience functions test completed")

def show_log_file_sample():
    """Show a sample of the log file."""
    print("\\n📁 Log File Sample (last 10 lines):")
    print("=" * 40)
    
    try:
        with open("stt_service.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())
    except FileNotFoundError:
        print("Log file not found")
    except Exception as e:
        print(f"Error reading log file: {e}")

def main():
    """Run all logging tests."""
    print("🧪 STT Service Enhanced Logging Tests")
    print("=" * 50)
    
    test_basic_logging()
    test_request_context()
    test_transcription_logging()
    test_performance_logging()
    test_error_logging()
    test_convenience_functions()
    
    print("\\n" + "=" * 50)
    print("🎉 All logging tests completed!")
    print("📝 Check 'stt_service.log' for JSON-formatted logs")
    print("👀 Console output shows human-readable format")
    
    show_log_file_sample()

if __name__ == "__main__":
    main()