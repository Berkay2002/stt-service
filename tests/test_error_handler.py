# tests/test_error_handler.py

"""
Test script for the error handling system.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.error_handler import (
    ErrorHandler, ErrorCode, STTError, STTException, 
    error_handler_decorator, global_error_handler
)

def test_error_classification():
    """Test error classification from raw exceptions."""
    print("🧪 Testing Error Classification")
    print("=" * 40)
    
    handler = ErrorHandler("Test_Handler")
    
    # Test GPU error
    gpu_error = RuntimeError("CUDA out of memory")
    classified = handler._classify_exception(gpu_error)
    print(f"✅ GPU Error: {classified.code.value} - {classified.message}")
    
    # Test timeout error
    timeout_error = TimeoutError("Request timeout after 30 seconds")
    classified = handler._classify_exception(timeout_error)
    print(f"✅ Timeout Error: {classified.code.value} - {classified.message}")
    
    # Test file not found
    file_error = FileNotFoundError("Model file not found")
    classified = handler._classify_exception(file_error)
    print(f"✅ File Error: {classified.code.value} - {classified.message}")
    
    # Test unknown error
    unknown_error = ValueError("Unknown validation issue")
    classified = handler._classify_exception(unknown_error)
    print(f"✅ Unknown Error: {classified.code.value} - {classified.message}")

def test_structured_errors():
    """Test creating and handling structured STT errors."""
    print("\\n🧪 Testing Structured Errors")
    print("=" * 40)
    
    handler = ErrorHandler("Test_Handler")
    
    # Create a structured error
    error = STTError(
        code=ErrorCode.MODEL_INFERENCE_FAILED,
        message="Model inference failed for audio chunk",
        details={"audio_duration": 5.0, "model": "whisper-base"},
        recoverable=True,
        retry_after=2.0,
        context={"request_id": "test-123", "user_id": "user-456"}
    )
    
    # Handle the error
    handled_error = handler.handle_error(error)
    
    print(f"✅ Structured Error: {handled_error.code.value}")
    print(f"   Message: {handled_error.message}")
    print(f"   Recoverable: {handled_error.recoverable}")
    print(f"   Retry After: {handled_error.retry_after}s")
    
    # Test JSON serialization
    json_data = handled_error.to_json()
    print(f"✅ JSON Serialization: {len(json_data)} characters")

def test_error_statistics():
    """Test error statistics and alerting."""
    print("\\n🧪 Testing Error Statistics & Alerting")
    print("=" * 40)
    
    handler = ErrorHandler("Test_Handler")
    
    # Simulate multiple errors
    for i in range(3):
        error = STTError(
            code=ErrorCode.AUDIO_LOAD_FAILED,
            message=f"Audio load failed attempt {i+1}",
            recoverable=True
        )
        handler.handle_error(error)
        time.sleep(0.1)  # Small delay
    
    # Simulate consecutive failures
    for i in range(6):
        error = STTError(
            code=ErrorCode.MODEL_INFERENCE_FAILED,
            message=f"Consecutive failure {i+1}",
            recoverable=True
        )
        handler.handle_error(error)
    
    # Get statistics
    stats = handler.get_error_statistics()
    print(f"✅ Error Statistics:")
    print(f"   Total unique errors: {stats['total_unique_errors']}")
    for error_code, stat in stats['error_statistics'].items():
        print(f"   {error_code}: {stat['total_count']} total, {stat['consecutive_failures']} consecutive")

@error_handler_decorator(global_error_handler, {"component": "test_function"})
def function_that_fails():
    """Test function that raises an exception."""
    raise RuntimeError("GPU memory allocation failed")

@error_handler_decorator(global_error_handler, {"component": "test_function"})
def function_that_succeeds():
    """Test function that succeeds."""
    return "Success!"

def test_decorator():
    """Test the error handler decorator."""
    print("\\n🧪 Testing Error Handler Decorator")
    print("=" * 40)
    
    # Test successful function
    try:
        result = function_that_succeeds()
        print(f"✅ Successful function: {result}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test failing function
    try:
        function_that_fails()
        print("❌ Function should have failed")
    except STTException as e:
        print(f"✅ Caught STTException: {e.error.code.value}")
        print(f"   Message: {e.error.message}")
        print(f"   Recoverable: {e.error.recoverable}")
    except Exception as e:
        print(f"❌ Unexpected exception type: {e}")

def test_recovery_strategies():
    """Test error recovery strategies (placeholder)."""
    print("\\n🧪 Testing Recovery Strategies")
    print("=" * 40)
    
    handler = ErrorHandler("Test_Handler")
    
    # Check if recovery strategies are registered
    strategies = handler.recovery_strategies
    print(f"✅ Recovery strategies registered: {len(strategies)}")
    
    for error_code, strategy in strategies.items():
        print(f"   {error_code.value}: {strategy.__name__}")

def simulate_critical_error():
    """Simulate a critical error for testing alerting."""
    print("\\n🧪 Testing Critical Error Alerting")
    print("=" * 40)
    
    handler = ErrorHandler("Test_Handler")
    
    # Create a critical error
    critical_error = STTError(
        code=ErrorCode.MEMORY_INSUFFICIENT,
        message="System running out of memory - service may become unstable",
        details={"available_memory_gb": 0.5, "required_memory_gb": 4.0},
        recoverable=False
    )
    
    handler.handle_error(critical_error)
    print("✅ Critical error handled and logged")

def show_error_codes():
    """Show all available error codes."""
    print("\\n📋 Available Error Codes:")
    print("=" * 40)
    
    categories = {
        "Model": [code for code in ErrorCode if code.value.startswith("STT_10")],
        "Audio": [code for code in ErrorCode if code.value.startswith("STT_11")],
        "Network": [code for code in ErrorCode if code.value.startswith("STT_12")],
        "System": [code for code in ErrorCode if code.value.startswith("STT_13")],
        "Config": [code for code in ErrorCode if code.value.startswith("STT_14")],
        "Service": [code for code in ErrorCode if code.value.startswith("STT_15")],
        "Generic": [code for code in ErrorCode if code.value.startswith("STT_19")]
    }
    
    for category, codes in categories.items():
        print(f"\\n{category} Errors:")
        for code in codes:
            print(f"  {code.value}: {code.name}")

def main():
    """Run all error handling tests."""
    print("🧪 STT Service Error Handling Tests")
    print("=" * 50)
    
    test_error_classification()
    test_structured_errors()
    test_error_statistics()
    test_decorator()
    test_recovery_strategies()
    simulate_critical_error()
    show_error_codes()
    
    print("\\n" + "=" * 50)
    print("🎉 All error handling tests completed!")
    print("📊 Check logs for error handling details")
    print("🚨 Alerts would be sent to monitoring system in production")

if __name__ == "__main__":
    main()