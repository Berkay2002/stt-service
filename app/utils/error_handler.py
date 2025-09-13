# app/utils/error_handler.py

"""
Centralized error handling middleware for the STT service.
Provides error codes, recovery mechanisms, and alerting.
"""

import traceback
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass
from functools import wraps
import json

from app.utils.logger import get_logger

T = TypeVar('T')

class ErrorCode(Enum):
    """Standardized error codes for the STT service."""
    
    # Model errors (1000-1099)
    MODEL_LOAD_FAILED = "STT_1001"
    MODEL_NOT_FOUND = "STT_1002"
    MODEL_INFERENCE_FAILED = "STT_1003"
    MODEL_TIMEOUT = "STT_1004"
    
    # Audio processing errors (1100-1199)
    AUDIO_LOAD_FAILED = "STT_1101"
    AUDIO_FORMAT_UNSUPPORTED = "STT_1102"
    AUDIO_TOO_LONG = "STT_1103"
    AUDIO_TOO_SHORT = "STT_1104"
    AUDIO_CORRUPTED = "STT_1105"
    
    # Network/Communication errors (1200-1299)
    WEBSOCKET_CONNECTION_FAILED = "STT_1201"
    WEBSOCKET_TIMEOUT = "STT_1202"
    HTTP_REQUEST_FAILED = "STT_1203"
    NETWORK_TIMEOUT = "STT_1204"
    
    # System resource errors (1300-1399)
    MEMORY_INSUFFICIENT = "STT_1301"
    DISK_SPACE_INSUFFICIENT = "STT_1302"
    CPU_OVERLOAD = "STT_1303"
    GPU_UNAVAILABLE = "STT_1304"
    
    # Configuration errors (1400-1499)
    CONFIG_INVALID = "STT_1401"
    CONFIG_MISSING = "STT_1402"
    ENV_VAR_MISSING = "STT_1403"
    
    # Service errors (1500-1599)
    SERVICE_UNAVAILABLE = "STT_1501"
    SERVICE_TIMEOUT = "STT_1502"
    RATE_LIMIT_EXCEEDED = "STT_1503"
    
    # Generic errors (1900-1999)
    UNKNOWN_ERROR = "STT_1901"
    VALIDATION_ERROR = "STT_1902"
    PERMISSION_DENIED = "STT_1903"

@dataclass
class STTError:
    """Structured error information."""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    retry_after: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.code.value,
            "message": self.message,
            "details": self.details or {},
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "context": self.context or {},
            "timestamp": time.time()
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string."""
        return json.dumps(self.to_dict())

class STTException(Exception):
    """Base exception class for STT service errors."""
    
    def __init__(self, error: STTError, original_exception: Optional[Exception] = None):
        self.error = error
        self.original_exception = original_exception
        super().__init__(error.message)

class ErrorHandler:
    """
    Centralized error handling system with recovery mechanisms and alerting.
    """
    
    def __init__(self, logger_name: str = "ErrorHandler"):
        self.logger = get_logger(logger_name)
        self.error_counts = {}
        self.recovery_strategies = {}
        self.alert_thresholds = {
            "error_rate_per_minute": 10,
            "consecutive_failures": 5,
            "critical_error_threshold": 3
        }
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        
        def retry_with_backoff(func, max_retries=3, base_delay=1.0):
            """Retry with exponential backoff."""
            for attempt in range(max_retries):
                try:
                    return func()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        def fallback_to_cpu(func):
            """Fallback to CPU if GPU fails."""
            try:
                return func()
            except Exception:
                # Switch to CPU mode and retry
                return func(use_gpu=False)
        
        self.recovery_strategies = {
            ErrorCode.MODEL_INFERENCE_FAILED: retry_with_backoff,
            ErrorCode.GPU_UNAVAILABLE: fallback_to_cpu,
            ErrorCode.NETWORK_TIMEOUT: retry_with_backoff,
            ErrorCode.WEBSOCKET_TIMEOUT: retry_with_backoff
        }
    
    def handle_error(self, error: Union[STTError, Exception], context: Optional[Dict[str, Any]] = None) -> STTError:
        """
        Handle an error with logging, recovery, and alerting.
        
        Args:
            error: STTError instance or raw exception
            context: Additional context information
            
        Returns:
            STTError instance
        """
        # Convert raw exception to STTError if needed
        if isinstance(error, Exception) and not isinstance(error, STTException):
            stt_error = self._classify_exception(error, context)
        elif isinstance(error, STTException):
            stt_error = error.error
        else:
            stt_error = error
        
        # Add context
        if context:
            stt_error.context = {**(stt_error.context or {}), **context}
        
        # Log the error
        self._log_error(stt_error, error if isinstance(error, Exception) else None)
        
        # Update error statistics
        self._update_error_stats(stt_error)
        
        # Check for alerting conditions
        self._check_alert_conditions(stt_error)
        
        return stt_error
    
    def _classify_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> STTError:
        """Classify a raw exception into a structured STT error."""
        exc_type = type(exception).__name__
        exc_message = str(exception)
        
        # Classification logic based on exception type and message
        if "CUDA" in exc_message or "GPU" in exc_message:
            return STTError(
                code=ErrorCode.GPU_UNAVAILABLE,
                message=f"GPU error: {exc_message}",
                details={"exception_type": exc_type},
                recoverable=True,
                retry_after=5.0
            )
        
        if "timeout" in exc_message.lower():
            return STTError(
                code=ErrorCode.SERVICE_TIMEOUT,
                message=f"Timeout error: {exc_message}",
                details={"exception_type": exc_type},
                recoverable=True,
                retry_after=2.0
            )
        
        if "memory" in exc_message.lower() or "OutOfMemoryError" in exc_type:
            return STTError(
                code=ErrorCode.MEMORY_INSUFFICIENT,
                message=f"Memory error: {exc_message}",
                details={"exception_type": exc_type},
                recoverable=False
            )
        
        if "FileNotFoundError" in exc_type:
            return STTError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message=f"File not found: {exc_message}",
                details={"exception_type": exc_type},
                recoverable=False
            )
        
        if "ConnectionError" in exc_type or "socket" in exc_message.lower():
            return STTError(
                code=ErrorCode.WEBSOCKET_CONNECTION_FAILED,
                message=f"Connection error: {exc_message}",
                details={"exception_type": exc_type},
                recoverable=True,
                retry_after=3.0
            )
        
        # Default classification
        return STTError(
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unknown error: {exc_message}",
            details={"exception_type": exc_type},
            recoverable=True
        )
    
    def _log_error(self, error: STTError, exception: Optional[Exception] = None):
        """Log the error with appropriate level and context."""
        log_extra = {
            "error_code": error.code.value,
            "recoverable": error.recoverable,
            "retry_after": error.retry_after,
            "error_details": error.details,
            "error_context": error.context
        }
        
        # Determine log level based on error severity
        if not error.recoverable or error.code in [ErrorCode.MEMORY_INSUFFICIENT, ErrorCode.MODEL_LOAD_FAILED]:
            log_level = "critical"
        elif error.code.value.startswith("STT_13"):  # System resource errors
            log_level = "error"
        else:
            log_level = "warning"
        
        if log_level == "critical":
            self.logger.error(f"CRITICAL: {error.message}", log_extra, exception)
        elif log_level == "error":
            self.logger.error(error.message, log_extra, exception)
        else:
            self.logger.warning(error.message, log_extra)
    
    def _update_error_stats(self, error: STTError):
        """Update error statistics for monitoring and alerting."""
        error_key = error.code.value
        current_time = time.time()
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = {
                "total": 0,
                "recent": [],
                "consecutive": 0,
                "last_occurrence": None
            }
        
        stats = self.error_counts[error_key]
        stats["total"] += 1
        stats["recent"].append(current_time)
        stats["last_occurrence"] = current_time
        
        # Clean up old entries (keep only last minute)
        stats["recent"] = [t for t in stats["recent"] if current_time - t <= 60]
        
        # Update consecutive count
        if hasattr(self, '_last_error_code') and self._last_error_code == error_key:
            stats["consecutive"] += 1
        else:
            stats["consecutive"] = 1
        
        self._last_error_code = error_key
    
    def _check_alert_conditions(self, error: STTError):
        """Check if error conditions warrant alerting."""
        error_key = error.code.value
        stats = self.error_counts.get(error_key, {})
        
        # Check error rate
        recent_count = len(stats.get("recent", []))
        if recent_count >= self.alert_thresholds["error_rate_per_minute"]:
            self._send_alert("high_error_rate", {
                "error_code": error_key,
                "count_per_minute": recent_count,
                "threshold": self.alert_thresholds["error_rate_per_minute"]
            })
        
        # Check consecutive failures
        consecutive = stats.get("consecutive", 0)
        if consecutive >= self.alert_thresholds["consecutive_failures"]:
            self._send_alert("consecutive_failures", {
                "error_code": error_key,
                "consecutive_count": consecutive,
                "threshold": self.alert_thresholds["consecutive_failures"]
            })
        
        # Check for critical errors
        if not error.recoverable:
            self._send_alert("critical_error", {
                "error_code": error_key,
                "message": error.message,
                "details": error.details
            })
    
    def _send_alert(self, alert_type: str, details: Dict[str, Any]):
        """Send alert (placeholder for actual alerting mechanism)."""
        alert_data = {
            "alert_type": alert_type,
            "timestamp": time.time(),
            "service": "stt-service",
            "details": details
        }
        
        # Log alert (in production, this would send to monitoring system)
        self.logger.error(f"ALERT: {alert_type}", {"alert_data": alert_data})
        
        # Here you would integrate with your alerting system:
        # - Send to Slack/Teams
        # - Send to PagerDuty
        # - Send email notification
        # - Push to monitoring dashboard
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics."""
        current_time = time.time()
        stats_summary = {}
        
        for error_code, stats in self.error_counts.items():
            recent_count = len([t for t in stats["recent"] if current_time - t <= 60])
            stats_summary[error_code] = {
                "total_count": stats["total"],
                "recent_count_per_minute": recent_count,
                "consecutive_failures": stats["consecutive"],
                "last_occurrence": stats["last_occurrence"]
            }
        
        return {
            "error_statistics": stats_summary,
            "alert_thresholds": self.alert_thresholds,
            "total_unique_errors": len(self.error_counts)
        }

def error_handler_decorator(error_handler: ErrorHandler, context: Optional[Dict[str, Any]] = None):
    """Decorator to automatically handle errors in functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add function context
                func_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                if context:
                    func_context.update(context)
                
                # Handle the error
                stt_error = error_handler.handle_error(e, func_context)
                
                # Re-raise as STTException
                raise STTException(stt_error, e)
        
        return wrapper
    return decorator

# Global error handler instance
global_error_handler = ErrorHandler("STT_ErrorHandler")