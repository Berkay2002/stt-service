# app/core/error_handler.py

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import traceback

from app.utils.logger import get_logger


class ErrorCategory(Enum):
    """Categories of errors that can occur in the STT service"""
    CONNECTION = "connection"
    AUDIO_FORMAT = "audio_format"
    PROCESSING = "processing"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    GPU = "gpu"
    TIMEOUT = "timeout"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Information about an error occurrence"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    session_id: Optional[str] = None
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class CircuitBreaker:
    """Circuit breaker pattern for protecting against cascading failures"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        # State tracking
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # "closed", "open", "half-open"

        self.logger = get_logger('CircuitBreaker')

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            # Check if we should transition to half-open
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception("Circuit breaker is open - service temporarily unavailable")

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Success - reset failure count if in half-open state
            if self.state == "half-open":
                self.failure_count = 0
                self.state = "closed"
                self.logger.info("Circuit breaker closed - service restored")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.error(f"Circuit breaker opened due to {self.failure_count} failures")

            raise

class RetryManager:
    """Manages retry logic with exponential backoff"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def retry_with_backoff(self, func: Callable, error_info: ErrorInfo, *args, **kwargs):
        """Retry function with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                return result

            except Exception as e:
                error_info.retry_count = attempt + 1

                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    raise

                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                await asyncio.sleep(delay)

        raise Exception(f"Max retries ({self.max_retries}) exceeded")

class WebSocketErrorHandler:
    """Comprehensive error handling for WebSocket STT operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger('ErrorHandler')

        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts_by_category: Dict[ErrorCategory, int] = {}
        self.error_counts_by_session: Dict[str, int] = {}

        # Circuit breakers for different operations
        self.transcription_breaker = CircuitBreaker(
            failure_threshold=config.get("transcription_failure_threshold", 5),
            timeout=config.get("transcription_timeout", 120.0)
        )

        self.gpu_breaker = CircuitBreaker(
            failure_threshold=config.get("gpu_failure_threshold", 3),
            timeout=config.get("gpu_timeout", 300.0)
        )

        # Retry managers
        self.transcription_retry = RetryManager(
            max_retries=config.get("transcription_max_retries", 2),
            base_delay=config.get("transcription_retry_delay", 1.0)
        )

        self.connection_retry = RetryManager(
            max_retries=config.get("connection_max_retries", 3),
            base_delay=config.get("connection_retry_delay", 0.5)
        )

    def categorize_error(self, exception: Exception, context: str = "") -> ErrorInfo:
        """Categorize an error and determine appropriate handling"""
        error_type = type(exception).__name__
        error_msg = str(exception)

        # Determine category based on error type and message
        if "websocket" in error_msg.lower() or "connection" in error_msg.lower():
            category = ErrorCategory.CONNECTION
            severity = ErrorSeverity.MEDIUM
            recoverable = True
        elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower() or "memory" in error_msg.lower():
            category = ErrorCategory.GPU
            severity = ErrorSeverity.HIGH
            recoverable = True
        elif "audio" in error_msg.lower() or "format" in error_msg.lower():
            category = ErrorCategory.AUDIO_FORMAT
            severity = ErrorSeverity.LOW
            recoverable = True
        elif "timeout" in error_msg.lower():
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
            recoverable = True
        elif error_type in ["RuntimeError", "ValueError"]:
            category = ErrorCategory.PROCESSING
            severity = ErrorSeverity.MEDIUM
            recoverable = True
        elif error_type in ["OSError", "MemoryError"]:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.HIGH
            recoverable = False
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.MEDIUM
            recoverable = True

        # Generate error code
        error_code = f"{category.value}_{error_type.lower()}"

        return ErrorInfo(
            category=category,
            severity=severity,
            message=error_msg,
            code=error_code,
            details={
                "exception_type": error_type,
                "context": context,
                "traceback": traceback.format_exc()
            },
            recoverable=recoverable
        )

    async def handle_error(self, exception: Exception, context: str = "", session_id: Optional[str] = None) -> ErrorInfo:
        """Handle an error with appropriate response strategy"""
        error_info = self.categorize_error(exception, context)
        error_info.session_id = session_id

        # Track error statistics
        self._track_error(error_info)

        # Log error
        self.logger.error(
            f"Handling error in {context}",
            extra={
                "error_category": error_info.category.value,
                "error_severity": error_info.severity.value,
                "error_code": error_info.code,
                "session_id": session_id,
                "recoverable": error_info.recoverable
            },
            exception=exception
        )

        # Apply specific handling strategies
        await self._apply_error_strategy(error_info)

        return error_info

    async def _apply_error_strategy(self, error_info: ErrorInfo) -> None:
        """Apply specific error handling strategies based on error category"""

        if error_info.category == ErrorCategory.GPU:
            # GPU errors - aggressive circuit breaking
            if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.logger.warning("GPU error detected - activating GPU circuit breaker")
                # Force circuit breaker to open temporarily
                self.gpu_breaker.failure_count = self.gpu_breaker.failure_threshold
                self.gpu_breaker.last_failure_time = time.time()
                self.gpu_breaker.state = "open"

        elif error_info.category == ErrorCategory.CONNECTION:
            # Connection errors - typically recoverable
            self.logger.info("Connection error - client should retry with backoff")

        elif error_info.category == ErrorCategory.RESOURCE:
            # Resource exhaustion - temporary backoff
            if error_info.severity == ErrorSeverity.HIGH:
                self.logger.warning("Resource exhaustion - implementing temporary cooldown")
                await asyncio.sleep(5.0)  # Brief cooldown

        elif error_info.category == ErrorCategory.AUDIO_FORMAT:
            # Audio format errors - usually client-side issue
            self.logger.info("Audio format error - sending format guidance to client")

        elif error_info.category == ErrorCategory.TIMEOUT:
            # Timeout errors - may need to adjust processing parameters
            self.logger.warning("Timeout detected - consider reducing processing complexity")

    def _track_error(self, error_info: ErrorInfo) -> None:
        """Track error for statistics and pattern detection"""
        # Add to history (keep last 1000 errors)
        self.error_history.append(error_info)
        if len(self.error_history) > 1000:
            self.error_history.pop(0)

        # Update category counts
        self.error_counts_by_category[error_info.category] = (
            self.error_counts_by_category.get(error_info.category, 0) + 1
        )

        # Update session counts
        if error_info.session_id:
            self.error_counts_by_session[error_info.session_id] = (
                self.error_counts_by_session.get(error_info.session_id, 0) + 1
            )

    async def should_reject_connection(self, session_id: Optional[str] = None) -> bool:
        """Determine if new connections should be rejected due to error conditions"""
        # Check circuit breakers
        if self.transcription_breaker.state == "open" or self.gpu_breaker.state == "open":
            return True

        # Check session-specific error rate
        if session_id and session_id in self.error_counts_by_session:
            session_errors = self.error_counts_by_session[session_id]
            if session_errors > self.config.get("max_session_errors", 10):
                return True

        # Check overall error rate
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
        if len(recent_errors) > self.config.get("max_recent_errors", 50):
            return True

        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour

        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "errors_by_category": dict(self.error_counts_by_category),
            "errors_by_severity": {
                severity.value: sum(1 for e in recent_errors if e.severity == severity)
                for severity in ErrorSeverity
            },
            "circuit_breaker_states": {
                "transcription": self.transcription_breaker.state,
                "gpu": self.gpu_breaker.state
            },
            "top_error_sessions": sorted(
                self.error_counts_by_session.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def reset_error_tracking(self) -> None:
        """Reset error tracking (useful for testing or after maintenance)"""
        self.error_history.clear()
        self.error_counts_by_category.clear()
        self.error_counts_by_session.clear()

        # Reset circuit breakers
        self.transcription_breaker.failure_count = 0
        self.transcription_breaker.state = "closed"
        self.gpu_breaker.failure_count = 0
        self.gpu_breaker.state = "closed"

        self.logger.info("Error tracking reset")

def create_error_response(error_info: ErrorInfo) -> Dict[str, Any]:
    """Create a standardized error response for WebSocket clients"""
    return {
        "type": "error",
        "error": {
            "code": error_info.code,
            "message": error_info.message,
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "recoverable": error_info.recoverable,
            "timestamp": error_info.timestamp,
            "retry_after": _calculate_retry_delay(error_info),
            "details": error_info.details or {}
        }
    }

def _calculate_retry_delay(error_info: ErrorInfo) -> Optional[float]:
    """Calculate suggested retry delay based on error type"""
    if not error_info.recoverable:
        return None

    if error_info.category == ErrorCategory.GPU and error_info.severity == ErrorSeverity.HIGH:
        return 60.0  # 1 minute for GPU issues
    elif error_info.category == ErrorCategory.RESOURCE:
        return 30.0  # 30 seconds for resource issues
    elif error_info.category == ErrorCategory.CONNECTION:
        return 5.0   # 5 seconds for connection issues
    else:
        return 1.0   # 1 second for other recoverable errors