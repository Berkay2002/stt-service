# app/utils/logger.py

"""
Enhanced logging system for the STT service.
Provides structured logging with different levels, request tracking, and error categorization.
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import functools

class StructuredLogger:
    """
    Structured logger with request tracking and context management.
    """
    
    def __init__(self, name: str = "STT_Service"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.setup_logger()
        
        # Request context tracking
        self._request_contexts = {}
    
    def setup_logger(self):
        """Setup structured logging with JSON formatter."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        json_formatter = JsonFormatter()
        console_formatter = ConsoleFormatter()
        
        # File handler for JSON logs
        file_handler = logging.FileHandler('stt_service.log', encoding='utf-8')
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for readable logs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _create_log_record(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a structured log record."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": "stt-service",
            "component": self.name,
            "message": message,
        }
        
        if extra:
            record.update(extra)
        
        return record
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        if extra:
            record = self._create_log_record("INFO", message, extra)
            # Use LogRecord for proper handling
            log_record = logging.LogRecord(
                name=self.logger.name,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=record,
                args=(),
                exc_info=None
            )
            self.logger.handle(log_record)
        else:
            self.logger.info(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        if extra:
            record = self._create_log_record("WARNING", message, extra)
            log_record = logging.LogRecord(
                name=self.logger.name,
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg=record,
                args=(),
                exc_info=None
            )
            self.logger.handle(log_record)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """Log error message."""
        if extra or exception:
            record = self._create_log_record("ERROR", message, extra)
            
            if exception:
                record["error_type"] = type(exception).__name__
                record["error_details"] = str(exception)
            
            log_record = logging.LogRecord(
                name=self.logger.name,
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg=record,
                args=(),
                exc_info=None
            )
            self.logger.handle(log_record)
        else:
            self.logger.error(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        if extra:
            record = self._create_log_record("DEBUG", message, extra)
            log_record = logging.LogRecord(
                name=self.logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=record,
                args=(),
                exc_info=None
            )
            self.logger.handle(log_record)
        else:
            self.logger.debug(message)
    
    @contextmanager
    def request_context(self, request_id: Optional[str] = None):
        """Context manager for tracking request-specific logs."""
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        self._request_contexts[request_id] = {
            "request_id": request_id,
            "start_time": start_time
        }
        
        self.info("Request started", {"request_id": request_id})
        
        try:
            yield request_id
        except Exception as e:
            self.error("Request failed", {
                "request_id": request_id,
                "duration_seconds": time.time() - start_time
            }, exception=e)
            raise
        finally:
            duration = time.time() - start_time
            self.info("Request completed", {
                "request_id": request_id,
                "duration_seconds": round(duration, 3)
            })
            
            # Clean up context
            self._request_contexts.pop(request_id, None)
    
    def log_transcription(self, request_id: str, audio_duration: float, processing_time: float, 
                         text_length: int, model_used: str, success: bool = True):
        """Log transcription-specific metrics."""
        extra = {
            "request_id": request_id,
            "event_type": "transcription",
            "audio_duration_seconds": audio_duration,
            "processing_time_seconds": processing_time,
            "real_time_factor": processing_time / max(audio_duration, 0.001),
            "text_length_chars": text_length,
            "model_used": model_used,
            "success": success
        }
        
        if success:
            self.info(f"Transcription completed successfully", extra)
        else:
            self.error(f"Transcription failed", extra)
    
    def log_performance(self, component: str, operation: str, duration: float, 
                       success: bool = True, extra: Optional[Dict[str, Any]] = None):
        """Log performance metrics for specific operations."""
        log_extra = {
            "event_type": "performance",
            "component": component,
            "operation": operation,
            "duration_seconds": duration,
            "success": success
        }
        
        if extra:
            log_extra.update(extra)
        
        message = f"{component}.{operation} completed in {duration:.3f}s"
        
        if success:
            self.info(message, log_extra)
        else:
            self.error(message, log_extra)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        if isinstance(record.msg, dict):
            # Already a structured record
            return json.dumps(record.msg)
        else:
            # Standard log message
            log_obj = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "service": "stt-service",
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(log_obj)

class ConsoleFormatter(logging.Formatter):
    """Console formatter for human-readable logs."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        if isinstance(record.msg, dict):
            # Extract human-readable message from structured log
            message = record.msg.get("message", str(record.msg))
            record = logging.makeLogRecord({
                'name': record.name,
                'msg': message,
                'args': record.args,
                'levelname': record.levelname,
                'levelno': record.levelno,
                'pathname': record.pathname,
                'filename': record.filename,
                'module': record.module,
                'lineno': record.lineno,
                'funcName': record.funcName,
                'created': record.created,
                'msecs': record.msecs,
                'relativeCreated': record.relativeCreated,
                'thread': record.thread,
                'threadName': record.threadName,
                'processName': record.processName,
                'process': record.process,
                'exc_info': record.exc_info,
                'exc_text': record.exc_text,
                'stack_info': record.stack_info
            })
            return super().format(record)
        else:
            # Standard formatting
            return super().format(record)

def log_execution_time(logger: StructuredLogger):
    """Decorator to log function execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            component = func.__module__.split('.')[-1]
            operation = func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance(component, operation, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_performance(component, operation, duration, success=False, 
                                     extra={"error": str(e)})
                raise
        
        return wrapper
    return decorator

# Global logger instance
stt_logger = StructuredLogger("STT_Service")

# Convenience functions
def get_logger(name: str = "STT_Service") -> StructuredLogger:
    """Get a logger instance."""
    return StructuredLogger(name)

def log_info(message: str, **kwargs):
    """Quick info logging."""
    stt_logger.info(message, kwargs if kwargs else None)

def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    """Quick error logging."""
    stt_logger.error(message, kwargs if kwargs else None, exception)

def log_warning(message: str, **kwargs):
    """Quick warning logging."""
    stt_logger.warning(message, kwargs if kwargs else None)