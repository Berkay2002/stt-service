# app/monitoring/service_monitor.py

import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request
from app.utils.config import load_config
from app.core.whisper_handler import WhisperHandler
from app.utils.logger import get_logger

class ServiceMonitor:
    """
    Monitoring and health check system for the STT service.
    Provides health checks, metrics collection, and status endpoints.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.start_time = time.time()
        self.request_count = 0
        self.transcription_count = 0
        self.error_count = 0
        self.total_transcription_time = 0.0
        self.total_audio_duration = 0.0
        
        # Health status
        self.is_healthy = True
        self.health_details = {}
        self.last_health_check = None
        
        # Model status
        self.model_loaded = False
        self.model_load_time = None
        self.whisper_handler = None
        
        # Setup logging using enhanced logger
        self.logger = get_logger('STT_Monitor')
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Whisper model and check if it loads successfully."""
        try:
            start_time = time.time()
            self.whisper_handler = WhisperHandler(self.config)
            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            self.logger.info(f"Model loaded successfully in {self.model_load_time:.2f}s")
        except Exception as e:
            self.model_loaded = False
            self.is_healthy = False
            self.logger.error(f"Failed to load model: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the STT service.
        Returns detailed health status including model, system resources, etc.
        """
        self.last_health_check = datetime.now().isoformat()
        
        # System resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Model health
        model_health = {
            "loaded": self.model_loaded,
            "load_time_seconds": self.model_load_time,
            "backend": self.config.get('whisper_backend', 'unknown')
        }
        
        # Service metrics
        uptime = time.time() - self.start_time
        transcription_count = max(self.transcription_count, 1)
        avg_transcription_time = self.total_transcription_time / transcription_count
        
        health_status = {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "timestamp": self.last_health_check,
            "uptime_seconds": round(uptime, 2),
            "model": model_health,
            "system": {
                "cpu_percent": float(cpu_percent),
                "memory_percent": float(memory_info.percent),
                "memory_available_gb": round(float(memory_info.available) / (1024**3), 2),
                "memory_total_gb": round(float(memory_info.total) / (1024**3), 2)
            },
            "metrics": {
                "total_requests": self.request_count,
                "total_transcriptions": self.transcription_count,
                "total_errors": self.error_count,
                "avg_transcription_time_seconds": round(avg_transcription_time, 3),
                "total_audio_processed_seconds": round(self.total_audio_duration, 2)
            }
        }
        
        # Additional health checks
        if float(cpu_percent) > 90:
            health_status["warnings"] = health_status.get("warnings", [])
            health_status["warnings"].append("High CPU usage")
        
        if float(memory_info.percent) > 90:
            health_status["warnings"] = health_status.get("warnings", [])
            health_status["warnings"].append("High memory usage")
        
        if not self.model_loaded:
            self.is_healthy = False
            health_status["status"] = "unhealthy"
            health_status["errors"] = ["Model not loaded"]
        
        self.health_details = health_status
        return health_status
    
    def record_transcription(self, duration: float, processing_time: float):
        """Record metrics for a transcription operation."""
        self.transcription_count += 1
        self.total_transcription_time += processing_time
        self.total_audio_duration += duration
        
        # Log performance metrics
        real_time_factor = processing_time / max(duration, 0.001)
        self.logger.info(
            f"Transcription completed - Duration: {duration:.2f}s, "
            f"Processing: {processing_time:.2f}s, RTF: {real_time_factor:.2f}"
        )
    
    def record_request(self):
        """Record an incoming request."""
        self.request_count += 1
    
    def record_error(self, error: Exception, context: str = ""):
        """Record an error occurrence."""
        self.error_count += 1
        self.logger.error(f"Error in {context}: {str(error)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics."""
        uptime = time.time() - self.start_time
        transcription_count = max(self.transcription_count, 1)
        avg_transcription_time = self.total_transcription_time / transcription_count
        
        return {
            "uptime_seconds": round(uptime, 2),
            "requests_total": self.request_count,
            "transcriptions_total": self.transcription_count,
            "errors_total": self.error_count,
            "avg_transcription_time": round(avg_transcription_time, 3),
            "total_audio_duration": round(self.total_audio_duration, 2),
            "requests_per_minute": round((self.request_count / (uptime / 60)), 2) if uptime > 0 else 0.0,
            "model_loaded": self.model_loaded,
            "service_healthy": self.is_healthy
        }

class MonitoringServer:
    """
    Flask-based HTTP server for health checks and monitoring endpoints.
    Runs in a separate thread to avoid blocking the main STT service.
    """
    
    def __init__(self, monitor: ServiceMonitor, port: int = 9091):
        self.monitor = monitor
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup monitoring endpoints."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            health_status = self.monitor.health_check()
            status_code = 200 if health_status["status"] == "healthy" else 503
            return jsonify(health_status), status_code
        
        @self.app.route('/health/ready', methods=['GET'])
        def ready():
            """Readiness check - is the service ready to accept requests?"""
            if self.monitor.model_loaded and self.monitor.is_healthy:
                return jsonify({"status": "ready"}), 200
            else:
                return jsonify({"status": "not ready", "reason": "Model not loaded or service unhealthy"}), 503
        
        @self.app.route('/health/live', methods=['GET'])
        def live():
            """Liveness check - is the service running?"""
            return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()}), 200
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Service metrics endpoint."""
            return jsonify(self.monitor.get_metrics()), 200
        
        @self.app.route('/info', methods=['GET'])
        def info():
            """Service information endpoint."""
            return jsonify({
                "service": "STT Speech-to-Text Service",
                "version": "1.0.0",
                "whisper_backend": self.monitor.config.get('whisper_backend', 'unknown'),
                "model_name": self.monitor.config.get('model_name', 'unknown'),
                "start_time": datetime.fromtimestamp(self.monitor.start_time).isoformat(),
                "endpoints": {
                    "health": "/health",
                    "ready": "/health/ready", 
                    "live": "/health/live",
                    "metrics": "/metrics",
                    "info": "/info"
                }
            }), 200
    
    def run(self, debug: bool = False):
        """Run the monitoring server."""
        self.monitor.logger.info(f"Starting monitoring server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)
    
    def run_in_thread(self):
        """Run the monitoring server in a separate thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread