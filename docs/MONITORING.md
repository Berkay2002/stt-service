# STT Service Monitoring

This document describes the monitoring and health check system for the STT (Speech-to-Text) service.

## Overview

The monitoring system provides comprehensive health checks, metrics collection, and status endpoints for production monitoring and observability.

## Endpoints

### Health Checks

- **`/health`** - Comprehensive health check with detailed status
- **`/health/ready`** - Readiness check (is service ready to accept requests?)
- **`/health/live`** - Liveness check (is service running?)

### Metrics & Info

- **`/metrics`** - Service metrics (requests, transcriptions, performance)
- **`/info`** - Service information and configuration

## Usage

### Running Monitoring Server

```bash
# Run monitoring server only
python app/main.py --monitor

# Or run standalone monitoring service
python app/monitoring_service.py --port 9091
```

### Testing Monitoring

```bash
# Test all endpoints
python app/test_monitoring.py

# Manual testing
curl http://localhost:9091/health
curl http://localhost:9091/metrics
```

### Integration with Main Service

The monitoring system automatically integrates when you run the main STT service:

```bash
# STT service with integrated monitoring
python app/main.py --realtime
```

## Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2025-09-13T10:30:00",
  "uptime_seconds": 1234.56,
  "model": {
    "loaded": true,
    "load_time_seconds": 2.34,
    "backend": "faster-whisper"
  },
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "memory_available_gb": 8.2,
    "memory_total_gb": 16.0
  },
  "metrics": {
    "total_requests": 150,
    "total_transcriptions": 45,
    "total_errors": 2,
    "avg_transcription_time_seconds": 0.123,
    "total_audio_processed_seconds": 567.8
  }
}
```

## Metrics Response

```json
{
  "uptime_seconds": 1234.56,
  "requests_total": 150,
  "transcriptions_total": 45,
  "errors_total": 2,
  "avg_transcription_time": 0.123,
  "total_audio_duration": 567.8,
  "requests_per_minute": 7.3,
  "model_loaded": true,
  "service_healthy": true
}
```

## Docker Integration

The monitoring system is automatically available in Docker:

```bash
# Build and run
docker build -t stt-service .
docker run -p 9090:9090 -p 9091:9091 stt-service

# Access monitoring
curl http://localhost:9091/health
```

## Production Usage

### Kubernetes Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 9091
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 9091
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Monitoring Integration

- **Prometheus**: Metrics endpoint can be scraped at `/metrics`
- **Grafana**: Create dashboards using the metrics data
- **Alerting**: Set up alerts based on health status and metrics

## Troubleshooting

### Common Issues

1. **Port 9091 already in use**
   ```bash
   python app/monitoring_service.py --port 9092
   ```

2. **Model not loading**
   - Check model path configuration
   - Verify model files exist
   - Check logs for detailed error messages

3. **High memory/CPU usage**
   - Monitor system metrics in `/health` endpoint
   - Check for memory leaks or resource issues

### Logs

Monitoring logs are written to:
- Console output
- `stt_service.log` file

### Debug Mode

```bash
python app/monitoring_service.py --debug
```