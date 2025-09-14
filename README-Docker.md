# STT Service Docker Deployment (High-End RTX Optimized)

This guide covers deploying the STT service using Docker optimized for high-end RTX cards.

## 🎯 Features

- **High-End RTX Optimized**: Configured for high VRAM and maximum compute power
- **Large-v3 Model**: Best accuracy with GPU acceleration
- **Docker Health Checks**: Automatic monitoring and restart
- **Persistent Storage**: Models and logs persist across container restarts
- **Production Ready**: Security hardening and resource limits

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **NVIDIA Docker Runtime** (`nvidia-docker2`)
3. **High-End RTX GPU** with latest drivers
4. **CUDA 12.3+** compatible drivers

### Windows Deployment

```bash
# Run the deployment script
run-gpu.bat
```

### Linux/macOS Deployment

```bash
# Make script executable
chmod +x run-gpu.sh

# Run deployment
./run-gpu.sh
```

### Manual Deployment

```bash
# Create directories
mkdir -p models logs

# Build container
docker build -f Dockerfile.gpu -t stt-service:gpu .

# Deploy with docker-compose
docker-compose -f docker-compose.gpu.yml up -d
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `docker-compose.gpu.yml`:

```yaml
- CUDA_VISIBLE_DEVICES=0          # Use first GPU
- CUDA_MEMORY_FRACTION=0.95       # Use 95% of VRAM
- PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
```

### Resource Limits

```yaml
mem_limit: 32g        # Adjust based on your system RAM
shm_size: 2g         # Shared memory for CUDA operations
```

### GPU Configuration

```yaml
devices:
  - driver: nvidia
    device_ids: ['0']  # Use GPU 0 (high-end RTX card)
    capabilities: [gpu, compute, utility]
```

## 📊 Monitoring

### Health Checks

- **Health Endpoint**: `http://localhost:9091/health`
- **Metrics Endpoint**: `http://localhost:9091/metrics`
- **Service Info**: `http://localhost:9091/info`

### Container Monitoring

```bash
# View logs
docker-compose -f docker-compose.gpu.yml logs -f

# Check GPU usage
docker exec stt-gpu-service nvidia-smi

# Check container stats
docker stats stt-gpu-service
```

## 🛠️ Management Commands

### Start/Stop/Restart

```bash
# Start service
docker-compose -f docker-compose.gpu.yml up -d

# Stop service
docker-compose -f docker-compose.gpu.yml down

# Restart service
docker-compose -f docker-compose.gpu.yml restart

# View logs
docker-compose -f docker-compose.gpu.yml logs -f stt-service-gpu
```

### Scaling and Updates

```bash
# Rebuild and redeploy
docker-compose -f docker-compose.gpu.yml up -d --build

# Pull latest base images
docker-compose -f docker-compose.gpu.yml pull

# Clean unused images
docker image prune -a
```

## 🔍 Troubleshooting

### Common Issues

1. **NVIDIA Runtime Not Found**
   ```bash
   # Install nvidia-docker2
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Out of Memory Errors**
   ```yaml
   # Reduce memory fraction in docker-compose.gpu.yml
   - CUDA_MEMORY_FRACTION=0.8
   ```

3. **Audio Device Issues**
   ```bash
   # Check if audio devices are accessible
   docker exec stt-rtx3090 aplay -l
   ```

4. **Model Download Issues**
   ```bash
   # Pre-download models
   docker exec stt-gpu-service python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
   ```

### Performance Tuning

1. **VRAM Usage**
   ```bash
   # Monitor VRAM usage
   docker exec stt-gpu-service nvidia-smi -l 1
   ```

2. **CPU Usage**
   ```bash
   # Adjust CPU thread count in config
   # Modify app/utils/config.py in container
   ```

3. **Network Performance**
   ```yaml
   # Use host networking for better performance
   network_mode: "host"
   ```

## 🏗️ Production Deployment

### With Nginx Proxy

```bash
# Start with production profile
docker-compose -f docker-compose.gpu.yml --profile production up -d
```

### SSL Configuration

1. Place SSL certificates in `./ssl/`
2. Update `nginx.conf` with your domain
3. Restart proxy: `docker-compose restart stt-proxy`

### Load Balancing

For multiple GPU setups:

```yaml
# Scale service across GPUs
services:
  stt-service-gpu-0:
    # ... GPU 0 config
  stt-service-gpu-1:
    # ... GPU 1 config
```

## 📈 Performance Metrics

### Expected Performance (High-End RTX)

- **Model**: large-v3
- **Processing Speed**: ~15-20x real-time
- **VRAM Usage**: ~8-12GB
- **Latency**: <500ms for real-time transcription

### Monitoring Dashboard

Access Grafana-compatible metrics at:
- `http://localhost:9091/metrics`

## 🛡️ Security

### Container Security

- Runs as non-root user (`uid: 1000`)
- No privileged access required
- Limited resource access
- Network isolation

### Production Hardening

- Update base images regularly
- Scan for vulnerabilities
- Use secrets management
- Enable audit logging

## 📝 Logs

### Log Locations

- **Container logs**: `docker-compose logs`
- **Application logs**: `./logs/` (persistent volume)
- **Model cache**: `./models/` (persistent volume)

### Log Rotation

Configured for automatic log rotation:
- Max size: 100MB per file
- Keep 3 historical files
- JSON format for structured logging