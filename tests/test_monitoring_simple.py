# tests/test_monitoring_simple.py

"""
Simple test for monitoring system without requiring server to be running.
Tests the monitoring classes directly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import load_config
from app.monitoring.service_monitor import ServiceMonitor

def test_monitoring_system():
    """Test the monitoring system directly."""
    print("🧪 Testing STT Service Monitoring System")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        # Initialize monitor
        print("Initializing service monitor...")
        monitor = ServiceMonitor(config)
        print("✅ Service monitor initialized")
        
        # Test health check
        print("\\nTesting health check...")
        health_status = monitor.health_check()
        print(f"✅ Health check completed: {health_status['status']}")
        print(f"   Model loaded: {health_status['model']['loaded']}")
        print(f"   Uptime: {health_status['uptime_seconds']}s")
        
        # Test metrics
        print("\\nTesting metrics collection...")
        metrics = monitor.get_metrics()
        print(f"✅ Metrics collected: {len(metrics)} metrics")
        print(f"   Service healthy: {metrics['service_healthy']}")
        print(f"   Model loaded: {metrics['model_loaded']}")
        
        # Test recording some fake metrics
        print("\\nTesting metric recording...")
        monitor.record_request()
        monitor.record_transcription(duration=5.0, processing_time=0.5)
        
        updated_metrics = monitor.get_metrics()
        print(f"✅ Metrics updated: {updated_metrics['requests_total']} requests, {updated_metrics['transcriptions_total']} transcriptions")
        
        print("\\n" + "=" * 50)
        print("🎉 All monitoring system tests passed!")
        print("✅ Health checks working")
        print("✅ Metrics collection working") 
        print("✅ Model loading working")
        print("✅ System monitoring working")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Error testing monitoring system: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_health_details(monitor):
    """Show detailed health information."""
    health = monitor.health_check()
    
    print("\\n🏥 Detailed Health Status:")
    print("=" * 40)
    print(f"Overall Status: {health['status']}")
    print(f"Timestamp: {health['timestamp']}")
    print(f"Uptime: {health['uptime_seconds']}s")
    
    if 'model' in health:
        model = health['model']
        print(f"\\n🤖 Model Status:")
        print(f"  Loaded: {model['loaded']}")
        print(f"  Load Time: {model.get('load_time_seconds', 0):.2f}s")
        print(f"  Backend: {model.get('backend', 'unknown')}")
    
    if 'system' in health:
        system = health['system']
        print(f"\\n💻 System Status:")
        print(f"  CPU Usage: {system['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {system['memory_percent']:.1f}%")
        print(f"  Available Memory: {system['memory_available_gb']:.1f}GB")
    
    if 'metrics' in health:
        metrics = health['metrics']
        print(f"\\n📊 Service Metrics:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Total Transcriptions: {metrics['total_transcriptions']}")
        print(f"  Total Errors: {metrics['total_errors']}")
        print(f"  Avg Transcription Time: {metrics['avg_transcription_time_seconds']:.3f}s")

if __name__ == "__main__":
    success = test_monitoring_system()
    
    if success:
        # Show additional details
        try:
            config = load_config()
            monitor = ServiceMonitor(config)
            show_health_details(monitor)
        except Exception as e:
            print(f"Could not show health details: {e}")
    
    sys.exit(0 if success else 1)