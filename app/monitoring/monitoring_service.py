# app/monitoring/monitoring_service.py

"""
Standalone monitoring service for the STT service.
Can be run independently to provide health checks and metrics.
"""

import sys
import time
import argparse
from app.utils.config import load_config
from app.monitoring.service_monitor import ServiceMonitor, MonitoringServer

def run_monitoring_service():
    """Run the monitoring service standalone."""
    parser = argparse.ArgumentParser(description='STT Service Monitoring Server')
    parser.add_argument('--port', type=int, default=9091, help='Port to run monitoring server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize monitor
        print("🔍 Initializing STT Service Monitor...")
        monitor = ServiceMonitor(config)
        
        # Create monitoring server
        monitoring_server = MonitoringServer(monitor, port=args.port)
        
        print(f"🌐 Starting monitoring server on port {args.port}")
        print(f"📊 Health check: http://localhost:{args.port}/health")
        print(f"📈 Metrics: http://localhost:{args.port}/metrics")
        print(f"ℹ️  Service info: http://localhost:{args.port}/info")
        print("Press Ctrl+C to stop...")
        
        # Run server
        monitoring_server.run(debug=args.debug)
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring service stopped by user")
    except Exception as e:
        print(f"❌ Error running monitoring service: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_monitoring_service()