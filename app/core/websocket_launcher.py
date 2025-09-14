# app/core/websocket_launcher.py

"""
WebSocket STT Server Launcher
Provides entry point for starting the WebSocket STT service with proper configuration
"""

import asyncio
import signal
import sys
import logging
from typing import Optional

from app.core.websocket_server import WebSocketSTTServer
from app.utils.config import load_config, validate_config, print_config_summary
from app.utils.logger import get_logger


class WebSocketLauncher:
    """Launcher for the WebSocket STT Server with proper lifecycle management"""

    def __init__(self, config_override: Optional[dict] = None):
        # Initialize logger
        self.logger = get_logger("WebSocket_Launcher")

        # Load and validate configuration
        self.config = config_override or load_config()
        is_valid, warnings = validate_config(self.config)

        if warnings:
            for warning in warnings:
                self.logger.warning(f"Configuration warning: {warning}")

        if not is_valid:
            self.logger.error("Configuration validation failed")
            sys.exit(1)

        # Print configuration summary
        print_config_summary(self.config)

        self.server: Optional[WebSocketSTTServer] = None
        self.shutdown_event = asyncio.Event()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def startup(self) -> None:
        """Startup sequence for the WebSocket server"""
        self.logger.info("Starting WebSocket STT Server...")

        try:
            # Initialize server
            self.server = WebSocketSTTServer(self.config)

            # Log startup information
            self.logger.info(f"WebSocket endpoint: ws://localhost:8000/ws/transcribe")
            self.logger.info(f"Health check: http://localhost:8000/health")
            self.logger.info(f"Statistics: http://localhost:8000/stats")
            self.logger.info(f"Monitoring: http://localhost:9091/health")

            # Start the server
            host = self.config.get("websocket", {}).get("host", "0.0.0.0")
            port = self.config.get("websocket", {}).get("port", 8000)

            await self.server.start_server(host=host, port=port)

        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown sequence"""
        self.logger.info("Shutting down WebSocket STT Server...")
        self.shutdown_event.set()

        if self.server:
            # Add any cleanup logic here
            pass

        self.logger.info("Shutdown complete")

    async def run(self) -> None:
        """Main run loop"""
        self.setup_signal_handlers()

        try:
            await self.startup()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    launcher = WebSocketLauncher()
    await launcher.run()


def run_websocket_server():
    """Synchronous entry point for running the WebSocket server"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("WebSocket STT Server stopped")
    except Exception as e:
        print(f"Error starting WebSocket STT Server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_websocket_server()