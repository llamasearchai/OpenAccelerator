#!/usr/bin/env python3
"""
FastAPI server entry point for Open Accelerator API.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
import sys

try:
    import uvicorn
except ImportError:
    print(
        "[ERROR] uvicorn is required but not installed. Install with: pip install uvicorn"
    )
    sys.exit(1)

try:
    from .main import app
except ImportError:
    # Fallback for development
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from open_accelerator.api.main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the FastAPI server."""

    # Server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))

    # Log configuration
    logger.info("Starting OpenAccelerator API server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    logger.info(f"Workers: {workers}")

    # Start server
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers if not debug else 1,
            log_level="info",
            access_log=True,
            server_header=False,
            reload=debug,
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
