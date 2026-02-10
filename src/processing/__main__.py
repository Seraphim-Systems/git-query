"""
Data Processing Service - Entry Point
Handles data cleaning, transformation, and vectorization
"""

import asyncio
import logging
import uvicorn
from multiprocessing import Process
from processing.processor import DataProcessor
from processing.config import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_health_server():
    """Run health check server in separate process"""
    uvicorn.run(
        "processing.health:app",
        host="0.0.0.0",
        port=8090,
        log_level=settings.log_level.lower()
    )


async def main():
    """Main processing loop"""
    logger.info("Starting Data Processing Service...")
    
    # Start health check server in background
    health_process = Process(target=run_health_server)
    health_process.start()
    logger.info("Health check server started on port 8090")
    
    processor = DataProcessor()
    
    try:
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await processor.cleanup()
        health_process.terminate()
        health_process.join()


if __name__ == "__main__":
    asyncio.run(main())