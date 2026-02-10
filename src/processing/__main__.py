"""
Data Processing Service - Entry Point
Handles data cleaning, transformation, and vectorization
"""

import asyncio
import logging
from processing.processor import DataProcessor
from processing.config import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main processing loop"""
    logger.info("Starting Data Processing Service...")
    
    processor = DataProcessor()
    
    try:
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())