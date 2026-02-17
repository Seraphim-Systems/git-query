"""Main entry point for the recommender service."""

import uvicorn
from .config import settings

if __name__ == "__main__":
    uvicorn.run(
        "recommender.api:app",
        host=settings.recommender_host,
        port=settings.recommender_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
