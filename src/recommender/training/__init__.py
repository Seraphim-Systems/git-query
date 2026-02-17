"""Training pipelines for the recommendation system."""

# Don't import other modules here - they have dependencies we don't need
# Just export the unified pipeline for Docker usage

__all__ = [
    "UnifiedTrainingPipeline",
]

# Main entry point for Docker container
if __name__ == "__main__":
    from .unified_pipeline import main

    main()
