"""Tests for data processing pipeline"""

import pytest
from processing.pipelines.transformation import DataTransformer


def test_data_transformer():
    """Test data transformation"""
    transformer = DataTransformer()

    raw_record = {
        "repo_id": "123456",
        "full_name": "owner/repo-name",
        "description": "A great repository for testing",
        "language": "Python",
        "stars": 100,
        "forks": 10,
        "topics": ["python", "testing", "automation"],
        "created_at": "2024-01-01T00:00:00Z",
    }

    cleaned = transformer.transform(raw_record)

    assert cleaned is not None
    assert cleaned["repo_id"] == "123456"
    assert cleaned["name"] == "repo-name"
    assert cleaned["owner"] == "owner"
    assert cleaned["language"] == "Python"
    assert len(cleaned["topics"]) == 3
