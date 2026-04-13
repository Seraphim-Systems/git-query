"""Shared utilities for training and inference text preparation."""

from typing import Dict


def prepare_repo_text(repo: Dict) -> str:
    """Prepare repository text for embedding.

    Canonical text representation used by both training and inference.
    Fields: name, description, topics, language, readme (truncated to 500 chars).

    Args:
        repo: Repository dict from MongoDB (repositories collection).

    Returns:
        Pipe-separated text suitable for embedding.
    """
    parts = []

    if repo.get("name"):
        parts.append(f"Repository: {repo['name']}")
    if repo.get("description"):
        parts.append(f"Description: {repo['description']}")
    if repo.get("topics"):
        topics = repo["topics"]
        if isinstance(topics, list) and topics:
            topic_strings = []
            for topic in topics:
                if isinstance(topic, dict):
                    topic_str = topic.get("name") or topic.get("topic") or str(topic)
                    topic_strings.append(topic_str)
                else:
                    topic_strings.append(str(topic))
            if topic_strings:
                parts.append(f"Topics: {', '.join(topic_strings)}")
    if repo.get("language"):
        parts.append(f"Language: {repo['language']}")
    if repo.get("readme"):
        readme = str(repo["readme"])[:500]
        parts.append(f"README: {readme}")

    return " | ".join(parts)
