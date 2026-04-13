"""JWT token creation and verification utilities."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from src.shared.config import settings

logger = logging.getLogger(__name__)


def create_access_token(user_id: str, username: str, is_admin: bool = False) -> str:
    """Create a signed JWT access token.

    Args:
        user_id: The user's unique identifier.
        username: The user's display name.
        is_admin: Whether the user has admin privileges.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + timedelta(seconds=settings.jwt_expiration)
    payload = {
        "sub": user_id,
        "username": username,
        "is_admin": is_admin,
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> Optional[dict]:
    """Verify a JWT token and return its payload.

    Args:
        token: The JWT string to verify.

    Returns:
        Decoded payload dict, or None if invalid/expired.
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug("Invalid JWT token: %s", e)
        return None
