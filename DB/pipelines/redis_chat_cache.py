"""
Redis Chatbot Cache Layer
Manages chat history, user sessions, query results, and user preferences
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisChatCache:
    """Redis cache layer for chatbot operations."""
    
    # TTL configuration (1 week)
    TTL_WEEK = 7 * 24 * 60 * 60  # seconds
    
    # Key prefixes
    PREFIX_CHAT_HISTORY = "chat:history:"
    PREFIX_USER_SESSION = "user:session:"
    PREFIX_QUERY_RESULT = "query:result:"
    PREFIX_USER_PREFS = "user:prefs:"
    PREFIX_REPO_CACHE = "repo:cache:"
    
    def __init__(self):
        """Initialize Redis cache."""
        self.db_manager = DatabaseManager()
        self.redis_client = self.db_manager.get_redis()
    
    # =========================================================================
    # Chat History Management
    # =========================================================================
    
    def save_chat_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Save a chat message to history."""
        try:
            key = f"{self.PREFIX_CHAT_HISTORY}{session_id}"
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Append to list
            self.redis_client.rpush(key, json.dumps(message))
            
            # Set expiration
            self.redis_client.expire(key, self.TTL_WEEK)
            
            logger.debug(f"Saved chat message for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session."""
        try:
            key = f"{self.PREFIX_CHAT_HISTORY}{session_id}"
            
            # Get last N messages
            messages = self.redis_client.lrange(key, -limit, -1)
            
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []
    
    def clear_chat_history(self, session_id: str):
        """Clear chat history for a session."""
        try:
            key = f"{self.PREFIX_CHAT_HISTORY}{session_id}"
            self.redis_client.delete(key)
            logger.info(f"Cleared chat history for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
    
    # =========================================================================
    # User Session Management
    # =========================================================================
    
    def create_user_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create a new user session."""
        try:
            session_id = f"{user_id}_{datetime.utcnow().timestamp()}"
            key = f"{self.PREFIX_USER_SESSION}{session_id}"
            
            session_info = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                **session_data
            }
            
            self.redis_client.setex(
                key,
                self.TTL_WEEK,
                json.dumps(session_info)
            )
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return ""
    
    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data."""
        try:
            key = f"{self.PREFIX_USER_SESSION}{session_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            return None
    
    def update_user_session(self, session_id: str, updates: Dict[str, Any]):
        """Update user session data."""
        try:
            session = self.get_user_session(session_id)
            if session:
                session.update(updates)
                session["updated_at"] = datetime.utcnow().isoformat()
                
                key = f"{self.PREFIX_USER_SESSION}{session_id}"
                self.redis_client.setex(
                    key,
                    self.TTL_WEEK,
                    json.dumps(session)
                )
                logger.debug(f"Updated session {session_id}")
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    def end_user_session(self, session_id: str):
        """End a user session."""
        try:
            key = f"{self.PREFIX_USER_SESSION}{session_id}"
            self.redis_client.delete(key)
            logger.info(f"Ended session {session_id}")
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    # =========================================================================
    # Query Result Caching
    # =========================================================================
    
    def cache_query_result(self, query_hash: str, results: List[Dict[str, Any]]):
        """Cache query results."""
        try:
            key = f"{self.PREFIX_QUERY_RESULT}{query_hash}"
            
            cache_data = {
                "results": results,
                "cached_at": datetime.utcnow().isoformat(),
                "result_count": len(results)
            }
            
            self.redis_client.setex(
                key,
                self.TTL_WEEK,
                json.dumps(cache_data)
            )
            
            logger.debug(f"Cached query results for hash {query_hash}")
        except Exception as e:
            logger.error(f"Error caching query results: {e}")
    
    def get_cached_query(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached query results."""
        try:
            key = f"{self.PREFIX_QUERY_RESULT}{query_hash}"
            data = self.redis_client.get(key)
            
            if data:
                cache_data = json.loads(data)
                return cache_data.get("results", [])
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached query: {e}")
            return None
    
    # =========================================================================
    # User Preferences Management
    # =========================================================================
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Save user preferences (language filters, topic preferences, etc.)."""
        try:
            key = f"{self.PREFIX_USER_PREFS}{user_id}"
            
            prefs_data = {
                "user_id": user_id,
                "preferences": preferences,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.redis_client.setex(
                key,
                self.TTL_WEEK,
                json.dumps(prefs_data)
            )
            
            logger.info(f"Saved preferences for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user preferences."""
        try:
            key = f"{self.PREFIX_USER_PREFS}{user_id}"
            data = self.redis_client.get(key)
            
            if data:
                prefs_data = json.loads(data)
                return prefs_data.get("preferences", {})
            return None
        except Exception as e:
            logger.error(f"Error retrieving preferences: {e}")
            return None
    
    def update_user_language_filter(self, user_id: str, languages: List[str]):
        """Update user's language filter preference."""
        try:
            prefs = self.get_user_preferences(user_id) or {}
            prefs["language_filter"] = languages
            self.save_user_preferences(user_id, prefs)
            logger.info(f"Updated language filter for user {user_id}: {languages}")
        except Exception as e:
            logger.error(f"Error updating language filter: {e}")
    
    # =========================================================================
    # Repository Data Caching
    # =========================================================================
    
    def cache_repository_data(self, repo_id: str, repo_data: Dict[str, Any]):
        """Cache frequently accessed repository data."""
        try:
            key = f"{self.PREFIX_REPO_CACHE}{repo_id}"
            
            self.redis_client.setex(
                key,
                self.TTL_WEEK,
                json.dumps(repo_data)
            )
            
            logger.debug(f"Cached repository data for {repo_id}")
        except Exception as e:
            logger.error(f"Error caching repository data: {e}")
    
    def get_cached_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached repository data."""
        try:
            key = f"{self.PREFIX_REPO_CACHE}{repo_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached repository: {e}")
            return None
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_user_data(self, user_id: str):
        """Clear all cached data for a user."""
        try:
            # Get all keys for user
            patterns = [
                f"{self.PREFIX_USER_SESSION}{user_id}_*",
                f"{self.PREFIX_USER_PREFS}{user_id}",
                f"{self.PREFIX_CHAT_HISTORY}{user_id}_*"
            ]
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            logger.info(f"Cleared all data for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing user data: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        try:
            stats = {
                "chat_histories": len(self.redis_client.keys(f"{self.PREFIX_CHAT_HISTORY}*")),
                "user_sessions": len(self.redis_client.keys(f"{self.PREFIX_USER_SESSION}*")),
                "query_results": len(self.redis_client.keys(f"{self.PREFIX_QUERY_RESULT}*")),
                "user_preferences": len(self.redis_client.keys(f"{self.PREFIX_USER_PREFS}*")),
                "cached_repos": len(self.redis_client.keys(f"{self.PREFIX_REPO_CACHE}*"))
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


def main():
    """Example usage of Redis cache."""
    cache = RedisChatCache()
    
    # Example: Create session and chat
    session_id = cache.create_user_session("user123", {"name": "Test User"})
    cache.save_chat_message(session_id, "user", "Hello, recommend me Python repos")
    cache.save_chat_message(session_id, "assistant", "Here are some Python repositories...")
    
    # Example: Set preferences
    cache.save_user_preferences("user123", {
        "language_filter": ["Python", "JavaScript"],
        "min_stars": 100
    })
    
    # Get stats
    stats = cache.get_cache_stats()
    logger.info(f"Cache stats: {stats}")


if __name__ == "__main__":
    main()
