"""Data transformation pipeline - cleans and validates data"""

import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

from processing.config import settings

logger = logging.getLogger(__name__)

class DataTransformer:
    """Transforms and cleans raw repository data"""
    
    def __init__(self):
        self.required_fields = settings.required_fields
        self.min_desc_length = settings.min_description_length
        self.max_desc_length = settings.max_description_length
    
    def transform(self, raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform and clean a raw record
        
        Args:
            raw_record: Raw repository data
            
        Returns:
            Cleaned record or None if invalid
        """
        try:
            # Validate required fields
            if not self._validate_required_fields(raw_record):
                logger.warning(f"Record missing required fields: {raw_record.get('_id')}")
                return None
            
            cleaned = {
                # Core fields
                "repo_id": self._clean_string(raw_record.get("repo_id")),
                "full_name": self._clean_string(raw_record.get("full_name")),
                "name": self._extract_repo_name(raw_record.get("full_name")),
                "owner": self._extract_owner(raw_record.get("full_name")),
                
                # Description
                "description": self._clean_description(raw_record.get("description")),
                
                # Metadata
                "language": self._clean_string(raw_record.get("language")),
                "stars": self._clean_integer(raw_record.get("stars", 0)),
                "forks": self._clean_integer(raw_record.get("forks", 0)),
                "watchers": self._clean_integer(raw_record.get("watchers", 0)),
                "open_issues": self._clean_integer(raw_record.get("open_issues", 0)),
                
                # Dates
                "created_at": self._parse_date(raw_record.get("created_at")),
                "updated_at": self._parse_date(raw_record.get("updated_at")),
                "pushed_at": self._parse_date(raw_record.get("pushed_at")),
                
                # Booleans
                "is_fork": bool(raw_record.get("is_fork", False)),
                "is_archived": bool(raw_record.get("is_archived", False)),
                "is_private": bool(raw_record.get("is_private", False)),
                
                # Additional fields
                "topics": self._clean_topics(raw_record.get("topics", [])),
                "license": self._clean_string(raw_record.get("license")),
                "homepage": self._clean_url(raw_record.get("homepage")),
                
                # URLs
                "html_url": self._clean_url(raw_record.get("html_url")),
                "clone_url": self._clean_url(raw_record.get("clone_url")),
                
                # Search metadata
                "search_text": self._create_search_text(raw_record),
            }
            
            # Validate cleaned record
            if not self._validate_cleaned_record(cleaned):
                return None
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            return None
    
    def _validate_required_fields(self, record: Dict[str, Any]) -> bool:
        """Check if all required fields are present"""
        for field in self.required_fields:
            if field not in record or not record[field]:
                return False
        return True
    
    def _validate_cleaned_record(self, record: Dict[str, Any]) -> bool:
        """Validate the cleaned record"""
        # Check description length
        desc = record.get("description", "")
        if len(desc) < self.min_desc_length:
            logger.warning(f"Description too short: {record.get('full_name')}")
            return False
        
        # Check for valid full_name format
        if not record.get("full_name") or "/" not in record.get("full_name", ""):
            logger.warning(f"Invalid full_name: {record.get('full_name')}")
            return False
        
        return True
    
    def _clean_string(self, value: Any) -> str:
        """Clean and normalize string"""
        if value is None:
            return ""
        
        value = str(value).strip()
        # Remove excessive whitespace
        value = re.sub(r'\s+', ' ', value)
        return value
    
    def _clean_description(self, description: Any) -> str:
        """Clean repository description"""
        desc = self._clean_string(description)
        
        if not desc:
            return ""
        
        # Truncate if too long
        if len(desc) > self.max_desc_length:
            desc = desc[:self.max_desc_length] + "..."
        
        # Remove common prefixes/patterns
        desc = re.sub(r'^(A |An |The )', '', desc, flags=re.IGNORECASE)
        
        return desc.strip()
    
    def _clean_integer(self, value: Any) -> int:
        """Clean and convert to integer"""
        try:
            return max(0, int(value))
        except (ValueError, TypeError):
            return 0
    
    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        try:
            # Try ISO format
            return datetime.fromisoformat(str(date_value).replace('Z', '+00:00'))
        except:
            return None
    
    def _clean_topics(self, topics: Any) -> list[str]:
        """Clean and normalize topics"""
        if not topics or not isinstance(topics, list):
            return []
        
        cleaned = []
        for topic in topics:
            topic = self._clean_string(topic).lower()
            if topic and len(topic) <= 50:  # Reasonable topic length
                cleaned.append(topic)
        
        return list(set(cleaned))[:20]  # Max 20 unique topics
    
    def _clean_url(self, url: Any) -> str:
        """Clean and validate URL"""
        url = self._clean_string(url)
        
        if not url:
            return ""
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            return ""
        
        return url
    
    def _extract_repo_name(self, full_name: str) -> str:
        """Extract repository name from full_name"""
        if not full_name or "/" not in full_name:
            return ""
        return full_name.split("/", 1)[1]
    
    def _extract_owner(self, full_name: str) -> str:
        """Extract owner from full_name"""
        if not full_name or "/" not in full_name:
            return ""
        return full_name.split("/", 1)[0]
    
    def _create_search_text(self, record: Dict[str, Any]) -> str:
        """Create searchable text field combining multiple fields"""
        parts = [
            record.get("full_name", ""),
            record.get("description", ""),
            record.get("language", ""),
            " ".join(record.get("topics", [])),
        ]
        
        search_text = " ".join(filter(None, parts))
        return self._clean_string(search_text).lower()