"""
GitHub Repository Scraper Pipeline
Collects repository data from GitHub API and stores in Cosmos DB
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseManager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubScraper:
    """Scrapes GitHub for repository data and stores in Cosmos DB."""
    
    def __init__(self):
        """Initialize scraper with database connections."""
        self.db_manager = DatabaseManager()
        self.cosmos_db = self.db_manager.get_cosmos()["gitquery_cosmos"]
        self.collection = self.cosmos_db["repository_activity"]
        
        # GitHub API setup
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable required")
        
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.headers)
    
    def fetch_repository_data(self, owner: str, name: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive repository data from GitHub API."""
        try:
            # Main repository data
            repo_url = f"{self.base_url}/repos/{owner}/{name}"
            response = self.session.get(repo_url, timeout=30)
            response.raise_for_status()
            repo_data = response.json()
            
            # Language statistics
            languages_url = f"{repo_url}/languages"
            lang_response = self.session.get(languages_url, timeout=30)
            languages = lang_response.json() if lang_response.status_code == 200 else {}
            
            # Topics
            topics_url = f"{repo_url}/topics"
            topics_response = self.session.get(
                topics_url,
                headers={**self.headers, "Accept": "application/vnd.github.mercy-preview+json"},
                timeout=30
            )
            topics = topics_response.json().get("names", []) if topics_response.status_code == 200 else []
            
            # Code of conduct
            conduct_url = f"{repo_url}/community/code_of_conduct"
            conduct_response = self.session.get(conduct_url, timeout=30)
            conduct = conduct_response.json().get("name", "None") if conduct_response.status_code == 200 else "None"
            
            # Transform to schema format
            return {
                "owner": repo_data["owner"]["login"],
                "name": repo_data["name"],
                "stars": repo_data["stargazers_count"],
                "forks": repo_data["forks_count"],
                "watchers": repo_data["watchers_count"],
                "isFork": repo_data["fork"],
                "isArchived": repo_data["archived"],
                "languages": list(languages.keys()),
                "languageCount": len(languages),
                "topics": topics,
                "topicCount": len(topics),
                "diskUsageKb": repo_data.get("size", 0),
                "pullRequests": repo_data.get("open_issues_count", 0),  # Approximation
                "issues": repo_data.get("open_issues_count", 0),
                "description": repo_data.get("description", ""),
                "primaryLanguage": repo_data.get("language", ""),
                "createdAt": repo_data["created_at"],
                "pushedAt": repo_data["pushed_at"],
                "defaultBranchCommitCount": 0,  # Requires additional API call
                "license": repo_data.get("license", {}).get("name", "") if repo_data.get("license") else "",
                "assignableUserCount": 0,  # Requires additional API call
                "codeOfConduct": conduct,
                "forkingAllowed": not repo_data.get("disabled", False),
                "nameWithOwner": repo_data["full_name"],
                "parent": repo_data.get("parent", {}).get("full_name") if repo_data.get("parent") else None,
                "scrapedAt": datetime.utcnow().isoformat(),
                "_id": f"{owner}_{name}"  # Cosmos DB identifier
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {owner}/{name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {owner}/{name}: {e}")
            return None
    
    def scrape_trending_repositories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Scrape trending/popular repositories from GitHub."""
        try:
            # Search for popular repositories updated in last week
            query = "stars:>1000 pushed:>2025-01-01"
            search_url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(limit, 100)
            }
            
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json()
            
            repositories = []
            for item in results.get("items", []):
                owner = item["owner"]["login"]
                name = item["name"]
                
                logger.info(f"Fetching detailed data for {owner}/{name}")
                repo_data = self.fetch_repository_data(owner, name)
                
                if repo_data:
                    repositories.append(repo_data)
                
                # Rate limit handling
                time.sleep(1)  # Be nice to GitHub API
            
            return repositories
            
        except Exception as e:
            logger.error(f"Error scraping trending repositories: {e}")
            return []
    
    def store_to_cosmos(self, repositories: List[Dict[str, Any]]) -> int:
        """Store scraped repositories in Cosmos DB."""
        stored_count = 0
        
        for repo in repositories:
            try:
                # Upsert to Cosmos DB
                self.collection.update_one(
                    {"_id": repo["_id"]},
                    {"$set": repo},
                    upsert=True
                )
                stored_count += 1
                logger.info(f"Stored {repo['nameWithOwner']} in Cosmos DB")
            except Exception as e:
                logger.error(f"Failed to store {repo.get('nameWithOwner')}: {e}")
        
        return stored_count
    
    def run(self, limit: int = 100):
        """Run the scraping pipeline."""
        logger.info("Starting GitHub scraping pipeline")
        start_time = datetime.now()
        
        # Scrape repositories
        repositories = self.scrape_trending_repositories(limit=limit)
        logger.info(f"Scraped {len(repositories)} repositories")
        
        # Store in Cosmos DB
        stored = self.store_to_cosmos(repositories)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed: {stored} repositories stored in {elapsed:.2f}s")
        
        return stored


def main():
    """Main entry point for scraper."""
    scraper = GitHubScraper()
    scraper.run(limit=100)


if __name__ == "__main__":
    main()
