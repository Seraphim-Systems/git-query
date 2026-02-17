"""Fetch repository data from the server for local training."""

import requests
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class DataFetcher:
    """Fetch data from MongoDB API in batches."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        output_dir: str = "./data/training"
    ):
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://gitquery.davidhoerz.com")
        self.api_key = api_key or os.getenv("APIKEY_MONGODB", "apikey")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def fetch_batch(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Fetch a batch of repositories."""

        payload = {
            "database": "gitquery",
            "collection": "raw_repositories",
            "filter": filters or {},
            "limit": limit,
            "skip": skip,
            "sort": {"stargazers_count": -1}  # Get most popular first
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/mongodb/query",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            documents = result.get("documents", [])

            return documents

        except Exception as e:
            print(f"❌ Error fetching batch (skip={skip}): {e}")
            return []

    def get_total_count(self) -> int:
        """Get total number of repositories."""

        try:
            response = requests.post(
                f"{self.base_url}/api/mongodb/query",
                headers=self.headers,
                json={
                    "database": "gitquery",
                    "collection": "raw_repositories",
                    "filter": {},
                    "limit": 1,
                    "skip": 0
                },
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            return result.get("count", 0)

        except Exception as e:
            print(f"❌ Error getting count: {e}")
            return 0

    def fetch_all(
        self,
        batch_size: int = 100,
        max_repos: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Fetch all repositories in batches."""

        print("\n" + "="*60)
        print("FETCHING DATA FROM SERVER")
        print("="*60)

        # Get total count
        print("\nGetting total count...")
        total = self.get_total_count()
        print(f"✓ Total repositories available: {total}")

        if max_repos:
            total = min(total, max_repos)
            print(f"  Limiting to: {max_repos}")

        all_repos = []
        skip = 0

        while skip < total:
            current_batch_size = min(batch_size, total - skip)
            print(f"\nFetching batch {skip//batch_size + 1} (repos {skip+1}-{skip+current_batch_size})...")

            batch = self.fetch_batch(skip=skip, limit=current_batch_size, filters=filters)

            if not batch:
                print("⚠ No more data returned, stopping...")
                break

            all_repos.extend(batch)
            print(f"✓ Fetched {len(batch)} repositories (total: {len(all_repos)})")

            skip += batch_size

        return all_repos

    def save_data(self, repositories: List[Dict], filename: str = None):
        """Save repository data to JSON file."""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"repositories_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(repositories, f, indent=2, default=str)

        print(f"\n✓ Saved {len(repositories)} repositories to {filepath}")

        # Also save as "latest"
        latest_path = self.output_dir / "repositories_latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(repositories, f, indent=2, default=str)

        print(f"✓ Saved as latest: {latest_path}")

        return filepath

    def print_summary(self, repositories: List[Dict]):
        """Print data summary."""

        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)

        print(f"\nTotal repositories: {len(repositories)}")

        # Language distribution
        languages = {}
        for repo in repositories:
            lang = repo.get("language") or "Unknown"
            languages[lang] = languages.get(lang, 0) + 1

        print(f"\nTop 10 languages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / len(repositories)) * 100
            print(f"  {lang}: {count} ({pct:.1f}%)")

        # Star distribution
        stars = [repo.get("stargazers_count", 0) or repo.get("stars", 0) for repo in repositories]
        if stars:
            print(f"\nStar statistics:")
            print(f"  Min: {min(stars):,}")
            print(f"  Max: {max(stars):,}")
            print(f"  Avg: {sum(stars)/len(stars):,.0f}")

        print("\n" + "="*60)


def main():
    """Main function to fetch data."""

    print("\n" + "="*60)
    print("FETCH TRAINING DATA FROM SERVER")
    print("="*60)

    base_url = input(f"\nServer URL: ").strip()
    if not base_url:
        base_url = os.getenv("API_BASE_URL")
        if not base_url:
            raise ValueError("API_BASE_URL must be provided or set in environment")

    api_key = input("MongoDB API key: ").strip()
    if not api_key:
        api_key = os.getenv("APIKEY_MONGODB")
        if not api_key:
            raise ValueError("APIKEY_MONGODB must be provided or set in environment")

    batch_size = input("Batch size (default: 100): ").strip()
    batch_size = int(batch_size) if batch_size else 100

    max_repos = input("Max repositories (default: all): ").strip()
    max_repos = int(max_repos) if max_repos else None

    # Initialize fetcher
    fetcher = DataFetcher(base_url=base_url, api_key=api_key)

    # Fetch data
    repositories = fetcher.fetch_all(
        batch_size=batch_size,
        max_repos=max_repos
    )

    if not repositories:
        print("\n❌ No data fetched!")
        return

    # Save data
    fetcher.save_data(repositories)

    # Print summary
    fetcher.print_summary(repositories)

    print("\n✓ Data ready for training!")
    print("\nNext step: python -m src.recommender.scripts.train_local")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFetch interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
