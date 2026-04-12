"""Dataset access for data science notebook workflows.

Provides a simple interface to load the git-query repository dataset
into pandas DataFrames for exploration, feature engineering, and training.

Usage from a Jupyter notebook::

    from src.recommender.data import RepoDataset

    # Fetch from gateway
    ds = RepoDataset.from_gateway(url="https://...", api_key="sk-...")
    df = ds.to_dataframe()

    # Or load from local cache
    ds = RepoDataset.from_local("repos.parquet")
    df = ds.to_dataframe()
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RepoDataset:
    """Lightweight wrapper around a list of repository dicts.

    Supports conversion to pandas DataFrame / numpy array, local
    persistence (JSON and Parquet), and basic summary statistics.
    """

    def __init__(self, repos: List[Dict[str, Any]]) -> None:
        self.repos: List[Dict[str, Any]] = repos
        self._df = None  # lazy-cached DataFrame

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_gateway(
        cls,
        url: str,
        api_key: str,
        max_repos: Optional[int] = None,
        batch_size: int = 500,
    ) -> "RepoDataset":
        """Fetch repositories from the gateway API.

        Uses the same ``POST /api/mongodb/query`` endpoint and pagination
        strategy as ``MongoDataFetcher`` (training/data/mongo_data_fetcher.py).

        Parameters
        ----------
        url:
            Base URL of the gateway (e.g. ``https://gateway.example.com``).
        api_key:
            Bearer token for the ``Authorization`` header.
        max_repos:
            Optional cap on how many repos to fetch.
        batch_size:
            Number of documents to request per HTTP call.
        """
        import requests

        base_url = url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{base_url}/api/mongodb/query"

        # -- determine total count --
        total = cls._get_total_count(endpoint, headers)
        if max_repos is not None:
            total = min(total, max_repos)

        print(f"[RepoDataset] Target repo count: {total}")

        if total == 0:
            print("[RepoDataset] No repositories found.")
            return cls([])

        # -- paginated fetch --
        all_repos: List[Dict[str, Any]] = []
        skip = 0

        while skip < total:
            limit = min(batch_size, total - skip)
            payload = {
                "database": "gitquery",
                "collection": "repositories",
                "filter": {},
                "limit": limit,
                "skip": skip,
                "sort": {"stars": -1},
            }

            try:
                resp = requests.post(
                    endpoint, headers=headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                batch = resp.json().get("documents", [])
            except Exception as exc:
                print(f"[RepoDataset] Error at skip={skip}: {exc}")
                break

            if not batch:
                print("[RepoDataset] Empty batch returned, stopping.")
                break

            all_repos.extend(batch)
            print(
                f"[RepoDataset] Fetched batch {skip // batch_size + 1} "
                f"({len(batch)} docs, total so far: {len(all_repos)})"
            )

            skip += batch_size
            time.sleep(0.1)

        print(f"[RepoDataset] Done. {len(all_repos)} repositories loaded.")
        return cls(all_repos)

    @classmethod
    def from_local(cls, path: str) -> "RepoDataset":
        """Load a dataset from a local JSON or Parquet file.

        Parameters
        ----------
        path:
            Filesystem path.  Extension determines format:
            ``.json`` uses the stdlib json module;
            ``.parquet`` requires *pyarrow* and *pandas*.
        """
        p = Path(path)
        ext = p.suffix.lower()

        if ext == ".json":
            with open(p, "r", encoding="utf-8") as fh:
                repos = json.load(fh)
            return cls(repos)

        if ext == ".parquet":
            import pandas as pd

            df = pd.read_parquet(p)
            repos = df.to_dict(orient="records")
            return cls(repos)

        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .json or .parquet."
        )

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Convert to a pandas ``DataFrame``.

        Flattens nested fields:

        * ``topics`` list -> ``topics_str`` (comma-separated) and
          ``topics_list`` (kept as Python list for easy explosion).
        * Date-like columns (``created_at``, ``updated_at``, ``pushed_at``)
          are converted to ``datetime64``.

        The result is cached; subsequent calls return the same object.
        """
        if self._df is not None:
            return self._df

        import pandas as pd

        df = pd.json_normalize(self.repos, sep="_")

        # -- flatten topics --
        if "topics" in df.columns:
            df["topics_list"] = df["topics"].apply(
                lambda v: v if isinstance(v, list) else []
            )
            df["topics_str"] = df["topics_list"].apply(
                lambda lst: ",".join(str(t) for t in lst)
            )

        # -- convert date columns --
        date_cols = [c for c in df.columns if c.endswith("_at") or c == "created_at"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        self._df = df
        return df

    def to_numpy(self, columns: Optional[List[str]] = None):
        """Return a numpy array of the requested (numeric) columns.

        Parameters
        ----------
        columns:
            Column names to include.  If ``None``, all numeric columns
            from the DataFrame are used.
        """
        import numpy as np

        df = self.to_dataframe()

        if columns is not None:
            return df[columns].to_numpy()

        numeric = df.select_dtypes(include=[np.number])
        return numeric.to_numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the dataset to a local file.

        Parameters
        ----------
        path:
            Destination path.  Extension determines format:
            ``.json`` writes raw repo dicts;
            ``.parquet`` writes a DataFrame (requires *pyarrow*).
        """
        p = Path(path)
        ext = p.suffix.lower()
        p.parent.mkdir(parents=True, exist_ok=True)

        if ext == ".json":
            if not self.repos:
                print("[RepoDataset] Dataset is empty, nothing to save.")
                return
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self.repos, fh, indent=2, default=str)
            print(f"[RepoDataset] Saved {len(self.repos)} repos to {p}")
            return

        if ext == ".parquet":
            if not self.repos:
                print("[RepoDataset] Dataset is empty, nothing to save.")
                return
            import io as _io
            import pyarrow.json as _pa_json
            import pyarrow.parquet as _pa_pq

            rows = [
                {k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
                 for k, v in repo.items()}
                for repo in self.repos
            ]
            jsonl = "\n".join(json.dumps(row, default=str) for row in rows)
            buf = _io.BytesIO(jsonl.encode())
            table = _pa_json.read_json(buf)
            _pa_pq.write_table(table, str(p))
            print(f"[RepoDataset] Saved {len(self.repos)} repos to {p}")
            return

        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .json or .parquet."
        )

    # ------------------------------------------------------------------
    # Iteration / collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.repos)

    def __getitem__(self, index):
        return self.repos[index]

    def __repr__(self) -> str:
        return f"RepoDataset(n={len(self.repos)})"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print basic descriptive statistics about the dataset."""
        n = len(self.repos)
        print(f"Repositories: {n}")

        if n == 0:
            return

        # -- language distribution (top 10) --
        langs: Dict[str, int] = {}
        for repo in self.repos:
            lang = repo.get("language") or "Unknown"
            langs[lang] = langs.get(lang, 0) + 1

        top_langs = sorted(langs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop 10 languages:")
        for lang, count in top_langs:
            pct = count / n * 100
            print(f"  {lang}: {count} ({pct:.1f}%)")

        # -- star stats --
        stars = [
            repo.get("stargazers_count", 0) or repo.get("stars", 0)
            for repo in self.repos
        ]
        stars_clean = [s for s in stars if isinstance(s, (int, float))]
        if stars_clean:
            print(f"\nStars:")
            print(f"  min: {min(stars_clean):,}")
            print(f"  max: {max(stars_clean):,}")
            print(f"  mean: {sum(stars_clean) / len(stars_clean):,.0f}")
            sorted_stars = sorted(stars_clean)
            median = sorted_stars[len(sorted_stars) // 2]
            print(f"  median: {median:,}")

        # -- topics --
        topic_counts: Dict[str, int] = {}
        repos_with_topics = 0
        for repo in self.repos:
            topics = repo.get("topics", [])
            if topics:
                repos_with_topics += 1
                for t in topics:
                    topic_counts[t] = topic_counts.get(t, 0) + 1

        print(f"\nTopics:")
        print(f"  repos with topics: {repos_with_topics} ({repos_with_topics / n * 100:.1f}%)")
        print(f"  unique topics: {len(topic_counts)}")
        if topic_counts:
            top_topics = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("  top 10:")
            for topic, count in top_topics:
                print(f"    {topic}: {count}")

        # -- date range --
        date_field = "created_at"
        dates = [repo.get(date_field) for repo in self.repos if repo.get(date_field)]
        if dates:
            dates_str = sorted(str(d) for d in dates)
            print(f"\nDate range ({date_field}):")
            print(f"  earliest: {dates_str[0]}")
            print(f"  latest:   {dates_str[-1]}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_total_count(endpoint: str, headers: Dict[str, str]) -> int:
        """Query the gateway for the total document count.

        Handles the known count-API bug: when the reported count is <= 1
        we return float("inf") and rely on empty-batch detection to stop
        pagination — no arbitrary ceiling needed.
        """
        import requests

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json={
                    "database": "gitquery",
                    "collection": "repositories",
                    "filter": {},
                    "limit": 1,
                    "skip": 0,
                },
                timeout=10,
            )
            resp.raise_for_status()
            count = resp.json().get("count", 0)

            if count <= 1:
                print(
                    f"[RepoDataset] Count API returned {count} -- "
                    "using fallback (will stop on empty batch)."
                )
                return float("inf")

            return count

        except Exception as exc:
            print(f"[RepoDataset] Count request failed ({exc}), using fallback.")
            return 999999
