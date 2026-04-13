"""Feature extraction for recommendation ranking models.

Provides a ``FeatureExtractor`` class that computes numeric features from
raw repository DataFrames.  Designed for use in notebooks and training
pipelines with LightGBM, XGBoost, or similar gradient-boosted models.

Expected DataFrame columns (from ``RepoDataset.to_dataframe()``):
    name, description, stars, forks, language, license, topics,
    readme, updated_at (or pushed_at).

Usage::

    from src.recommender.data import RepoDataset, FeatureExtractor

    ds = RepoDataset.from_local("repos.json")
    fe = FeatureExtractor()
    df = ds.to_dataframe()
    features = fe.extract_all(df, query="python web framework")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERMISSIVE_LICENSES = frozenset(
    [
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "0BSD",
    ]
)

TOP_N_LANGUAGES = 20

_STALE_SENTINEL_DAYS = 3650.0  # ~10 years, used for missing dates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_topic_strings(topics_value) -> list:
    """Normalize a topics field into a flat list of lowercase strings.

    Handles:
        - ``list[str]``             -> ``["python", "ml"]``
        - ``list[dict]`` with key   -> ``[{"name": "python"}, ...]``
        - ``None`` / ``NaN``        -> ``[]``
    """
    if topics_value is None:
        return []

    import pandas as _pd

    if isinstance(topics_value, float) and _pd.isna(topics_value):
        return []

    if not isinstance(topics_value, list):
        return []

    result: list[str] = []
    for item in topics_value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("topic") or ""
            result.append(str(name).lower().strip())
        else:
            result.append(str(item).lower().strip())
    return [t for t in result if t]


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Feature extraction for recommendation ranking models.

    All methods accept a pandas DataFrame whose rows represent repositories
    and return either a ``Series`` (single feature) or a ``DataFrame``
    (multi-column features such as one-hot encodings).  Missing values are
    handled gracefully with sensible defaults.

    Usage::

        fe = FeatureExtractor()
        df = dataset.to_dataframe()
        features = fe.extract_all(df)
        # features is a DataFrame ready for model.fit(features, labels)
    """

    # ------------------------------------------------------------------
    # Numeric / log-scaled
    # ------------------------------------------------------------------

    @staticmethod
    def stars_log(df: pd.DataFrame) -> pd.Series:
        """Log-scaled star count: ``log1p(stars)``.

        Compresses the heavy-tailed star distribution so that a repo with
        100 k stars does not dominate one with 1 k.

        Range: [0, inf).
        """
        import numpy as np

        return np.log1p(df["stars"].fillna(0).clip(lower=0)).rename("stars_log")

    @staticmethod
    def forks_log(df: pd.DataFrame) -> pd.Series:
        """Log-scaled fork count: ``log1p(forks)``.

        Same rationale as :meth:`stars_log`.

        Range: [0, inf).
        """
        import numpy as np

        return np.log1p(df["forks"].fillna(0).clip(lower=0)).rename("forks_log")

    @staticmethod
    def fork_star_ratio(df: pd.DataFrame) -> pd.Series:
        """Ratio of forks to stars (engagement signal).

        A high ratio suggests the project is actively forked / contributed
        to relative to passive starring.  Repos with 0 stars receive a
        ratio of 0.

        Range: [0.0, 1.0] (capped at 1).
        """
        stars = df["stars"].fillna(0).clip(lower=0)
        forks = df["forks"].fillna(0).clip(lower=0)
        ratio = forks / stars.replace(0, 1)  # avoid division by zero
        return ratio.clip(upper=1.0).rename("fork_star_ratio")

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------

    @staticmethod
    def days_since_update(df: pd.DataFrame) -> pd.Series:
        """Days since last update (freshness signal).

        Uses ``updated_at`` if available, falling back to ``pushed_at``.
        Missing dates are filled with a large sentinel value
        (3 650 days / ~10 years) so the model treats them as stale.

        Range: [0, inf).
        """
        import pandas as _pd

        now = _pd.Timestamp.utcnow()

        if "updated_at" in df.columns:
            ts = _pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
        elif "pushed_at" in df.columns:
            ts = _pd.to_datetime(df["pushed_at"], errors="coerce", utc=True)
        else:
            return _pd.Series(
                _STALE_SENTINEL_DAYS,
                index=df.index,
                name="days_since_update",
            )

        delta = (now - ts).dt.total_seconds() / 86_400.0
        return (
            delta.fillna(_STALE_SENTINEL_DAYS)
            .clip(lower=0)
            .rename(
                "days_since_update",
            )
        )

    # ------------------------------------------------------------------
    # Readme / description
    # ------------------------------------------------------------------

    @staticmethod
    def has_readme(df: pd.DataFrame) -> pd.Series:
        """Boolean (0/1): repository has a non-empty readme.

        Range: {0, 1}.
        """
        if "readme" not in df.columns:
            import pandas as _pd

            return _pd.Series(0, index=df.index, name="has_readme")

        return df["readme"].fillna("").astype(str).str.strip().str.len().gt(0).astype(int).rename("has_readme")

    @staticmethod
    def readme_length(df: pd.DataFrame) -> pd.Series:
        """Character count of the readme (documentation quality proxy).

        Longer readmes generally indicate better documentation, though the
        relationship plateaus.

        Range: [0, inf).
        """
        if "readme" not in df.columns:
            import pandas as _pd

            return _pd.Series(0, index=df.index, name="readme_length")

        return df["readme"].fillna("").astype(str).str.len().rename("readme_length")

    @staticmethod
    def description_length(df: pd.DataFrame) -> pd.Series:
        """Character count of the description.

        Repos without descriptions receive 0.

        Range: [0, inf).
        """
        return df["description"].fillna("").astype(str).str.len().rename("description_length")

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    @staticmethod
    def num_topics(df: pd.DataFrame) -> pd.Series:
        """Number of topics / tags attached to the repository.

        More topics may indicate a well-categorised repo.

        Range: [0, ~30].
        """
        if "topics" not in df.columns:
            import pandas as _pd

            return _pd.Series(0, index=df.index, name="num_topics")

        return df["topics"].apply(lambda v: len(_extract_topic_strings(v))).rename("num_topics")

    # ------------------------------------------------------------------
    # License
    # ------------------------------------------------------------------

    @staticmethod
    def has_license(df: pd.DataFrame) -> pd.Series:
        """Boolean (0/1): repository has any license declared.

        Range: {0, 1}.
        """
        return df["license"].fillna("").astype(str).str.strip().str.len().gt(0).astype(int).rename("has_license")

    @staticmethod
    def is_permissive_license(df: pd.DataFrame) -> pd.Series:
        """Boolean (0/1): license is in the permissive set.

        Permissive licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause,
        ISC, Unlicense, 0BSD.

        Range: {0, 1}.
        """
        return (
            df["license"]
            .fillna("")
            .astype(str)
            .str.strip()
            .isin(PERMISSIVE_LICENSES)
            .astype(int)
            .rename("is_permissive_license")
        )

    # ------------------------------------------------------------------
    # Language encoding
    # ------------------------------------------------------------------

    @staticmethod
    def language_encoded(
        df: pd.DataFrame,
        top_n: int = TOP_N_LANGUAGES,
    ) -> pd.DataFrame:
        """One-hot encode the primary language column.

        The *top_n* most frequent languages each get their own binary column
        (``lang_python``, ``lang_javascript``, ...).  All remaining languages
        are grouped into ``lang_other``.  Missing values map to
        ``lang_other``.

        Returns:
            DataFrame with up to ``top_n + 1`` boolean (int) columns.
        """
        import pandas as _pd

        lang = df["language"].fillna("other").astype(str).str.strip().str.lower()

        top_languages = lang.value_counts().head(top_n).index.tolist()
        lang = lang.where(lang.isin(top_languages), other="other")

        dummies = _pd.get_dummies(lang, prefix="lang", dtype=int)

        # Ensure ``lang_other`` always exists even if every repo has a
        # top-N language.
        if "lang_other" not in dummies.columns:
            dummies["lang_other"] = 0

        return dummies

    # ------------------------------------------------------------------
    # Query-dependent features
    # ------------------------------------------------------------------

    @staticmethod
    def topic_overlap(df: pd.DataFrame, query_terms: List[str]) -> pd.Series:
        """Fraction of *query_terms* found in the repository's topics.

        A score of 1.0 means every query term appears in the repo topics.

        Args:
            df: Repository DataFrame.
            query_terms: Lowercased query tokens to match against topics.

        Range: [0.0, 1.0].
        """
        import pandas as _pd

        if not query_terms:
            return _pd.Series(0.0, index=df.index, name="topic_overlap")

        if "topics" not in df.columns:
            return _pd.Series(0.0, index=df.index, name="topic_overlap")

        query_set = set(t.lower().strip() for t in query_terms)
        n_terms = len(query_set)

        def _overlap(topics_value) -> float:
            topic_strings = set(_extract_topic_strings(topics_value))
            return len(query_set & topic_strings) / n_terms

        return df["topics"].apply(_overlap).rename("topic_overlap")

    @staticmethod
    def text_match_score(df: pd.DataFrame, query: str) -> pd.Series:
        """Token overlap between query and repo name + description.

        Computes the fraction of query tokens that appear in the combined
        name and description text::

            |query_tokens & repo_tokens| / |query_tokens|

        This is intentionally simple -- data scientists can plug in TF-IDF,
        BM25, or learned scores externally.

        Args:
            df: Repository DataFrame.
            query: Raw query string.

        Range: [0.0, 1.0].
        """
        import pandas as _pd

        if not query or not query.strip():
            return _pd.Series(0.0, index=df.index, name="text_match_score")

        query_tokens = set(query.lower().split())
        n_query = len(query_tokens)

        if n_query == 0:
            return _pd.Series(0.0, index=df.index, name="text_match_score")

        name = df["name"].fillna("").astype(str).str.lower()
        desc = df["description"].fillna("").astype(str).str.lower()
        combined = name + " " + desc

        def _score(text: str) -> float:
            tokens = set(text.split())
            return len(query_tokens & tokens) / n_query

        return combined.apply(_score).rename("text_match_score")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def extract_all(
        self,
        df: pd.DataFrame,
        query: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract all numeric features into a single DataFrame.

        Returns a DataFrame with one column per feature, sharing the same
        index as the input *df*.  If *query* is provided, the
        query-dependent features ``topic_overlap`` and ``text_match_score``
        are included as well.

        Args:
            df: Repository DataFrame (from ``RepoDataset.to_dataframe()``).
            query: Optional search query for query-dependent features.

        Returns:
            Feature DataFrame ready for ``model.fit(X, y)``.
        """
        import pandas as _pd

        parts: list[_pd.DataFrame | _pd.Series] = [
            self.stars_log(df),
            self.forks_log(df),
            self.fork_star_ratio(df),
            self.days_since_update(df),
            self.has_readme(df),
            self.readme_length(df),
            self.num_topics(df),
            self.description_length(df),
            self.has_license(df),
            self.is_permissive_license(df),
            self.language_encoded(df),
        ]

        if query:
            query_terms = query.lower().split()
            parts.append(self.topic_overlap(df, query_terms))
            parts.append(self.text_match_score(df, query))

        return _pd.concat(parts, axis=1)
