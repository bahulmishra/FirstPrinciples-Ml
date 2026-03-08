"""
DataIngestor — Tier 1 of the FirstPrinciple ML pipeline.

Supports:
  • CSV  (file bytes or file path)
  • JSON (file bytes or file path)
  • Remote URL (auto-detects CSV vs JSON from Content-Type / filename)
"""
import io
import json
from typing import Union

import httpx
import pandas as pd


class DataIngestor:
    """Load data from various sources into a Pandas DataFrame."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(self, source: Union[bytes, str]) -> pd.DataFrame:
        """Load a CSV from raw bytes or a file path string."""
        if isinstance(source, bytes):
            return pd.read_csv(io.BytesIO(source))
        return pd.read_csv(source)

    def load_json(self, source: Union[bytes, str]) -> pd.DataFrame:
        """Load a JSON from raw bytes or a file path string."""
        if isinstance(source, bytes):
            data = json.loads(source.decode("utf-8"))
        else:
            with open(source, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            # Support {records: [...]} wrapper or flat key→column dicts
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"])
            return pd.DataFrame(data)
        raise ValueError("Unsupported JSON structure. Expected array or object.")

    def load_url(self, url: str) -> pd.DataFrame:
        """Fetch a remote CSV or JSON and return as DataFrame."""
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        lower_url = url.lower()

        if "json" in content_type or lower_url.endswith(".json"):
            return self.load_json(response.content)
        # Default: try CSV
        return self.load_csv(response.content)

    def get_columns(self, df: pd.DataFrame) -> list:
        """Return column names and dtype info for the feature mapper."""
        return [
            {"name": col, "dtype": str(df[col].dtype), "sample": self._safe_sample(df, col)}
            for col in df.columns
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_sample(df: pd.DataFrame, col: str, n: int = 3) -> list:
        """Return up to n non-null sample values from a column."""
        return df[col].dropna().head(n).astype(str).tolist()
