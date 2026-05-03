"""
Data Version Tracker — Lightweight Dataset Provenance (GAP-03)

Provides hash-based data versioning without requiring DVC or lakeFS
infrastructure. Every time the training pipeline runs, this module
computes a SHA-256 fingerprint of the training data and logs it alongside
the model artifacts for full reproducibility.

Usage:
    from src.data_version import DataVersionTracker
    tracker = DataVersionTracker()
    version = tracker.record_version(df, source="postgresql://...")
    tracker.verify_version(df, expected_hash=version["data_hash"])
"""
import hashlib
import json
import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class DataVersionTracker:
    """Track dataset versions through content hashing.

    Stores version history in a JSON file for lightweight provenance
    without external infrastructure. Each version records:
    - Content hash (SHA-256 of sorted, serialized DataFrame)
    - Row count, column count, column names
    - Source URI (where data came from)
    - Timestamp
    """

    def __init__(self, history_path: Path | None = None):
        base = Path(__file__).resolve().parent.parent
        self.history_path = history_path or base / "models" / "data_versions.json"
        self._history: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load existing version history from disk."""
        if self.history_path.exists():
            with open(self.history_path) as f:
                self._history = json.load(f)

    def _save_history(self) -> None:
        """Persist version history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    @staticmethod
    def compute_hash(df: pd.DataFrame) -> str:
        """Compute a deterministic SHA-256 hash of a DataFrame.

        Sorts by all columns first to ensure order-independence,
        then hashes the CSV representation for reproducibility.
        """
        sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()

    def record_version(
        self,
        df: pd.DataFrame,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a new data version and append to history.

        Args:
            df: The training DataFrame to fingerprint.
            source: URI or description of where the data came from.
            metadata: Optional additional metadata to store.

        Returns:
            Version record dict with hash, stats, and timestamp.
        """
        data_hash = self.compute_hash(df)

        version = {
            "data_hash": data_hash,
            "data_hash_short": data_hash[:12],
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "source": source,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "attrition_rate": round(float(df["Attrition"].mean()), 4) if "Attrition" in df.columns else None,
        }
        if metadata:
            version["metadata"] = metadata

        # Check if this exact hash already exists
        existing_hashes = {v["data_hash"] for v in self._history}
        if data_hash in existing_hashes:
            print(f"  Data version {data_hash[:12]} already recorded (no change).")
        else:
            self._history.append(version)
            self._save_history()
            print(f"  New data version recorded: {data_hash[:12]}")
            print(f"  History: {len(self._history)} versions tracked")

        return version

    def verify_version(self, df: pd.DataFrame, expected_hash: str) -> bool:
        """Verify that a DataFrame matches an expected hash.

        Raises ValueError if the hashes don't match, indicating
        data has changed unexpectedly.
        """
        actual_hash = self.compute_hash(df)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Data version mismatch! "
                f"Expected {expected_hash[:12]}, got {actual_hash[:12]}. "
                f"The dataset has changed since the model was trained."
            )
        return True

    def get_latest(self) -> dict[str, Any] | None:
        """Return the most recent version record, or None if empty."""
        return self._history[-1] if self._history else None

    def get_history(self) -> list[dict[str, Any]]:
        """Return full version history."""
        return self._history.copy()
