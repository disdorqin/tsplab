"""Experiment tracker with local SQLite storage and optional git auto-commit."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import datetime
from typing import Any


class ExperimentTracker:
    """Track experiment parameters, metrics, and artifacts.

    Usage::

        tracker = ExperimentTracker(name="informer_etth1")
        tracker.log_params({"model": "informer", "lr": 1e-4, "bs": 32})
        tracker.log_metrics({"mae": 0.321, "smape": 12.5, "rmse": 0.45})
        tracker.save()  # Saves to SQLite + optional git commit
    """

    def __init__(
        self,
        name: str,
        storage_dir: str = "./experiments",
        auto_git: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name : experiment name
        storage_dir : where to store the SQLite database
        auto_git : if True, auto-commit after each save()
        """
        self.name = name
        self.storage_dir = storage_dir
        self.auto_git = auto_git
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.notes: str = ""
        self.created_at = datetime.datetime.now().isoformat()

        os.makedirs(storage_dir, exist_ok=True)
        self.db_path = os.path.join(storage_dir, "tsplab_experiments.db")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TEXT,
                params TEXT,
                metrics TEXT,
                notes TEXT,
                git_hash TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.metrics.update(metrics)

    def log_note(self, note: str) -> None:
        """Add a note to this experiment."""
        self.notes = note

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _git_commit(self) -> None:
        """Auto-commit experiment results."""
        try:
            subprocess.run(["git", "add", "."], capture_output=True, timeout=10)
            commit_msg = f"[tsplab] experiment: {self.name} | mae={self.metrics.get('mae', '?')}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass

    def save(self) -> int:
        """Save experiment to database. Returns experiment ID."""
        conn = sqlite3.connect(self.db_path)
        git_hash = self._get_git_hash()

        cursor = conn.execute(
            "INSERT INTO experiments (name, created_at, params, metrics, notes, git_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.name,
                self.created_at,
                json.dumps(self.params, default=str),
                json.dumps(self.metrics, default=str),
                self.notes,
                git_hash,
            ),
        )
        exp_id = cursor.lastrowid
        conn.commit()
        conn.close()

        if self.auto_git:
            self._git_commit()

        print(f"[tsplab] Experiment '{self.name}' saved (id={exp_id})")
        return exp_id

    @classmethod
    def list_experiments(cls, storage_dir: str = "./experiments") -> list[dict]:
        """List all experiments from the database."""
        db_path = os.path.join(storage_dir, "tsplab_experiments.db")
        if not os.path.exists(db_path):
            print("No experiments found.")
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        conn.close()

        experiments = []
        for row in rows:
            experiments.append({
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "params": json.loads(row["params"]),
                "metrics": json.loads(row["metrics"]),
                "notes": row["notes"],
                "git_hash": row["git_hash"],
            })
        return experiments

    @classmethod
    def compare(cls, storage_dir: str = "./experiments") -> None:
        """Print a comparison table of all experiments."""
        experiments = cls.list_experiments(storage_dir)
        if not experiments:
            return

        print("=" * 80)
        print("  TSPLab Experiment Comparison")
        print("=" * 80)
        print()

        for exp in experiments:
            print(f"  #{exp['id']} {exp['name']} ({exp['created_at'][:19]})")
            if exp["params"]:
                print(f"      params: {exp['params']}")
            if exp["metrics"]:
                metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in exp["metrics"].items())
                print(f"      metrics: {metrics_str}")
            if exp["git_hash"]:
                print(f"      git: {exp['git_hash'][:8]}")
            print()

        print("=" * 80)
