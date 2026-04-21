#!/usr/bin/env python3
"""
Inspect a product-level feature database.

The output is meant for quick checks during the midpoint stage: row counts,
label balance and feature summaries.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a product-level feature SQLite DB."
    )
    parser.add_argument(
        "--db-path",
        default="data/product_recall_features_2004_2005.db",
        help="Path to the product feature SQLite database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of example positive rows to print.",
    )
    return parser.parse_args()


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def compact_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    if not db_path.is_file():
        print(f"Error: DB not found: {db_path}")
        return 1

    conn = connect_read_only(db_path)
    try:
        print(f"Database: {db_path}")
        print(f"File size: {db_path.stat().st_size / (1024 * 1024):.2f} MB")

        print_section("Dataset Metadata")
        metadata = conn.execute(
            "SELECT key, value FROM dataset_metadata ORDER BY key"
        ).fetchall()
        print_json({row["key"]: row["value"] for row in metadata})

        print_section("Label Distribution")
        counts = conn.execute(
            """
            SELECT
                COUNT(*) AS total_products,
                SUM(label_recalled) AS positive_products,
                COUNT(*) - SUM(label_recalled) AS negative_products,
                AVG(label_recalled) AS positive_rate
            FROM product_features
            """
        ).fetchone()
        print_json(compact_row(counts))

        print_section("Feature Summary")
        summary = conn.execute(
            """
            SELECT
                MIN(faers_report_count_total) AS min_report_count,
                AVG(faers_report_count_total) AS avg_report_count,
                MAX(faers_report_count_total) AS max_report_count,
                AVG(serious_rate) AS avg_serious_rate,
                AVG(death_rate) AS avg_death_rate,
                AVG(hospitalization_rate) AS avg_hospitalization_rate,
                AVG(faers_report_growth_365d) AS avg_report_growth_365d
            FROM product_features
            """
        ).fetchone()
        print_json(compact_row(summary))

        print_section(f"Positive Examples (LIMIT {args.limit})")
        positives = conn.execute(
            """
            SELECT
                entity_key,
                label_recalled,
                faers_report_count_total,
                faers_report_count_last_365d,
                faers_report_growth_365d,
                serious_rate,
                death_rate,
                hospitalization_rate,
                report_count_per_active_month
            FROM product_features
            WHERE label_recalled = 1
            ORDER BY faers_report_count_total DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()
        for idx, row in enumerate(positives, start=1):
            print(f"Positive {idx}:")
            print_json(compact_row(row))

        print_section("Evaluation Notes")
        print(
            "The product-level feature table is available for model testing. "
            "The positive class is small and the final evaluation should report "
            "PR-AUC, ROC-AUC, F1-score and Recall@Top-K% rather than accuracy only."
        )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
