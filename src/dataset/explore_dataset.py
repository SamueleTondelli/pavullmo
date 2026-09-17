"""Profile a local Parquet text dataset without modifying the source file.

This is intentionally a small, readable Phase 3 exploration tool. It adds
document-level measurements to a separate Parquet file, uses DuckDB for the
aggregations, and writes a handful of plots and CSV tables for inspection.
Nothing in this module decides whether a document should be removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

import duckdb

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "artifacts" / "fineweb2_sample.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "exploration" / "fineweb2"
WORD_PATTERN = re.compile(r"\S+")

# These are inspection flags, not filtering decisions. Their purpose is to
# surface unusual examples so that thresholds can be reviewed by a person.
WARNING_THRESHOLDS = {
    "short_document": ("word_count", "<", 100),
    "very_long_document": ("word_count", ">", 5_000),
    "many_digits": ("digit_ratio", ">", 0.20),
    "many_symbols": ("symbol_ratio", ">", 0.20),
    "repeated_lines": ("repeated_line_ratio", ">", 0.20),
    "borderline_language": ("language_score", "<", 0.995),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a Parquet text sample and generate local reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"source Parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing exploration directory",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def repeated_line_ratio(text: str) -> float:
    """Fraction of nonempty line occurrences that repeat an earlier line."""

    normalized_lines = [
        " ".join(line.split()).casefold()
        for line in text.splitlines()
        if line.strip()
    ]
    return safe_ratio(
        len(normalized_lines) - len(set(normalized_lines)),
        len(normalized_lines),
    )


def url_domain(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return (urlsplit(value).hostname or "").casefold()
    except ValueError:
        return ""


def measure_document(text: str, url: object) -> dict[str, int | float | str]:
    characters = len(text)
    alphabetic = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    uppercase = sum(character.isupper() for character in text)
    symbols = sum(
        not character.isalnum() and not character.isspace() for character in text
    )
    return {
        "character_count": characters,
        "word_count": len(WORD_PATTERN.findall(text)),
        "line_count": max(1, len(text.splitlines())),
        "url_domain": url_domain(url),
        "digit_ratio": safe_ratio(digits, characters),
        "uppercase_ratio": safe_ratio(uppercase, alphabetic),
        "symbol_ratio": safe_ratio(symbols, characters),
        "repeated_line_ratio": repeated_line_ratio(text),
    }


def add_profile_columns(table: pa.Table) -> pa.Table:
    if "text" not in table.column_names:
        raise ValueError("input Parquet file must contain a 'text' column")

    texts = table.column("text").to_pylist()
    urls = (
        table.column("url").to_pylist()
        if "url" in table.column_names
        else [None] * len(texts)
    )
    measurements = [
        measure_document(text or "", url)
        for text, url in tqdm(
            zip(texts, urls, strict=True),
            total=len(texts),
            desc="Measuring documents",
            unit="docs",
        )
    ]
    for name in measurements[0] if measurements else ():
        table = table.append_column(
            name,
            pa.array(measurement[name] for measurement in measurements),
        )
    return table


def write_csv(path: Path, headers: list[str], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def query_rows(
    connection: duckdb.DuckDBPyConnection, query: str
) -> tuple[list[str], list[tuple[object, ...]]]:
    result = connection.execute(query)
    headers = [column[0] for column in result.description]
    return headers, result.fetchall()


def warning_expression(name: str) -> str:
    field, operator, threshold = WARNING_THRESHOLDS[name]
    return f"{field} {operator} {threshold}"


def make_plots(connection: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    word_counts = [
        row[0]
        for row in connection.execute(
            "SELECT word_count FROM documents WHERE word_count > 0"
        ).fetchall()
    ]
    upper_limit = connection.execute(
        "SELECT quantile_cont(word_count, 0.99) FROM documents"
    ).fetchone()[0]
    central_counts = [value for value in word_counts if value <= upper_limit]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(central_counts, bins=50, color="#3572A5", edgecolor="white")
    axis.set_title("Document length distribution (largest 1% excluded)")
    axis.set_xlabel("Words per document")
    axis.set_ylabel("Documents")
    figure.tight_layout()
    figure.savefig(output_dir / "document_lengths.png", dpi=160)
    plt.close(figure)

    domain_rows = connection.execute(
        """
        SELECT coalesce(nullif(url_domain, ''), '(missing)') AS domain, count(*) AS n
        FROM documents
        GROUP BY domain
        ORDER BY n DESC, domain
        LIMIT 20
        """
    ).fetchall()
    figure, axis = plt.subplots(figsize=(9, 6))
    labels = [row[0] for row in reversed(domain_rows)]
    counts = [row[1] for row in reversed(domain_rows)]
    axis.barh(labels, counts, color="#4C956C")
    axis.set_title("Twenty most frequent domains")
    axis.set_xlabel("Documents")
    figure.tight_layout()
    figure.savefig(output_dir / "top_domains.png", dpi=160)
    plt.close(figure)

    language_scores = [
        row[0]
        for row in connection.execute(
            "SELECT language_score FROM documents WHERE language_score IS NOT NULL"
        ).fetchall()
    ]
    if language_scores:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(language_scores, bins=40, color="#7B2CBF", edgecolor="white")
        axis.set_title("FineWeb2 Italian language confidence")
        axis.set_xlabel("GlotLID language score")
        axis.set_ylabel("Documents")
        figure.tight_layout()
        figure.savefig(output_dir / "language_scores.png", dpi=160)
        plt.close(figure)

    warning_rows = []
    total = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    for name in WARNING_THRESHOLDS:
        count = connection.execute(
            f"SELECT count(*) FROM documents WHERE {warning_expression(name)}"
        ).fetchone()[0]
        warning_rows.append((name, count, 100 * safe_ratio(count, total)))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    names = [row[0].replace("_", " ") for row in reversed(warning_rows)]
    percentages = [row[2] for row in reversed(warning_rows)]
    axis.barh(names, percentages, color="#D97706")
    axis.set_title("Exploratory warning rates (not rejection rates)")
    axis.set_xlabel("Documents flagged (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "warning_rates.png", dpi=160)
    plt.close(figure)

    write_csv(
        output_dir / "warning_rates.csv",
        ["warning", "documents", "percentage"],
        warning_rows,
    )


def build_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT
            count(*) AS documents,
            sum(word_count) AS words,
            min(word_count) AS min_words,
            median(word_count) AS median_words,
            avg(word_count) AS mean_words,
            quantile_cont(word_count, 0.95) AS p95_words,
            quantile_cont(word_count, 0.99) AS p99_words,
            max(word_count) AS max_words,
            count(DISTINCT nullif(url_domain, '')) AS domains,
            avg(language_score) AS mean_language_score,
            min(language_score) AS min_language_score
        FROM documents
        """
    ).fetchone()
    names = [column[0] for column in connection.description]
    return {
        name: round(value, 6) if isinstance(value, float) and math.isfinite(value) else value
        for name, value in zip(names, row, strict=True)
    }


def explore(input_path: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset not found: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite to replace its reports"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    table = add_profile_columns(pq.read_table(input_path))
    profiled_path = output_dir / "profiled_documents.parquet"
    pq.write_table(table, profiled_path, compression="zstd")

    connection = duckdb.connect()
    connection.from_parquet(str(profiled_path)).create_view("documents")
    try:
        summary = build_summary(connection)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        headers, rows = query_rows(
            connection,
            """
            SELECT coalesce(nullif(url_domain, ''), '(missing)') AS domain,
                   count(*) AS documents,
                   sum(word_count) AS words
            FROM documents
            GROUP BY domain
            ORDER BY documents DESC, domain
            LIMIT 100
            """,
        )
        write_csv(output_dir / "top_domains.csv", headers, rows)

        headers, rows = query_rows(
            connection,
            """
            SELECT id, url, word_count, character_count,
                   substr(replace(text, chr(10), ' '), 1, 240) AS preview
            FROM documents
            ORDER BY word_count DESC
            LIMIT 100
            """,
        )
        write_csv(output_dir / "longest_documents.csv", headers, rows)
        make_plots(connection, output_dir)
    finally:
        connection.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nReports written to {output_dir}")


def main() -> None:
    args = parse_args()
    explore(args.input, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
