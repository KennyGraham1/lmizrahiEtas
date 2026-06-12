#!/usr/bin/env python3
"""Compare bibliography records with publisher-deposited Crossref metadata."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
BIB = HERE / "references.bib"
OUTPUT = HERE / "tables" / "reference_vor_check.csv"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.replace("--", "-")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_bibtex(text: str) -> list[dict]:
    records = []
    for block in re.split(r"(?=@article\{)", text):
        key_match = re.match(r"@article\{([^,]+),", block)
        if not key_match:
            continue
        fields = {
            match.group(1).lower(): match.group(2).replace("\n", " ").strip()
            for match in re.finditer(
                r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|\n\})", block, flags=re.S
            )
        }
        fields["key"] = key_match.group(1)
        records.append(fields)
    return records


def crossref_record(doi: str) -> dict:
    response = requests.get(
        "https://api.crossref.org/works/" + requests.utils.quote(doi, safe=""),
        timeout=30,
        headers={"User-Agent": "lmizrahiEtas-reference-audit/1.0"},
    )
    response.raise_for_status()
    return response.json()["message"]


def main() -> None:
    rows = []
    for record in parse_bibtex(BIB.read_text()):
        metadata = crossref_record(record["doi"])
        authors_bib = [
            normalize(piece.split(",")[0])
            for piece in re.split(r"\s+and\s+", record.get("author", ""))
        ]
        authors_vor = [
            normalize(author.get("family", "")) for author in metadata.get("author", [])
        ]
        published = (
            metadata.get("published-print")
            or metadata.get("published-online")
            or metadata.get("published")
        )
        year_vor = str(published["date-parts"][0][0]) if published else ""
        comparisons = {
            "title_match": normalize(record.get("title", "").replace("{", "").replace("}", ""))
            == normalize((metadata.get("title") or [""])[0]),
            "journal_match": normalize(record.get("journal", ""))
            == normalize((metadata.get("container-title") or [""])[0]),
            "year_match": record.get("year", "") == year_vor,
            "volume_match": normalize(record.get("volume", ""))
            == normalize(metadata.get("volume", "")),
            "issue_match": normalize(record.get("number", ""))
            == normalize(metadata.get("issue", "")),
            "pages_match": normalize(record.get("pages", ""))
            == normalize(metadata.get("page", "")),
            "authors_match": authors_bib == authors_vor,
        }
        rows.append({
            "key": record["key"],
            "doi": record["doi"],
            **comparisons,
            "all_registry_fields_match": all(comparisons.values()),
            "bib_authors": "; ".join(authors_bib),
            "crossref_authors": "; ".join(authors_vor),
            "version_of_record_url": metadata.get("resource", {})
            .get("primary", {})
            .get("URL", metadata.get("URL", "")),
        })
        print(
            f"{record['key']:18s} "
            f"{'PASS' if all(comparisons.values()) else 'REVIEW'}"
        )
    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
