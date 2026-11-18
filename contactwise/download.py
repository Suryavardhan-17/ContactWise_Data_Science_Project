"""Download the official UCI source and extract the full chronological CSV."""
from __future__ import annotations

import hashlib
import sys
import urllib.request

from .config import RAW_CSV
from .data import EXPECTED_COLUMNS, find_csv_in_zip

SOURCES = [
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
]


def download() -> None:
    if RAW_CSV.exists():
        print(f"Dataset already exists: {RAW_CSV}")
        print("To re-download, delete this CSV and run the command again.")
        return
    failures = []
    for url in SOURCES:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ContactWise-portfolio/1.0 (research)"}
            )
            with urllib.request.urlopen(request, timeout=35) as response:
                archive = response.read()
            csv_data = find_csv_in_zip(archive)
            # Quick structural check before saving.
            first = csv_data.splitlines()[0].decode("utf-8-sig").replace('"', "").split(";")
            if not EXPECTED_COLUMNS.issubset(set(first)):
                raise ValueError("Unexpected UCI dataset format")
            RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
            RAW_CSV.write_bytes(csv_data)
            print(f"Downloaded: {RAW_CSV}")
            print(f"Bytes: {len(csv_data):,}")
            print(f"SHA256: {hashlib.sha256(csv_data).hexdigest()}")
            print(f"Original UCI source: {url}")
            return
        except Exception as exc:
            failures.append(f"{url}: {exc}")
    print("Download failed. You can manually download the UCI dataset:", file=sys.stderr)
    print("https://archive.ics.uci.edu/dataset/222/bank+marketing", file=sys.stderr)
    print(f"Place bank-additional-full.csv at {RAW_CSV}", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download()
