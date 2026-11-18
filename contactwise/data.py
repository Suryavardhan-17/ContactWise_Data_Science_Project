"""Data loading, schema checks, leakage-safe feature creation, time proxy split."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                     SPLIT_FRACTIONS, TARGET)

EXPECTED_COLUMNS = {
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
}
SPLIT_NAMES = ("train", "calibration", "selection", "test")


def find_csv_in_zip(raw_zip_bytes: bytes) -> bytes:
    """Read only target CSV (direct ZIP or an outer ZIP containing another ZIP)."""
    with zipfile.ZipFile(io.BytesIO(raw_zip_bytes)) as zf:
        candidates = [x for x in zf.namelist() if x.endswith("bank-additional-full.csv")]
        if candidates:
            return zf.read(candidates[0])
        for entry in zf.namelist():
            if entry.lower().endswith(".zip"):
                try:
                    return find_csv_in_zip(zf.read(entry))
                except (ValueError, zipfile.BadZipFile):
                    pass
    raise ValueError("Could not find bank-additional-full.csv inside UCI archive")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m contactwise.download")
    raw = pd.read_csv(path, sep=";")
    missing = EXPECTED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Dataset has no rows")
    if raw[TARGET].isna().any() or not raw[TARGET].isin(["yes", "no"]).all():
        raise ValueError("Target must be non-null and contain only 'yes'/'no'")
    return raw


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-contact-only X and refuse any unexpected feature columns."""
    needed = set(CATEGORICAL_FEATURES) | {"age", "previous", "pdays"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Feature generation missing columns: {sorted(missing)}")
    x = df[list(CATEGORICAL_FEATURES) + ["age", "previous"]].copy()
    days = pd.to_numeric(df["pdays"], errors="coerce")
    # In this version of UCI data, 999 means not previously contacted.
    x["previously_contacted"] = ((days != 999) & days.notna()).astype(int)
    x["days_since_previous_contact"] = days.mask(days == 999, np.nan)
    for col in ("age", "previous", "days_since_previous_contact"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].fillna("missing").astype(str)
    # Explicit whitelist prevents post-call features entering either model.
    assert set(x.columns) == set(FEATURES)
    return x[FEATURES]


def target_as_binary(raw: pd.DataFrame) -> pd.Series:
    return raw[TARGET].map({"no": 0, "yes": 1}).astype(int)


def chronological_slices(n: int) -> dict[str, slice]:
    """Contiguous row-order splits; dates themselves aren't provided by UCI."""
    if n < 80:
        raise ValueError("Need at least 80 records for four ordered partitions")
    a = int(n * SPLIT_FRACTIONS[0])
    b = int(n * sum(SPLIT_FRACTIONS[:2]))
    c = int(n * sum(SPLIT_FRACTIONS[:3]))
    return dict(zip(SPLIT_NAMES, (slice(0, a), slice(a, b), slice(b, c), slice(c, n))))


def add_split_column(raw: pd.DataFrame) -> pd.DataFrame:
    tagged = raw.copy()
    tagged["source_row"] = np.arange(len(raw))
    for name, sl in chronological_slices(len(raw)).items():
        tagged.loc[sl.start:sl.stop-1, "dataset_split"] = name
    return tagged
