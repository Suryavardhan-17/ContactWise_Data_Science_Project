from io import BytesIO
from zipfile import ZipFile

import numpy as np

from contactwise.config import FEATURES, UNSAFE_OR_EXCLUDED
from contactwise.data import chronological_slices, find_csv_in_zip, transform_features


def test_features_exclude_post_contact_data(tiny_campaigns):
    x = transform_features(tiny_campaigns)
    assert list(x.columns) == FEATURES
    assert not (set(x.columns) & set(UNSAFE_OR_EXCLUDED))
    assert "y" not in x.columns
    assert "duration" not in x.columns
    assert "campaign" not in x.columns


def test_previous_contact_sentinel(tiny_campaigns):
    x = transform_features(tiny_campaigns)
    assert x.loc[1, "previously_contacted"] == 1
    assert x.loc[0, "previously_contacted"] == 0
    assert np.isnan(x.loc[0, "days_since_previous_contact"])
    assert x.loc[1, "days_since_previous_contact"] == 3


def test_ordered_splits_are_disjoint_and_exhaustive():
    slices = chronological_slices(101)
    seen = []
    for sl in slices.values():
        seen.extend(range(sl.start, sl.stop))
    assert seen == list(range(101))
    assert len(slices) == 4


def make_zip(name: str, payload: bytes):
    buf = BytesIO()
    with ZipFile(buf, "w") as z:
        z.writestr(name, payload)
    return buf.getvalue()


def test_find_csv_in_nested_or_direct_zip():
    expected = b'"age";"y"\n30;"yes"\n'
    direct = make_zip("bank-additional/bank-additional-full.csv", expected)
    outer = make_zip("bank-additional.zip", direct)
    assert find_csv_in_zip(direct) == expected
    assert find_csv_in_zip(outer) == expected
