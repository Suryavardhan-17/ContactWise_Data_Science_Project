import numpy as np

from contactwise.data import transform_features, target_as_binary
from contactwise.models import SmoothedSegmentBaseline, build_candidates


def test_baseline_predictions_valid(tiny_campaigns):
    x = transform_features(tiny_campaigns)
    y = target_as_binary(tiny_campaigns)
    model = SmoothedSegmentBaseline().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_logistic_pipeline_fit_on_tiny_fixture(tiny_campaigns):
    x = transform_features(tiny_campaigns)
    y = target_as_binary(tiny_campaigns)
    model = build_candidates()["logistic_regression"]
    model.fit(x.iloc[:80], y.iloc[:80])
    p = model.predict_proba(x.iloc[80:])[:, 1]
    assert len(p) == 20
    assert np.isfinite(p).all()
