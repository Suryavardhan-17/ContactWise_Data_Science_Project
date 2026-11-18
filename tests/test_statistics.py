from contactwise.statistics import wilson_interval, export_descriptive_statistics


def test_wilson_interval_with_bounds():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert 0 <= lo <= hi <= 1


def test_statistics_exports(tiny_campaigns, tmp_path):
    export_descriptive_statistics(tiny_campaigns, tmp_path)
    assert (tmp_path / "segment_uncertainty.csv").exists()
    assert (tmp_path / "statistical_checks.json").exists()
