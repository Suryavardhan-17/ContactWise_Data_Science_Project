"""Small fixture for deterministic tests; not representative of real outcomes."""
import pandas as pd
import pytest


@pytest.fixture
def tiny_campaigns():
    n = 100
    return pd.DataFrame({
        "age": [28 + i % 39 for i in range(n)],
        "job": ["admin.", "technician", "services", "unknown"] * 25,
        "marital": ["married", "single"] * 50,
        "education": ["university.degree", "high.school"] * 50,
        "default": ["no"] * n,
        "housing": ["yes", "no"] * 50,
        "loan": ["no"] * n,
        "contact": ["cellular"] * n,
        "month": ["may"] * n,
        "day_of_week": ["mon"] * n,
        "duration": [10_000 if i % 2 else 1 for i in range(n)],
        "campaign": [i % 4 + 1 for i in range(n)],
        "pdays": [999 if i % 4 == 0 else 3 for i in range(n)],
        "previous": [i % 3 for i in range(n)],
        "poutcome": ["nonexistent", "failure", "success", "nonexistent"] * 25,
        "emp.var.rate": [1.1] * n,
        "cons.price.idx": [92.2] * n,
        "cons.conf.idx": [-32.0] * n,
        "euribor3m": [4.1] * n,
        "nr.employed": [5100.0] * n,
        "y": ["no", "yes", "no", "no"] * 25,
    })
