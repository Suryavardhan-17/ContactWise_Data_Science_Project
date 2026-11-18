from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated"

TARGET = "y"
# The supplied UCI rows are in chronological campaign-contact order.
# This is a *row-order* proxy, not a reconstruction of exact daily periods.
SPLIT_FRACTIONS = (0.50, 0.15, 0.15, 0.20)  # train, calibrate, choose, final test

# Only prior customer/campaign information; unavailable/current-call data excluded.
CATEGORICAL_FEATURES = (
    "job", "marital", "education", "default", "housing", "loan", "poutcome"
)
NUMERIC_FEATURES = ("age", "previous", "days_since_previous_contact", "previously_contacted")
FEATURES = list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)

# Excluded deliberately. Some are only known *after* a call. Date/time and
# macro indicators are excluded from this conservative pre-call model.
UNSAFE_OR_EXCLUDED = (
    "duration", "campaign", "contact", "month", "day_of_week",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
)
