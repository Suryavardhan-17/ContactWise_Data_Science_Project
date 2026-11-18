# ContactWise — Who Should a Bank Contact First?

### A reproducible Data Science portfolio case study | Python · SQL · Statistics · Scikit-learn

**Business question:** A bank can contact only a limited number of records from its
historical campaign pool. Can we build a defensible *pre-contact ranking* that
prioritizes likely subscriptions, while accounting for time drift, uncertainty,
and marketing costs?

> **Status: Source code and tests complete.** Real-data benchmark figures and charts
> are generated only when you download and run the UCI dataset. No model accuracy,
> conversion lift, or financial benefit is invented in this repository.

## The distinctive parts

Many bank-marketing tutorials predict a yes/no outcome using random train/test
splits and `duration` (the length of the phone call). ContactWise instead:

1. **Defines the actual decision moment:** before the bank makes a call.
2. **Blocks post-call leakage:** excludes `duration`, current campaign call count,
   and other current-contact fields from the scoring model.
3. **Uses row-order temporal evaluation:** oldest 50% train, next 15% probability
   calibration, next 15% model selection, newest 20% untouched holdout.
4. **Benchmarks simple and complex approaches:** historical segment-rate baseline,
   Logistic Regression, Random Forest; model selection does not use holdout results.
5. **Optimizes an operational constraint:** precision and observed positives among
   the *top 500 ranked test records*, rather than overall accuracy alone.
6. **Checks model confidence:** Platt/sigmoid probability calibration, Brier score,
   and decile calibration plot.
7. **Does statistical and SQL analysis:** segment uncertainty with 95% Wilson
   intervals, chi-square association check, SQL segment summaries, data quality,
   and age-group diagnostic differences.
8. **Separates predictive association from causal impact:** no false ROI claims.

This is a retrospective case study, **not** a production marketing tool.

## Business framing

| Decision | Constraint | Observable metric |
|---|---|---|
| Which customer records should we contact? | Fixed top-K budget (500 records by default) | Precision@K, lift@K, ranking gain |
| Which model is trustworthy? | Old data may not match new periods | Out-of-time AP, Brier, calibration plot |
| Is a model better than simple rules? | Operational simplicity matters | Segment-rate baseline comparison |
| Do cost assumptions change the decision? | Historical data lacks economic outcomes | Explicit *illustrative* scenario table |

**Important:** “500 contacts” is a limit over the *evaluation cohort*, **not
500 calls per day**. The UCI file does not provide exact day-level timestamps
or unique customer IDs for a realistic daily policy simulation.

## Source data

**UCI Bank Marketing**, specifically `bank-additional-full.csv` (41,188 rows,
20 predictors and one `y` target, source records ordered by campaign contact date,
from May 2008 to November 2010).

- Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Related reference: S. Moro, P. Cortez, P. Rita (2014), *A Data-Driven Approach
  to Predict the Success of Bank Telemarketing*, Decision Support Systems.
- UCI license: **Creative Commons Attribution 4.0**. Cite the creators if using
  or redistributing the data.

The dataset is intentionally downloaded by the user; source records are not
committed in this repository. See [`data/README.md`](data/README.md).

## Reproduce the project

### Requirements

Python **3.10+** (3.11 or 3.12 recommended), internet access for the one-time
UCI download, and a typical laptop. No API keys, paid cloud resources, or web
servers required.

```bash
# 1. Clone the repository (replace YOUR_USERNAME after you upload it)
git clone https://github.com/YOUR_USERNAME/ContactWise-Data-Science.git
cd ContactWise-Data-Science

# 2. Create isolated environment
python -m venv .venv

# 3a. macOS / Linux
source .venv/bin/activate

# 3b. Windows PowerShell (run instead of 3a)
# .venv\Scripts\Activate.ps1

# 4. Install packages
python -m pip install -r requirements.txt

# 5. Download official UCI data
python -m contactwise.download

# 6. Run checks
python -m pytest -q

# 7. Generate real analysis and metrics
python -m contactwise.run --budget 500 --cost 5 --value 100
```

If you have downloaded the raw CSV manually, put it at
`data/raw/bank-additional-full.csv` and skip the download step. You can use
`--input PATH` and `--output PATH` with the runner.

**Troubleshooting:** On Windows some machines use `py` rather than `python`;
start with `py -m venv .venv`. If PowerShell blocks environment activation,
run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and the
same path for each `-m ...` command instead of changing global permissions.

## Project layout

```text
ContactWise-Data-Science/
├── contactwise/
│   ├── download.py       # UCI download, ZIP validation, no raw data in Git
│   ├── data.py           # schema checks, chronological splits, safe features
│   ├── models.py         # segment baseline, Logistic Regression, Random Forest
│   ├── metrics.py        # precision@K, lift@K, illustrative scenario metrics
│   ├── sql_analysis.py   # SQLite execution of checked-in SQL
│   ├── statistics.py     # Wilson intervals, descriptive chi-square, group audit
│   └── run.py            # end-to-end reproducible runner and visuals
├── sql/                  # four real SQL analysis queries
├── tests/                # deterministic offline unit tests
├── docs/                 # decisions, project plan, interview story
├── data/                 # provenance; raw CSV downloaded separately
├── reports/              # generated artifacts are gitignored by default
├── requirements.txt
├── .gitignore
└── README.md
```

## Validated modeling protocol

1. Check input schema and keep UCI row ordering intact.
2. Engineer a conservative list of features available *before* the phone call:
   `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`,
   `previous`, `poutcome`, and previous-campaign recency indicators.
3. Exclude **call duration**, current campaign contact count, current contact
   channel/date, and contemporaneous economic series in this base experiment.
   This is a deliberately restricted model, not a claim that such fields are
   always unusable.
4. Fit encoders, imputers, and model parameters only on the **train** period.
5. Calibrate trained classifier probabilities on the **calibration** period.
6. Pick the winner on the **selection** period using precision@K (AP as tie-break).
7. Score the newest 20% **once**, compare to a fixed segment-rate benchmark,
   and audit historical outcome shift and calibration.
8. Create scenario calculations with user-supplied assumptions. **Do not infer
   incremental campaign revenue or ad effectiveness without randomized data**.

> Time caveat: UCI confirms the full file is ordered by date, but does not
> release exact per-row timestamps, so contiguous source-row blocks are an
> imperfect *temporal proxy*. Also, multiple contacts for one client may
> appear without unique client IDs, so patient-/customer-level group leakage
> cannot be ruled out.

## Generated results (only after running)

`reports/generated/` will contain:

| File | Why it matters |
|---|---|
| `executive_summary.md` | Actual results, business interpretation and caveats |
| `metrics.json` | Auditable model-selection and holdout metrics |
| `model_selection.csv` | Shows how winner was selected without peeking at test |
| `test_ranked.csv` | Probability, rank, inclusion in top-K, observed label |
| `ranking_gain.png` | How many historical positives accumulate through ranking |
| `calibration.png` | Predicted versus observed risk by probability decile |
| `time_proxy_drift.png` | Evidence of changing positive rates over time |
| `segment_uncertainty.csv` | Descriptive 95% confidence intervals for occupations |
| `statistical_checks.json` | Exploratory chi-square and assumptions |
| `age_subgroup_audit.csv` | Diagnostic disparities (not fairness certification) |
| `scenario_sensitivity.csv` | Sensitivity to assumed value/cost, not ROI |
| `01_...04_*.csv` | Outputs from checked-in SQLite queries |
| `champion_model.joblib` | Fitted artifact — load only from trusted source |

Review generated metrics and **only then** consider committing selected charts
and an accurate findings summary to GitHub. All `.png`, CSV, and JSON figures
must be generated from the actual historical dataset; example numbers are not
presented as real findings.

## Constraints, edge cases, and responsible interpretation

- **No counterfactual:** All records relate to historical telemarketing;
  predicted subscription does **not** imply a call *caused* subscription.
- **Selection bias:** The dataset is composed of previously contacted records,
  not every eligible customer who might be called next month.
- **Label drift:** Response rates change sharply across campaign periods;
  rankings may underperform in unseen economic conditions.
- **Repeated people:** No unique customer ID; cannot enforce household/customer
  separation across splits.
- **Class imbalance:** Accuracy is not a sufficient metric; use AP and top-K.
- **Missingness:** `unknown` is retained as a category; `pdays=999` is treated
  as “not previously contacted.”
- **Sensitive features:** Age is included for retrospective diagnostic modeling
  only; any real-world use would require a lawful basis and bias assessment.
- **Costs:** Illustrative value and cost assumptions are not actual bank data.
- **External validity:** Historical Portuguese campaigns from 2008–2010 do not
  demonstrate current performance in the US or elsewhere.

## Interview story

See [`docs/INTERVIEW_STORY.md`](docs/INTERVIEW_STORY.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the trade-offs and an honest
2-minute walkthrough. See [`docs/THREE_WEEK_PLAN.md`](docs/THREE_WEEK_PLAN.md)
if you want to extend this from a reproducible core into a polished portfolio.

## License

Source **code** is released under the MIT License (`LICENSE`). The dataset is
separately licensed by UCI / its authors under CC BY 4.0 and is not included.
