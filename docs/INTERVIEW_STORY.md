# How to explain ContactWise in an interview

## Two-minute story

**Context:** A bank must decide whom to contact first with a limited outreach
budget. Ranking customers is more useful than just predicting a binary label.

**Problem framing:** I defined a strict pre-contact decision moment. That
exposes a common mistake: call duration predicts outcomes well but is not
available before the call. I removed such variables before modeling.

**Data and methods:** I used the UCI Bank Marketing dataset, SQL for historical
segmentation and quality checks, Pandas for preprocessing, and a smoothed
segment baseline plus Logistic Regression and Random Forest. I designed
chronological row-order partitions for fitting, probability calibration, model
selection, and final evaluation.

**Validation:** I selected the champion using precision at a fixed contact
budget and checked average precision, Brier score and out-of-period drift.
I reported uncertainty and compared the model with a simple benchmark.

**Business interpretation:** Rankings can help prioritize records from a
previously contacted cohort, but they cannot prove that calling customers
increases subscriptions. My cost table is expressly a scenario proxy, not
verified incremental ROI.

**Next step:** A future test would randomize contact among eligible customers,
track actual intervention costs, customer IDs, and treatment effects.

## Questions a recruiter might ask

**Why remove duration?** It is measured during a call, so it is unavailable at
scoring time; including it gives a misleading retrospective advantage.

**Why precision@500 rather than accuracy?** The bank can act on only 500 records
in the evaluation cohort; positive conversion prevalence makes overall accuracy
unhelpful for ranking.

**Why not stratify randomly?** It would mix chronologically ordered campaign
records, potentially exaggerating future performance.

**Why not claim higher revenue?** Labeled positive outcomes in historical calls
are not counterfactual business lift. Intervention effect requires experimental
or causal evidence.

**How did you prevent overfitting?** Separate train/calibration/selection/test
periods, train-only preprocessing, fixed candidate list, and no test-driven
changes. Repeated clients without IDs remain a limitation.

**What would make this better?** Customer IDs, exact dates, additional
prospective features, treatment/control assignments, actual unit economics,
and subgroup stability tests.

## Before adding this to your resume

Insert only factual results after running the real UCI dataset. Example
structure (placeholders must be replaced):

> Built a leakage-aware bank outreach ranking framework with Python, SQL,
> Logistic Regression and Random Forest; evaluated **[real precision@K]** and
> **[real AP]** on a chronological holdout against a segment-rate baseline.

Do not claim deployed production models, real revenue growth, or uplift.
