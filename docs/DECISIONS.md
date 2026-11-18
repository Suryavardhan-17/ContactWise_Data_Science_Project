# Design decisions and trade-offs

## 1. At what exact moment are predictions made?

Before a telemarketing call. We cannot rely on `duration`, because duration is
measured after the call begins. The UCI `campaign` feature includes the last
contact performed during the current campaign. `contact`, `month`, and
`day_of_week` refer to the last contact; we exclude them conservatively because
the bank may not yet know the final channel/date. Contemporaneous macro series
are excluded because real release timing and availability are not audited here.

Allowed core predictors: age and customer attributes, and **previous campaign**
history. This is an explicit whitelist in `contactwise/config.py`.

## 2. Why not a random train-test split?

The dataset's full file is ordered by campaign date; random splitting mixes
historical and later periods. We reserve the newest 20% for one-time testing,
and keep independent calibration and model-selection partitions before it.
This remains an imperfect temporal proxy because exact timestamps are absent.

## 3. Why not optimize accuracy?

A majority-class guess can achieve deceptively high accuracy when few clients
subscribe. The actual operational constraint is top-K contacts in a fixed
cohort. We select on precision@K and report average precision, ROC-AUC,
calibration (Brier), and lift to the cohort's observed base rate.

## 4. Why calibrate separately?

Risk probabilities can be biased. Sigmoid calibration is fitted on a distinct
later period, without updating the underlying classifier. A separate selection
period decides which model to report, preserving the final holdout.

## 5. Why benchmark a simple segment model?

Operational transparency matters. A smoothed occupation × prior-campaign
outcome rate is a meaningful low-complexity benchmark, and models only add
value if they produce an empirically useful ranking against it.

## 6. What does “scenario value” mean?

Exactly: `observed_positives_among_selected_records × assumed_subscription_value
− number_selected × assumed_call_cost`.

It **does not** estimate incremental gains, because positive historical labels
were observed *after contact* and the dataset has no untreated comparison. Real
incrementality requires randomization or strong causal identification.

## 7. How is uncertainty represented?

Wilson intervals are applied to unadjusted historical group conversion rates.
The exploratory chi-square association test assumes independent observations;
repeat contacts violate that assumption in potentially unknown ways. It is not
used to claim causal difference, nor to make a policy recommendation.

## 8. What would be required before deployment?

Prospective customer IDs; exact timestamps; proof every feature existed at the
scoring moment; contact eligibility/consent; compliance and fairness review;
actual costs/value; randomized holdout for uplift; drift monitoring; security
controls. No deployment is included in this portfolio project.
