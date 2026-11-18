# Three-week plan to turn the provided core into your own portfolio story

The repository includes the functioning **core implementation**. This plan is
for understanding, reviewing, and extending it personally before interviews.

## Week 1 — Business and data understanding
- Read the dataset and UCI variable descriptions.
- Explain the pre-contact feature whitelist and `duration` leakage in your own words.
- Inspect the four SQL outputs, missing/unknown categories and historical groups.
- Recreate one segment chart manually and explain its uncertainty interval.
- Write a 150-word problem statement and list what is *not* measurable.

## Week 2 — Statistical modeling
- Run the pipeline and examine all three candidates on the selection period.
- Explain AP, ROC-AUC, precision@K, calibration and Brier score.
- Change `--budget` to 100 and 1000; compare **observed** ranks.
- Add a model error analysis: false positives/negatives in representative groups.
- Make one hypothesis about period drift; check it with data and report caveats.

## Week 3 — Decision support, documentation and portfolio
- Change cost/value assumptions, report how sensitive the *scenario proxy* is.
- Write one-page findings with actual generated numbers and limitations.
- Publish 2–3 generated charts after reviewing them for correctness.
- Update README status with genuine results and a reproducibility screenshot.
- Practice your 2-minute interview story without reading from the notes.

**Success condition:** You can defend every feature, validation split, metric,
and business caveat, and can re-run the project on another laptop.
