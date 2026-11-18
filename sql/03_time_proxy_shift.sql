-- Source row order is chronological, but rows are not exact calendar days.
SELECT dataset_split, COUNT(*) AS records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_pct,
       ROUND(AVG(age), 1) AS avg_age
FROM campaigns
GROUP BY dataset_split
ORDER BY MIN(source_row);
