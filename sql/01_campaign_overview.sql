-- Historical subscription labels are ASSOCIATIONS, not causal effects.
SELECT COUNT(*) AS records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*), 2)
           AS subscription_rate_pct,
       SUM(CASE WHEN pdays = 999 THEN 1 ELSE 0 END) AS never_previously_contacted,
       COUNT(DISTINCT job) AS unique_occupations
FROM campaigns;
