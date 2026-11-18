-- Exclude tiny segments from operational conclusions, not from the data.
SELECT job, poutcome,
       COUNT(*) AS records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_pct
FROM campaigns
GROUP BY job, poutcome
HAVING COUNT(*) >= 30
ORDER BY subscription_rate_pct DESC, records DESC;
