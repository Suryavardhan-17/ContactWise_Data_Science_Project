SELECT COUNT(*) AS records,
       SUM(CASE WHEN job = 'unknown' THEN 1 ELSE 0 END) AS unknown_job,
       SUM(CASE WHEN education = 'unknown' THEN 1 ELSE 0 END) AS unknown_education,
       SUM(CASE WHEN "default" = 'unknown' THEN 1 ELSE 0 END) AS unknown_default,
       SUM(CASE WHEN housing = 'unknown' THEN 1 ELSE 0 END) AS unknown_housing,
       SUM(CASE WHEN loan = 'unknown' THEN 1 ELSE 0 END) AS unknown_loan,
       SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS null_duration
FROM campaigns;
