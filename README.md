The code detects anomalies in job upload times using a 2-week rolling mean and the Isolation Forest algorithm. It works as follows:
1. Simulated Data Generation: Creates data for multiple jobs with upload times over 30 days.
2. Rolling Mean Calculation: Computes a 14-day rolling average for each job to capture typical performance trends.
3. Deviation Calculation: Measures the difference (time_diff) between actual upload times and the rolling mean.
4. Feature Standardization: Standardizes time_diff for consistent anomaly detection.
5. Anomaly Detection: Uses Isolation Forest to flag significant deviations as anomalies.
6. Output: Lists jobs and dates with unusually high or low upload times for further investigation.
This approach adapts dynamically to job-specific patterns and identifies irregular behavior without requiring labeled data.
