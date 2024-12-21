import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta


# Step 1: Simulated data for the jobs and time taken to upload to cloud
# Assume `job_id`, `time_taken` (in seconds), and `date` (the date the job was run) are in the database.

# Simulated data generation (for the purpose of this example)
job_ids = [f"job_{i}" for i in range(1, 21)]  # 20 different jobs
dates = pd.date_range(datetime.today() - timedelta(30), periods=30).tolist()  # Last 30 days

# Create a DataFrame
data = []
for job in job_ids:
    for date in dates:
        data.append([job, date, np.random.randint(100, 500)])  # Simulating time_taken (in seconds)

df = pd.DataFrame(data, columns=['job_id', 'date', 'time_taken'])
df


# Step 2: Calculate the 2-week (14 days) average time for each job
# Ensure data is sorted by job_id and date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['job_id', 'date'])

# Calculate a rolling mean for the last two weeks (14 days)
df['rolling_mean_2w'] = df.groupby('job_id')['time_taken'].rolling(window=14, min_periods=1).mean().reset_index(0, drop=True)

# Calculate the difference between the current time taken and the 2-week rolling mean
df['time_diff'] = df['time_taken'] - df['rolling_mean_2w']


# Step 3: Standardize the features (for better performance in anomaly detection)
scaler = StandardScaler()
df['scaled_time_diff'] = scaler.fit_transform(df[['time_diff']])


# Step 4: Apply anomaly detection (Isolation Forest) to detect jobs with significant time increases
model = IsolationForest(contamination=0.1)  # Contamination set to 10% (tune based on your data)
df['anomaly'] = model.fit_predict(df[['scaled_time_diff']])



# Step 5: Output the jobs that are flagged as anomalies
# -1 means anomaly, 1 means normal
anomalies = df[df['anomaly'] == -1]
# Display anomalies
print(anomalies[['job_id', 'date', 'time_taken', 'rolling_mean_2w', 'time_diff']])