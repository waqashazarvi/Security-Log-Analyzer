import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Read the CSV file
file_path = 'HDFS_100k.log_structured.csv'  # Replace 'path_to_your_file.csv' with the actual file path
log_data = pd.read_csv(file_path)

# Display the first few rows of the log data
print(log_data.head())

# Assuming 'Date' and 'Time' are in the format YYMMDD and HHMMSS respectively
log_data['Timestamp'] = pd.to_datetime(log_data['Date'].astype(str) + log_data['Time'].astype(str), format='%y%m%d%H%M%S')

# Drop unnecessary columns if needed
log_data = log_data.drop(['LineId', 'Date', 'Time'], axis=1)

# Display the updated DataFrame
print(log_data.head())


# Grouping events by timestamp
event_counts = log_data.groupby('Timestamp').size()

# Plotting the number of events over time
plt.figure(figsize=(12, 6))
event_counts.plot()
plt.title('Event Frequency over Time')
plt.xlabel('Timestamp')
plt.ylabel('Event Count')
plt.grid(True)
plt.show()

# ---------------------------------------------
# Calculating rolling mean and standard deviation
window = 30  # Choose an appropriate window size
rolling_mean = event_counts.rolling(window=window).mean()
rolling_std = event_counts.rolling(window=window).std()

# Plotting the original data and rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(event_counts, label='Event Count')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='green')
plt.title('Event Count, Rolling Mean, and Rolling Std')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# Set the anomaly threshold as a factor (e.g., 2) times the rolling standard deviation
anomaly_threshold = 2 * rolling_std

# Detect anomalies based on the threshold
anomalies = event_counts[event_counts > (rolling_mean + anomaly_threshold)]

# Display detected anomalies
print("Detected Anomalies:")
print(anomalies)
 
# Plotting detected anomalies
plt.figure(figsize=(12, 6))
plt.plot(event_counts, label='Event Count')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='green')
plt.scatter(anomalies.index, anomalies, label='Anomaly', color='black')
plt.title('Event Count, Rolling Mean, and Rolling Std')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------
# Reshape data for Isolation Forest input
X = event_counts.values.reshape(-1, 1)

# Fit Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination as needed
isolation_forest.fit(X)


# Predict anomalies
anomaly_preds = isolation_forest.predict(X)

# Create a DataFrame with the event counts and anomaly predictions
anomaly_df = pd.DataFrame({'EventCounts': event_counts.values.flatten(), 'AnomalyPrediction': anomaly_preds})

# Identify anomalies based on predictions
anomalies = anomaly_df[anomaly_df['AnomalyPrediction'] == -1]

# Display detected anomalies
print("Detected Anomalies using Isolation Forest:")
print(anomalies)
print("1----------------------------------------------------------")
# Retrieve log entries corresponding to identified anomalies
anomaly_log_entries = log_data.iloc[anomalies.index]
print("Anomaly Log Entries:")
print(anomaly_log_entries)
print("2----------------------------------------------------------")


# Plotting detected anomalies
plt.figure(figsize=(12, 6))
plt.plot(event_counts, label='Event Count')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='green')
plt.scatter(anomalies.index, event_counts.iloc[anomalies.index], color='black', label='Anomaly')
plt.title('Event Count, Rolling Mean, and Rolling Std with Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


#--------------------------------------------------------------


