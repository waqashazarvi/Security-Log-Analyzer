# Security-Log-Analyzer
**Overview**
This Python script implements a basic security log analyzer for detecting anomalies in log data. It uses Pandas for data manipulation, Matplotlib for data visualization, and Scikit-learn for the Isolation Forest algorithm.

**Instructions**
**Install Dependencies:**
Ensure you have Python installed on your system.
Install the required libraries using the following command:
          pip install pandas matplotlib scikit-learn

**Run the Script:**
Replace the file_path variable with the actual path to your log file (CSV format).
Adjust parameters such as the window size for rolling statistics and the contamination level for the Isolation Forest model if needed.
Run the script using this command
          python "Security Log Aanalyzer.py"
          
**Output:**
The script will display the log data, event frequency over time, rolling statistics, and detected anomalies.
