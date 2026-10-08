import pandas as pd
from datetime import datetime, timedelta
import os

# Define the number of waypoints
num_waypoints = 10

# Create sample data
data = []
start_time = datetime(2021, 9, 24, 14, 0, 0)  # Starting timestamp

for i in range(num_waypoints):
    timestamp = int((start_time + timedelta(seconds=i * 6)).timestamp() * 1e9)  # ROS timestamp in nanoseconds
    latitude = 34.052235 + i * 0.0001  # Increment latitude
    longitude = -118.243683 + i * 0.0001  # Increment longitude
    psis = i * 0.1  # Increment yaw angle
    data.append([timestamp, latitude, longitude, psis])

# Create a DataFrame
df = pd.DataFrame(data, columns=['ROS Timestamp', 'Latitude', 'Longitude', 'PSIS (Yaw Angle)'])

# Ensure the directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../data/waypoints')
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_dir, 'sample_wp.csv')

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)
