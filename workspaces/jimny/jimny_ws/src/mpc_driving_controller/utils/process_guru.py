import csv
from datetime import datetime
import math

# Define input and output filenames
input_file = '/home/axel/workspaces/jimny_ws/src/mpc_driving_controller/data/waypoints/dk-2024.10.26-pre.csv'
output_file = '/home/axel/workspaces/jimny_ws/src/mpc_driving_controller/data/waypoints/dk-2024.10.26.csv'

def convert_timestamp(timestamp_str):
    """Convert ISO timestamp to ROS timestamp in nanoseconds"""
    if timestamp_str and timestamp_str != '-':
        # Parse the timestamp and convert to nanoseconds
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1e9)  # Convert seconds to nanoseconds
    return None  # Return None if the timestamp is invalid

def calculate_yaw(course):
    """Convert course from clockwise North to counterclockwise East in radians"""
    if course and course != '-':
        # Convert clockwise from North to counterclockwise from East
        yaw_from_east = 90 - float(course)
        # Normalize the angle to be within [0, 360] degrees
        yaw_from_east = yaw_from_east % 360
        return math.radians(yaw_from_east)
    return None  # Return None if course is invalid

def convert_speed(speed):
    """Convert speed to float if valid"""
    if speed and speed != '-':
        return float(speed)
    return None  # Return None if speed is invalid

def clean_value(value):
    """Return None for non-essential fields if value is '-'"""
    return None if value == '-' else value

def convert_csv(input_file, output_file):
    with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(csvfile)  # Use DictReader to match column names
        
        # Write output CSV header
        writer = csv.writer(outfile)
        writer.writerow(["ROS Timestamp", "Latitude", "Longitude", "PSIS (Yaw Angle)", "Velocity"])

        # Process each row
        for row in reader:
            try:
                # Access essential data fields
                timestamp_str = row['time']
                latitude = row['latitude'] if row['latitude'] != '-' else None
                longitude = row['longitude'] if row['longitude'] != '-' else None
                speed = convert_speed(row['speed'])
                course = row['course']
                
                # Convert values
                ros_timestamp = convert_timestamp(timestamp_str)
                yaw_angle = calculate_yaw(course)

                # Check essential fields only
                if ros_timestamp is None or latitude is None or longitude is None or yaw_angle is None or speed is None:
                    print(f"Skipping row due to missing or invalid data: {row}")
                    continue

                # Clean up non-essential fields
                row['alt'] = clean_value(row['alt'])
                row['horiz_acc'] = clean_value(row['horiz_acc'])
                row['vert_acc'] = clean_value(row['vert_acc'])
                row['rel_alt'] = clean_value(row['rel_alt'])

                # Write the reformatted row to the output file
                writer.writerow([ros_timestamp, latitude, longitude, yaw_angle, speed])
            except ValueError as e:
                print(f"Skipping row due to error: {e}")

# Run the conversion function
convert_csv(input_file, output_file)
print(f"Conversion completed. Check the '{output_file}' for results.")
