import numpy as np 
import json  
from datetime import datetime
import os
import pyzed.sl as sl

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")

def convert_gnss_data_2_json(gnss_data: sl.GNSSData) -> json:
    latitude, longitude, altitude = gnss_data.get_coordinates(False)
    gnss_measure = {}
    gnss_measure["ts"] = gnss_data.ts.get_nanoseconds()
    coordinates_dict = {}
    coordinates_dict["latitude"] = latitude
    coordinates_dict["longitude"] = longitude
    coordinates_dict["altitude"] = altitude
    gnss_measure["coordinates"] = coordinates_dict
    position_covariance = [gnss_data.position_covariances[j] for j in range(9)]
    gnss_measure["position_covariance"] = position_covariance
    gnss_measure["longitude_std"] = np.sqrt(position_covariance[0 * 3 + 0])
    gnss_measure["latitude_std"] = np.sqrt(position_covariance[1 * 3 + 1])
    gnss_measure["altitude_std"] = np.sqrt(position_covariance[2 * 3 + 2])
    gnss_measure["mode"] = gnss_data.gnss_mode.value
    gnss_measure["status"] = gnss_data.gnss_status.value
    return gnss_measure

class GNSSSaver:
    def __init__(self, zed: sl.Camera):
        self.current_date = get_current_datetime()
        
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        self.file_path = "recordings/GNSS_"+self.current_date+".json"
        self.all_gnss_data = []
        self._zed = zed
        
    def addGNSSData(self, gnss_data):
        # CRITICAL FIX: Save to BOTH SVO and JSON list
        # This ensures data is available in both formats
        
        # Always add to list for JSON export
        self.all_gnss_data.append(gnss_data)
        
        # Also save to SVO if recording
        if self._zed is not None:
            data = sl.SVOData()
            data.key = "GNSS_json"
            data.set_string_content(json.dumps(convert_gnss_data_2_json(gnss_data)))
            data.timestamp_ns = gnss_data.ts
            self._zed.ingest_data_into_svo(data)
        
    def saveAllData(self):
        print("Start saving GNSS data...")
        all_gnss_measurements = [] 
        for i in range(len(self.all_gnss_data)):
            gnss_measure = convert_gnss_data_2_json(self.all_gnss_data[i])
            all_gnss_measurements.append(gnss_measure)
        
        final_dict = {"GNSS" : all_gnss_measurements}
        
        with open(self.file_path, "w") as outfile:
            json.dump(final_dict, outfile, indent=2)  # Added indent for readability
        
        print(f"All GNSS data saved: {len(all_gnss_measurements)} points")
        print(f"JSON file: {self.file_path}")