#!/usr/bin/env python3
"""
Live GPS Recording with Map Visualization
Records GPS waypoints with configurable frequency and format
Shows live map with 5m projection and full route trail
Start/Stop controlled via HTML button
"""

import os
import sys
import time
import json
import math
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from signal import signal, SIGINT
from gpsdclient import GPSDClient
import pyzed.sl as sl

# ============================================================================
# RECORDING CONFIGURATION
# ============================================================================
ENABLE_SVO_RECORDING = False      # Record to SVO file (requires ZED camera)
ENABLE_JSON_EXPORT = True         # Export GPS waypoints to JSON
ENABLE_KML_EXPORT = False         # Export GPS waypoints to KML

WAYPOINT_DECIMATION = 1           # Save every Nth waypoint
HTTP_PORT = 8000

os.makedirs("recordings", exist_ok=True)
# ============================================================================

is_running = True
is_recording = False  # <--- new global flag


def signal_handler(signal_received, frame):
    global is_running, is_recording
    print("\n\nCtrl+C received, stopping gracefully...")
    is_running = False
    is_recording = False


signal(SIGINT, signal_handler)


def get_current_datetime():
    """Get formatted datetime string for filenames"""
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")


class GPSRecorder:
    """Records GPS data and serves it to web interface"""

    def __init__(self):
        self.gnss_client = None
        self.current_gps_data = None
        self.gnss_running = False

        # Recording data
        self.all_gps_points = []
        self.decimation_counter = 0
        self.current_datetime = get_current_datetime()

        # Live display data
        self.current_position = None
        self.current_heading = None
        self.current_speed = 0.0
        self.waypoint_line = []
        self.data_lock = threading.Lock()

        # Statistics
        self.points_received = 0
        self.points_saved = 0
        self.start_time = time.time()

        print("=" * 70)
        print("GPS RECORDER WITH LIVE MAP")
        print("=" * 70)
        print(f"JSON Export:      {'ENABLED' if ENABLE_JSON_EXPORT else 'DISABLED'}")
        print(f"KML Export:       {'ENABLED' if ENABLE_KML_EXPORT else 'DISABLED'}")
        print(f"SVO Recording:    {'ENABLED' if ENABLE_SVO_RECORDING else 'DISABLED'}")
        print("=" * 70)
        print("\nInitializing GPS...")

        self.setup_gps()

        # KML file if enabled
        self.kml_file = None
        if ENABLE_KML_EXPORT:
            self.kml_file = open(f"recordings/route_{self.current_datetime}.kml", 'w')
            self.kml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            self.kml_file.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
            self.kml_file.write('<Document>\n')
            self.kml_file.write('<name>GPS Route</name>\n')
            self.kml_file.write('<Placemark>\n')
            self.kml_file.write('<name>Route</name>\n')
            self.kml_file.write('<LineString>\n')
            self.kml_file.write('<coordinates>\n')

    def setup_gps(self):
        """Initialize GPS reader"""
        try:
            self.gnss_client = GPSDClient(host="127.0.0.1")
            self.gnss_running = True
            self.gps_thread = threading.Thread(target=self.read_gps_loop, daemon=True)
            self.gps_thread.start()
            print("✓ GPS connected")
            print("  Waiting for GPS fix...")

        except Exception as e:
            print(f"ERROR: Could not connect to GPS: {e}")
            print("Make sure gpsd is running: sudo systemctl start gpsd")
            exit(1)

    def read_gps_loop(self):
        """Background thread to read GPS data"""
        try:
            gps_stream = self.gnss_client.dict_stream(convert_datetime=True, filter=["TPV"])
            for gps_data in gps_stream:
                if not self.gnss_running:
                    break
                if gps_data.get('class') == 'TPV':
                    self.current_gps_data = gps_data
                    self.process_gps_point()
        except Exception as e:
            print(f"\nGPS read error: {e}")

    def process_gps_point(self):
        """Process incoming GPS point"""
        global is_recording
        if not is_recording:
            return  # only record when started via button

        if self.current_gps_data is None:
            return

        mode = self.current_gps_data.get('mode', 0)
        if mode < 2:
            return

        lat = self.current_gps_data.get('lat')
        lon = self.current_gps_data.get('lon')
        if lat is None or lon is None:
            return

        self.points_received += 1
        self.decimation_counter += 1

        alt = self.current_gps_data.get('altMSL', 0.0)
        heading = self.current_gps_data.get('track')
        speed = self.current_gps_data.get('speed', 0.0)
        timestamp = self.current_gps_data.get('time')

        gps_point = {
            'ts': int(timestamp.timestamp() * 1e9) if timestamp else int(time.time() * 1e9),
            'coordinates': {'latitude': lat, 'longitude': lon, 'altitude': alt},
            'mode': mode,
            'status': self.current_gps_data.get('status', 0)
        }

        self.all_gps_points.append(gps_point)

        if self.decimation_counter >= WAYPOINT_DECIMATION:
            self.decimation_counter = 0
            self.points_saved += 1
            if ENABLE_KML_EXPORT and self.kml_file:
                self.kml_file.write(f"{lon},{lat},{alt}\n")
                self.kml_file.flush()

        with self.data_lock:
            self.current_position = {"latitude": lat, "longitude": lon}
            self.current_speed = speed
            if heading is not None:
                self.current_heading = heading
                self.waypoint_line = self.generate_waypoint_line(lat, lon, heading)
            else:
                self.current_heading = None
                self.waypoint_line = []

    def project_waypoint(self, lat, lon, heading_deg, distance_m):
        R = 6371000
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        heading_rad = math.radians(heading_deg)
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_m / R)
            + math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(heading_rad)
        )
        new_lon_rad = lon_rad + math.atan2(
            math.sin(heading_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
            math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

    def generate_waypoint_line(self, lat, lon, heading_deg, total_distance=5.0, spacing=1.0):
        waypoints = []
        num_points = int(total_distance / spacing) + 1
        for i in range(num_points):
            distance = i * spacing
            wp_lat, wp_lon = self.project_waypoint(lat, lon, heading_deg, distance)
            waypoints.append({"latitude": wp_lat, "longitude": wp_lon})
        return waypoints

    def get_current_data(self):
        """Get current data for web interface (thread-safe)"""
        global is_recording
        with self.data_lock:
            full_trail = [
                {"latitude": pt['coordinates']['latitude'], "longitude": pt['coordinates']['longitude']}
                for pt in self.all_gps_points
            ]
            return {
                "recording": is_recording,
                "waypoints": self.waypoint_line.copy(),
                "position": self.current_position,
                "heading": self.current_heading,
                "speed": self.current_speed,
                "trail": full_trail,
                "stats": {
                    "points_received": self.points_received,
                    "points_saved": self.points_saved,
                    "recording_time": int(time.time() - self.start_time)
                },
                "timestamp": time.time()
            }

    def save_json(self):
        if not ENABLE_JSON_EXPORT:
            return
        saved_points = self.all_gps_points[::WAYPOINT_DECIMATION] if WAYPOINT_DECIMATION > 1 else self.all_gps_points
        output_data = {'GNSS': saved_points}
        json_file = f"recordings/GNSS_{self.current_datetime}.json"
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ JSON saved: {json_file}")

    def cleanup(self):
        print("\nStopping and saving data...")
        self.gnss_running = False
        if ENABLE_KML_EXPORT and self.kml_file:
            self.kml_file.write('</coordinates>\n</LineString>\n</Placemark>\n</Document>\n</kml>\n')
            self.kml_file.close()
        self.save_json()
        print("Cleanup complete.\n")


class LiveDataHandler(SimpleHTTPRequestHandler):
    recorder = None

    def do_GET(self):
        global is_recording

        if self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = self.recorder.get_current_data()
            self.wfile.write(json.dumps(data).encode())

        elif self.path == '/start':
            is_recording = True
            print("Recording started via web interface.")
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status":"started"}')

        elif self.path == '/stop':
            is_recording = False
            print("Recording stopped via web interface.")
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status":"stopped"}')

        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


def run_http_server(recorder, port=HTTP_PORT):
    LiveDataHandler.recorder = recorder
    server = HTTPServer(('', port), LiveDataHandler)
    print(f"✓ HTTP server started on http://localhost:{port}\n")
    server.serve_forever()


def main():
    recorder = GPSRecorder()

    http_thread = threading.Thread(target=run_http_server, args=(recorder, HTTP_PORT), daemon=True)
    http_thread.start()

    print("=" * 70)
    print(f"Open: http://localhost:{HTTP_PORT}/live_recorder.html")
    print("Press Start Recording in the web interface to begin logging.")
    print("Press Ctrl+C to stop and save.")
    print("=" * 70 + "\n")

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()
