#!/usr/bin/env python3
"""
Live GPS Recording with Map Visualization
Records GPS waypoints with configurable frequency and format
Shows live map with 5m projection and full route trail
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

# GPS recording frequency
WAYPOINT_DECIMATION = 1           # Save every Nth waypoint
                                  # 1 = all points (~10 Hz)
                                  # 2 = every other point (~5 Hz)
                                  # 10 = every 10th point (~1 Hz)

# Map server port
HTTP_PORT = 8000

# Create recordings directory
os.makedirs("recordings", exist_ok=True)
# ============================================================================

is_running = True


def signal_handler(signal_received, frame):
    global is_running
    print("\n\nCtrl+C received, stopping gracefully...")
    is_running = False


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
        self.all_gps_points = []  # Full trail for JSON export
        self.decimation_counter = 0
        self.current_datetime = get_current_datetime()
        
        # Live display data
        self.current_position = None
        self.current_heading = None
        self.current_speed = 0.0
        self.waypoint_line = []  # 5m projection
        self.data_lock = threading.Lock()
        
        # Statistics
        self.points_received = 0
        self.points_saved = 0
        self.start_time = time.time()
        
        print("="*70)
        print("GPS RECORDER WITH LIVE MAP")
        print("="*70)
        print(f"JSON Export:      {'ENABLED' if ENABLE_JSON_EXPORT else 'DISABLED'}")
        print(f"KML Export:       {'ENABLED' if ENABLE_KML_EXPORT else 'DISABLED'}")
        print(f"SVO Recording:    {'ENABLED' if ENABLE_SVO_RECORDING else 'DISABLED'}")
        if WAYPOINT_DECIMATION > 1:
            print(f"Recording Mode:   Every {WAYPOINT_DECIMATION} points (~{10/WAYPOINT_DECIMATION:.1f} Hz)")
        else:
            print(f"Recording Mode:   All points (~10 Hz)")
        print("="*70)
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
            
            # Wait for first fix
            timeout = 30
            start = time.time()
            while time.time() - start < timeout:
                if self.current_gps_data is not None:
                    mode = self.current_gps_data.get('mode', 0)
                    if mode >= 2:
                        print("✓ GPS fix acquired!\n")
                        return
                time.sleep(0.1)
            
            print("  Warning: No GPS fix yet, continuing anyway...\n")
            
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
        if self.current_gps_data is None:
            return
        
        mode = self.current_gps_data.get('mode', 0)
        if mode < 2:  # No fix
            return
        
        lat = self.current_gps_data.get('lat')
        lon = self.current_gps_data.get('lon')
        if lat is None or lon is None:
            return
        
        self.points_received += 1
        self.decimation_counter += 1
        
        # Get additional data
        alt = self.current_gps_data.get('altMSL', 0.0)
        heading = self.current_gps_data.get('track')
        speed = self.current_gps_data.get('speed', 0.0)
        timestamp = self.current_gps_data.get('time')
        
        # Create GPS point record
        gps_point = {
            'ts': int(timestamp.timestamp() * 1e9) if timestamp else int(time.time() * 1e9),
            'coordinates': {
                'latitude': lat,
                'longitude': lon,
                'altitude': alt
            },
            'mode': mode,
            'status': self.current_gps_data.get('status', 0)
        }
        
        # Always save to full trail (for map display)
        self.all_gps_points.append(gps_point)
        
        # Save with decimation
        if self.decimation_counter >= WAYPOINT_DECIMATION:
            self.decimation_counter = 0
            self.points_saved += 1
            
            # Save to KML if enabled
            if ENABLE_KML_EXPORT and self.kml_file:
                self.kml_file.write(f"{lon},{lat},{alt}\n")
                self.kml_file.flush()
        
        # Update live display data
        with self.data_lock:
            self.current_position = {"latitude": lat, "longitude": lon}
            self.current_speed = speed
            
            if heading is not None:
                self.current_heading = heading
                # Generate 5m waypoint projection
                self.waypoint_line = self.generate_waypoint_line(lat, lon, heading)
            else:
                self.current_heading = None
                self.waypoint_line = []
    
    def project_waypoint(self, lat, lon, heading_deg, distance_m):
        """Project waypoint using haversine formula"""
        R = 6371000  # Earth radius in meters
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        heading_rad = math.radians(heading_deg)
        
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_m / R) +
            math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(heading_rad)
        )
        
        new_lon_rad = lon_rad + math.atan2(
            math.sin(heading_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
            math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
    def generate_waypoint_line(self, lat, lon, heading_deg, total_distance=5.0, spacing=1.0):
        """Generate line of waypoints ahead"""
        waypoints = []
        num_points = int(total_distance / spacing) + 1
        
        for i in range(num_points):
            distance = i * spacing
            wp_lat, wp_lon = self.project_waypoint(lat, lon, heading_deg, distance)
            waypoints.append({"latitude": wp_lat, "longitude": wp_lon})
        
        return waypoints
    
    def get_current_data(self):
        """Get current data for web interface (thread-safe)"""
        with self.data_lock:
            # Get full trail (all points ever recorded)
            full_trail = [
                {"latitude": pt['coordinates']['latitude'], 
                 "longitude": pt['coordinates']['longitude']}
                for pt in self.all_gps_points
            ]
            
            return {
                "waypoints": self.waypoint_line.copy(),
                "position": self.current_position,
                "heading": self.current_heading,
                "speed": self.current_speed,
                "trail": full_trail,  # Full route trail
                "stats": {
                    "points_received": self.points_received,
                    "points_saved": self.points_saved,
                    "recording_time": int(time.time() - self.start_time)
                },
                "timestamp": time.time()
            }
    
    def save_json(self):
        """Save GPS data to JSON file"""
        if not ENABLE_JSON_EXPORT:
            return
        
        # Apply decimation for saved file
        if WAYPOINT_DECIMATION > 1:
            saved_points = self.all_gps_points[::WAYPOINT_DECIMATION]
        else:
            saved_points = self.all_gps_points
        
        output_data = {'GNSS': saved_points}
        json_file = f"recordings/GNSS_{self.current_datetime}.json"
        
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ JSON saved: {json_file}")
        print(f"  Total points: {len(self.all_gps_points)}")
        print(f"  Saved points: {len(saved_points)} (decimation: {WAYPOINT_DECIMATION})")
    
    def cleanup(self):
        """Cleanup and save files"""
        print("\n" + "="*70)
        print("STOPPING RECORDING")
        print("="*70)
        
        self.gnss_running = False
        
        # Close KML file
        if ENABLE_KML_EXPORT and self.kml_file:
            self.kml_file.write('</coordinates>\n')
            self.kml_file.write('</LineString>\n')
            self.kml_file.write('</Placemark>\n')
            self.kml_file.write('</Document>\n')
            self.kml_file.write('</kml>\n')
            self.kml_file.close()
            print(f"✓ KML saved: recordings/route_{self.current_datetime}.kml")
        
        # Save JSON
        self.save_json()
        
        # Statistics
        duration = time.time() - self.start_time
        print(f"\n[STATISTICS]")
        print(f"Recording duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Points received:    {self.points_received}")
        print(f"Points saved:       {self.points_saved}")
        print(f"Average rate:       {self.points_received/duration:.1f} Hz")
        
        print("\n" + "="*70)
        print("Recording complete! Files saved in recordings/ folder")
        print("="*70 + "\n")


class LiveDataHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving live data"""
    
    recorder = None
    
    def do_GET(self):
        if self.path == '/data':
            # Serve current data as JSON
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            data = self.recorder.get_current_data()
            self.wfile.write(json.dumps(data).encode())
        else:
            # Serve static files
            super().do_GET()
    
    def log_message(self, format, *args):
        # Suppress HTTP log messages
        pass


def run_http_server(recorder, port=HTTP_PORT):
    """Run HTTP server in background"""
    LiveDataHandler.recorder = recorder
    server = HTTPServer(('', port), LiveDataHandler)
    print(f"✓ HTTP server started on http://localhost:{port}\n")
    server.serve_forever()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           GPS RECORDER WITH LIVE MAP VISUALIZATION               ║
║                                                                   ║
║  Records GPS waypoints and displays live map with:               ║
║  • Current position                                              ║
║  • 5m waypoint projection ahead                                  ║
║  • Complete route trail (never deleted)                          ║
║                                                                   ║
║  On exit, saves files to recordings/ folder:                     ║
║  • JSON file with GPS waypoints                                  ║
║  • KML file for Google Earth (if enabled)                        ║
║                                                                   ║
║  Press Ctrl+C to stop and save                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    recorder = GPSRecorder()
    
    # Start HTTP server
    http_thread = threading.Thread(
        target=run_http_server,
        args=(recorder, HTTP_PORT),
        daemon=True
    )
    http_thread.start()
    
    time.sleep(0.5)
    
    print("="*70)
    print("RECORDING IN PROGRESS")
    print("="*70)
    print(f"Open: http://localhost:{HTTP_PORT}/live_recorder.html")
    print("Press Ctrl+C to stop and save")
    print("="*70 + "\n")
    
    # Status updates
    try:
        last_status = time.time()
        while is_running:
            time.sleep(1)
            
            # Print status every 10 seconds
            if time.time() - last_status > 10:
                with recorder.data_lock:
                    pos = recorder.current_position
                    heading = recorder.current_heading
                    speed = recorder.current_speed
                
                if pos and heading is not None:
                    print(f"[STATUS] Points: {recorder.points_received} | "
                          f"Saved: {recorder.points_saved} | "
                          f"Position: {pos['latitude']:.6f}, {pos['longitude']:.6f} | "
                          f"Heading: {heading:.1f}° | Speed: {speed:.1f} m/s")
                else:
                    print(f"[STATUS] Points: {recorder.points_received} | "
                          f"Waiting for heading (start moving)...")
                
                last_status = time.time()
                
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()