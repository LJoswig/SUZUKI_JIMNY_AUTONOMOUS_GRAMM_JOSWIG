#!/usr/bin/env python3
"""
Convert image-space waypoints to GPS coordinates using saved capture metadata and depth map.
CORRECTED: Uses roll (not yaw) for heading with East/West flip option.
"""

import csv
import json
import argparse
import sys
import numpy as np
from math import cos, sin, radians, sqrt, isnan, isinf
import os

# ============================================
# CONFIGURATION: Heading calibration
# ============================================
# Roll tracks heading but North/South are flipped
# To flip North/South while keeping East/West: heading = 180 - roll
# This mirrors the angle across the East-West axis

FLIP_NORTH_SOUTH = True  # Set to False if north/south are correct
HEADING_CORRECTION = 0.0  # Add fixed offset in degrees (positive = clockwise)
                          # Adjust this if waypoints are consistently off by X degrees
# ============================================


class WaypointConverter:
    """Convert image waypoints to GPS using saved metadata and depth map"""
    
    def __init__(self, flip_north_south=True, heading_correction=0.0):
        self.flip_north_south = flip_north_south
        self.heading_correction = heading_correction
    
    def local_offset_to_gps(self, base_lat, base_lon, base_alt, offset_x, offset_y, offset_z, heading_deg=0):
        """
        Convert local ENU offset to GPS coordinates
        
        Args:
            base_lat, base_lon, base_alt: Base GPS position
            offset_x: Offset in meters (forward in camera frame)
            offset_y: Offset in meters (left in camera frame)
            offset_z: Offset in meters (up)
            heading_deg: Camera heading in degrees (0 = North, 90 = East)
        """
        heading_rad = radians(heading_deg)
        
        # Rotate offsets from camera frame to ENU frame
        # Camera: X=forward, Y=left, Z=up
        # ENU: X=East, Y=North, Z=Up
        east_offset = offset_x * sin(heading_rad) + offset_y * cos(heading_rad)
        north_offset = offset_x * cos(heading_rad) - offset_y * sin(heading_rad)
        up_offset = offset_z
        
        # Convert to GPS offset (approximate for small distances)
        lat_per_meter = 1.0 / 111320.0
        lon_per_meter = 1.0 / (111320.0 * cos(radians(base_lat)))
        
        new_lat = base_lat + (north_offset * lat_per_meter)
        new_lon = base_lon + (east_offset * lon_per_meter)
        new_alt = base_alt + up_offset
        
        return new_lat, new_lon, new_alt
    
    def convert_waypoints_from_metadata(self, waypoints, metadata_path, output_json, output_csv=None):
        """
        Convert waypoints using saved GPS metadata and depth map
        """
        # Load metadata
        print(f"\n[INFO] Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check GPS fix
        if not metadata['gps']['has_fix']:
            print("[ERROR] No GPS fix in metadata!")
            return False
        
        gps_info = metadata['gps']
        pose_info = metadata['camera_pose']
        
        # Use ROLL (not yaw) for heading
        roll_rad = pose_info['rotation']['roll']
        roll_deg = np.degrees(roll_rad)
        
        # Apply north/south flip: heading = 180 - roll
        # This mirrors the angle across East-West axis (flips North <-> South)
        if self.flip_north_south:
            heading_before_correction = 180.0 - roll_deg
        else:
            heading_before_correction = roll_deg
        
        # Apply fixed heading correction
        heading_deg = heading_before_correction + self.heading_correction
        
        # Normalize to 0-360
        heading_deg = heading_deg % 360
        if heading_deg < 0:
            heading_deg += 360
        
        print(f"[INFO] Base GPS: {gps_info['latitude']:.8f}, {gps_info['longitude']:.8f}")
        print(f"[INFO] Roll (raw): {roll_deg:.2f}°")
        if self.flip_north_south:
            print(f"[INFO] After flip (180 - roll): {heading_before_correction:.2f}°")
        if self.heading_correction != 0:
            print(f"[INFO] Heading correction: {self.heading_correction:+.1f}°")
            print(f"[INFO] After correction: {heading_deg:.2f}°")
        print(f"[INFO] Final heading used: {heading_deg:.2f}°")
        
        # Load depth map if available
        depth_array = None
        use_actual_depth = False
        
        if metadata.get('depth_available', False) and 'depth_file' in metadata:
            depth_file = metadata['depth_file']
            metadata_dir = os.path.dirname(metadata_path)
            depth_path = os.path.join(metadata_dir, depth_file)
            
            if os.path.exists(depth_path):
                print(f"[INFO] Loading depth map: {depth_path}")
                depth_array = np.load(depth_path)
                print(f"[INFO] Depth map shape: {depth_array.shape}")
                print(f"[INFO] Depth range: {np.nanmin(depth_array):.2f}m - {np.nanmax(depth_array):.2f}m")
                use_actual_depth = True
            else:
                print(f"[WARNING] Depth file not found: {depth_path}")
                print("[WARNING] Will use fixed depth fallback")
        
        if not use_actual_depth:
            print("[INFO] Using fixed depth: 10.0m (no depth map available)")
            fixed_depth = 10.0
        
        # Get camera calibration from metadata
        calib = metadata['camera_calibration']
        fx = calib['fx']
        fy = calib['fy']
        cx = calib['cx']
        cy = calib['cy']
        
        # Convert each waypoint
        gps_waypoints = []
        valid_count = 0
        invalid_depth_count = 0
        
        for i, (x_pixel, y_pixel) in enumerate(waypoints):
            # Get depth value
            if use_actual_depth:
                x_int = int(round(x_pixel))
                y_int = int(round(y_pixel))
                
                if 0 <= y_int < depth_array.shape[0] and 0 <= x_int < depth_array.shape[1]:
                    depth_value = depth_array[y_int, x_int]
                    
                    if isnan(depth_value) or isinf(depth_value) or depth_value <= 0.3 or depth_value > 50.0:
                        window_size = 5
                        y_start = max(0, y_int - window_size)
                        y_end = min(depth_array.shape[0], y_int + window_size + 1)
                        x_start = max(0, x_int - window_size)
                        x_end = min(depth_array.shape[1], x_int + window_size + 1)
                        
                        neighborhood = depth_array[y_start:y_end, x_start:x_end]
                        valid_depths = neighborhood[(~np.isnan(neighborhood)) & 
                                                    (~np.isinf(neighborhood)) & 
                                                    (neighborhood > 0.3) & 
                                                    (neighborhood <= 50.0)]
                        
                        if len(valid_depths) > 0:
                            depth_value = np.median(valid_depths)
                        else:
                            depth_value = 10.0
                            invalid_depth_count += 1
                else:
                    depth_value = 10.0
                    invalid_depth_count += 1
            else:
                depth_value = fixed_depth
            
            # Unproject to 3D camera coordinates
            x_cam = (x_pixel - cx) * depth_value / fx
            y_cam = (y_pixel - cy) * depth_value / fy
            z_cam = depth_value
            
            # Camera frame offsets
            offset_x = z_cam    # Forward distance
            offset_y = x_cam    # Lateral offset (right is positive)
            offset_z = y_cam    # Vertical offset (down is positive)
            
            # Convert to GPS
            lat, lon, alt = self.local_offset_to_gps(
                gps_info['latitude'],
                gps_info['longitude'],
                gps_info['altitude'],
                offset_x,
                offset_y,
                offset_z,
                heading_deg
            )
            
            waypoint_data = {
                'waypoint_id': i,
                'x_pixel': float(x_pixel),
                'y_pixel': float(y_pixel),
                'latitude': float(lat),
                'longitude': float(lon),
                'altitude': float(alt),
                'depth_m': float(depth_value),
                'distance_2d_m': float(sqrt(offset_x**2 + offset_y**2)),
                'depth_source': 'actual' if use_actual_depth else 'fixed'
            }
            
            gps_waypoints.append(waypoint_data)
            valid_count += 1
            
            if i < 5 or i % 10 == 0:
                print(f"  Waypoint {i}: ({x_pixel:.1f}, {y_pixel:.1f}) -> "
                      f"GPS({lat:.8f}, {lon:.8f}) @ {depth_value:.2f}m")
        
        print(f"\n[SUCCESS] Converted {valid_count}/{len(waypoints)} waypoints")
        if use_actual_depth:
            print(f"[INFO] Used actual depth for {valid_count - invalid_depth_count} waypoints")
            if invalid_depth_count > 0:
                print(f"[WARNING] {invalid_depth_count} waypoints used fallback depth (invalid/missing data)")

        # Sort waypoints by depth
        print("\n[INFO] Sorting waypoints by depth_m (lowest distance to highest distance)...")
        gps_waypoints.sort(key=lambda wp: wp['depth_m'])
        
        for new_id, wp in enumerate(gps_waypoints):
            wp['waypoint_id'] = new_id
        
        print(f"[INFO] First waypoint after sort (closest): ID {gps_waypoints[0]['waypoint_id']} at {gps_waypoints[0]['depth_m']:.2f}m")
        print(f"[INFO] Last waypoint after sort (farthest): ID {gps_waypoints[-1]['waypoint_id']} at {gps_waypoints[-1]['depth_m']:.2f}m")
        
        # Save to JSON
        json_data = {
            'waypoints': [
                {
                    'latitude': wp['latitude'],
                    'longitude': wp['longitude'],
                    'altitude': wp.get('altitude'),
                } for wp in gps_waypoints if wp.get('latitude') is not None
            ],
            'metadata': {
                'base_position': gps_info,
                'camera_pose': pose_info,
                'capture_timestamp': metadata['timestamp'],
                'capture_time': metadata['timestamp_iso'],
                'total_waypoints': len(waypoints),
                'valid_waypoints': valid_count,
                'source': 'centerline_extraction',
                'depth_mode': 'actual' if use_actual_depth else 'fixed',
                'invalid_depth_count': invalid_depth_count if use_actual_depth else 0,
                'sorted_by': 'depth_m_ascending',
                'heading_source': 'roll',
                'heading_formula': '180 - roll' if self.flip_north_south else 'roll'
            },
            'waypoints_detailed': gps_waypoints
        }
        
        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"[SAVED] JSON: {output_json}")
        
        if output_csv:
            with open(output_csv, 'w', newline='') as f:
                fieldnames = ['waypoint_id', 'x_pixel', 'y_pixel', 'latitude', 
                              'longitude', 'altitude', 'depth_m', 'distance_2d_m', 'depth_source']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(gps_waypoints)
            print(f"[SAVED] CSV: {output_csv}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert image waypoints to GPS coordinates using saved metadata and depth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('waypoints_csv', help='Input CSV file with waypoints (x_col, y_row)')
    parser.add_argument('--metadata', required=True, help='JSON metadata file from capture (required)')
    parser.add_argument('--output', default='waypoints_gps.json', help='Output JSON file')
    parser.add_argument('--csv', help='Also save to CSV file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"WAYPOINT TO GPS CONVERTER")
    print(f"Using ROLL for heading")
    print(f"North/South flip: {'ENABLED (180 - roll)' if FLIP_NORTH_SOUTH else 'DISABLED (roll)'}")
    if HEADING_CORRECTION != 0:
        print(f"Heading correction: {HEADING_CORRECTION:+.1f}°")
    print(f"{'='*60}\n")
    
    # Load waypoints
    waypoints = []
    with open(args.waypoints_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x_col'])
            y = float(row['y_row'])
            waypoints.append((x, y))
    
    print(f"[INFO] Loaded {len(waypoints)} waypoints from {args.waypoints_csv}")
    
    # Initialize converter
    converter = WaypointConverter(
        flip_north_south=FLIP_NORTH_SOUTH,
        heading_correction=HEADING_CORRECTION
    )
    
    # Convert waypoints
    success = converter.convert_waypoints_from_metadata(
        waypoints, 
        args.metadata,
        args.output,
        args.csv
    )
    
    if success:
        print(f"\n{'='*60}")
        print("CONVERSION COMPLETE")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print("\n[FAILED] Conversion failed")
        sys.exit(1)


if __name__ == '__main__':
    main()