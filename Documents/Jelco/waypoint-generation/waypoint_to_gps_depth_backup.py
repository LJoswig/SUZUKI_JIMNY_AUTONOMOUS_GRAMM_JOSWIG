#!/usr/bin/env python3
"""
Convert image-space waypoints to GPS coordinates using saved capture metadata and depth map.

This script reads waypoints from centerline extraction and converts them to GPS
coordinates using the GPS position, camera pose, and actual depth map from capture.

Usage:
  python3 waypoint_to_gps.py waypoints.csv --metadata capture_12345.json --output waypoints_gps.json
"""

import csv
import json
import argparse
import sys
import numpy as np
from math import cos, sin, radians, sqrt, isnan, isinf
import os


class WaypointConverter:
    """Convert image waypoints to GPS using saved metadata and depth map"""
    
    def __init__(self):
        pass
    
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
        # Convert to radians
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
        
        Args:
            waypoints: List of (x, y) pixel coordinates
            metadata_path: Path to JSON metadata file from capture
            output_json: Output JSON filename
            output_csv: Optional output CSV filename
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
        
        print(f"[INFO] Base GPS: {gps_info['latitude']:.8f}, {gps_info['longitude']:.8f}")
        print(f"[INFO] Camera heading: {pose_info['heading_deg']:.1f}°")
        
        # Load depth map if available
        depth_array = None
        use_actual_depth = False
        
        if metadata.get('depth_available', False) and 'depth_file' in metadata:
            depth_file = metadata['depth_file']
            # Depth file is in same directory as metadata
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
                # Sample depth at waypoint pixel
                x_int = int(round(x_pixel))
                y_int = int(round(y_pixel))
                
                # Bounds check
                if 0 <= y_int < depth_array.shape[0] and 0 <= x_int < depth_array.shape[1]:
                    depth_value = depth_array[y_int, x_int]
                    
                    # Check for invalid depth (NaN, inf, or too close/far)
                    if isnan(depth_value) or isinf(depth_value) or depth_value <= 0.3 or depth_value > 50.0:
                        # Fallback: sample neighboring pixels
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
                            depth_value = 10.0  # Final fallback
                            invalid_depth_count += 1
                    # Else: depth_value is valid
                else:
                    depth_value = 10.0  # Out of bounds
                    invalid_depth_count += 1
            else:
                depth_value = fixed_depth
            
            # Unproject to 3D camera frame
            x_cam = (x_pixel - cx) * depth_value / fx
            y_cam = (y_pixel - cy) * depth_value / fy
            # The depth value z_cam (forward distance) is depth_value
            z_cam = depth_value
            
            # Convert to GPS
            lat, lon, alt = self.local_offset_to_gps(
                gps_info['latitude'],
                gps_info['longitude'],
                gps_info['altitude'],
                x_cam,
                y_cam,
                z_cam, # This is the forward distance/depth
                pose_info['heading_deg']
            )
            
            waypoint_data = {
                'waypoint_id': i,
                'x_pixel': float(x_pixel),
                'y_pixel': float(y_pixel),
                'latitude': float(lat),
                'longitude': float(lon),
                'altitude': float(alt),
                'depth_m': float(depth_value),
                'distance_2d_m': float(sqrt(x_cam**2 + y_cam**2)),
                'depth_source': 'actual' if use_actual_depth else 'fixed'
            }
            
            gps_waypoints.append(waypoint_data)
            valid_count += 1
            
            if i < 5 or i % 10 == 0:  # Print first 5 and every 10th
                print(f"  Waypoint {i}: ({x_pixel:.1f}, {y_pixel:.1f}) -> "
                      f"GPS({lat:.8f}, {lon:.8f}) @ {depth_value:.2f}m")
        
        print(f"\n[SUCCESS] Converted {valid_count}/{len(waypoints)} waypoints")
        if use_actual_depth:
            print(f"[INFO] Used actual depth for {valid_count - invalid_depth_count} waypoints")
            if invalid_depth_count > 0:
                print(f"[WARNING] {invalid_depth_count} waypoints used fallback depth (invalid/missing data)")

        # --- NEW CODE: Sort waypoints by depth (lowest distance to highest distance) ---
        print("\n[INFO] Sorting waypoints by depth_m (lowest distance to highest distance)...")
        # The 'depth_m' value represents the distance (depth) of the waypoint from the camera.
        gps_waypoints.sort(key=lambda wp: wp['depth_m'])
        
        # Re-assign IDs to reflect the new sequence order
        for new_id, wp in enumerate(gps_waypoints):
            wp['waypoint_id'] = new_id
        
        print(f"[INFO] First waypoint after sort (closest): ID {gps_waypoints[0]['waypoint_id']} at {gps_waypoints[0]['depth_m']:.2f}m")
        print(f"[INFO] Last waypoint after sort (farthest): ID {gps_waypoints[-1]['waypoint_id']} at {gps_waypoints[-1]['depth_m']:.2f}m")
        # --- END NEW CODE ---

        
        # Save to JSON (compatible with HTML viewer)
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
                'sorted_by': 'depth_m_ascending' # Metadata update
            },
            'waypoints_detailed': gps_waypoints
        }
        
        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"[SAVED] JSON: {output_json}")
        
        # Save CSV
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert waypoints using saved GPS metadata (automatically loads depth if available)
  python3 waypoint_to_gps.py waypoints.csv --metadata road_masks/capture_1234567890.json --output mission.json
  
  # Also save CSV output
  python3 waypoint_to_gps.py waypoints.csv --metadata road_masks/capture_1234567890.json --output mission.json --csv mission.csv
        """
    )
    
    parser.add_argument('waypoints_csv', help='Input CSV file with waypoints (x_col, y_row)')
    parser.add_argument('--metadata', required=True, help='JSON metadata file from capture (required)')
    parser.add_argument('--output', default='waypoints_gps.json', help='Output JSON file')
    parser.add_argument('--csv', help='Also save to CSV file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"WAYPOINT TO GPS CONVERTER WITH DEPTH")
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
    converter = WaypointConverter()
    
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
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. View waypoints in HTML viewer:")
        print(f"  - Open gps_viewer.html in browser")
        print(f"  - Load {args.output}")
        print(f"\n2. Downsample for mission (optional):")
        print(f"  python3 downsample_waypoints.py {args.output} --frequency 1.0 --mission")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print("\n[FAILED] Conversion failed")
        sys.exit(1)


if __name__ == '__main__':
    main()