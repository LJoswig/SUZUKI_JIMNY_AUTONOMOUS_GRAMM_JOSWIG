#!/usr/bin/env python3
"""
Waypoint Downsampler for CubePilot Orange Cube
Converts high-frequency GPS recordings to mission-ready waypoints

Usage: python3 downsample_waypoints.py input.json [--frequency 1.0] [--output mission.json]
"""

import json
import argparse
import math
from typing import List, Dict, Tuple

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters (Haversine formula)"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def downsample_by_frequency(waypoints: List[Dict], target_hz: float) -> List[Dict]:
    """Downsample by keeping every Nth point to achieve target frequency"""
    if not waypoints:
        return []
    
    # Estimate original frequency from timestamps
    if len(waypoints) > 1:
        time_diff = (waypoints[-1]['ts'] - waypoints[0]['ts']) / 1e9  # Convert ns to seconds
        original_hz = len(waypoints) / time_diff
        decimation = max(1, round(original_hz / target_hz))
    else:
        decimation = 1
    
    downsampled = [waypoints[i] for i in range(0, len(waypoints), decimation)]
    
    print(f"Original frequency: ~{original_hz:.1f} Hz")
    print(f"Target frequency: {target_hz} Hz")
    print(f"Decimation factor: {decimation}")
    print(f"Original points: {len(waypoints)}")
    print(f"Downsampled points: {len(downsampled)}")
    
    return downsampled

def downsample_by_distance(waypoints: List[Dict], min_distance_m: float) -> List[Dict]:
    """Downsample by keeping points that are at least min_distance_m apart"""
    if not waypoints:
        return []
    
    downsampled = [waypoints[0]]  # Always keep first point
    
    for wp in waypoints[1:]:
        last_kept = downsampled[-1]
        
        lat1 = last_kept['coordinates']['latitude']
        lon1 = last_kept['coordinates']['longitude']
        lat2 = wp['coordinates']['latitude']
        lon2 = wp['coordinates']['longitude']
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        
        if distance >= min_distance_m:
            downsampled.append(wp)
    
    print(f"Distance-based filtering: {min_distance_m}m minimum spacing")
    print(f"Original points: {len(waypoints)}")
    print(f"Downsampled points: {len(downsampled)}")
    
    return downsampled

def downsample_adaptive(waypoints: List[Dict], max_points: int = 700) -> List[Dict]:
    """
    Adaptive downsampling that preserves path shape
    Keeps more points in curves, fewer on straight sections
    """
    if len(waypoints) <= max_points:
        return waypoints
    
    def calculate_curvature(p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate curvature at middle point"""
        # Simple angle-based curvature
        lat1 = p1['coordinates']['latitude']
        lon1 = p1['coordinates']['longitude']
        lat2 = p2['coordinates']['latitude']
        lon2 = p2['coordinates']['longitude']
        lat3 = p3['coordinates']['latitude']
        lon3 = p3['coordinates']['longitude']
        
        # Vectors
        v1_lat, v1_lon = lat2 - lat1, lon2 - lon1
        v2_lat, v2_lon = lat3 - lat2, lon3 - lon2
        
        # Angle between vectors (higher = more curvature)
        dot = v1_lat * v2_lat + v1_lon * v2_lon
        mag1 = math.sqrt(v1_lat**2 + v1_lon**2)
        mag2 = math.sqrt(v2_lat**2 + v2_lon**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp
        angle = math.acos(cos_angle)
        
        return angle
    
    # Calculate importance score for each point
    importance = [1.0]  # First point always important
    
    for i in range(1, len(waypoints) - 1):
        curvature = calculate_curvature(waypoints[i-1], waypoints[i], waypoints[i+1])
        importance.append(curvature)
    
    importance.append(1.0)  # Last point always important
    
    # Sort by importance and keep top N
    indexed_importance = list(enumerate(importance))
    indexed_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Keep the most important points
    keep_indices = sorted([idx for idx, _ in indexed_importance[:max_points]])
    downsampled = [waypoints[i] for i in keep_indices]
    
    print(f"Adaptive downsampling to {max_points} points")
    print(f"Original points: {len(waypoints)}")
    print(f"Downsampled points: {len(downsampled)}")
    
    return downsampled

def export_to_mavlink_mission(waypoints: List[Dict], output_file: str, altitude: float = 50.0):
    """
    Export to QGroundControl/Mission Planner waypoint format
    Creates a .waypoints file compatible with ArduPilot
    """
    with open(output_file, 'w') as f:
        f.write("QGC WPL 110\n")  # QGroundControl waypoint format version
        
        # Home position (first waypoint)
        if waypoints:
            first = waypoints[0]['coordinates']
            f.write(f"0\t1\t0\t16\t0\t0\t0\t0\t{first['latitude']}\t{first['longitude']}\t{altitude}\t1\n")
        
        # Mission waypoints
        for i, wp in enumerate(waypoints, start=1):
            coords = wp['coordinates']
            # Format: index current frame command param1-4 lat lon alt autocontinue
            # Command 16 = MAV_CMD_NAV_WAYPOINT
            f.write(f"{i}\t0\t0\t16\t0\t0\t0\t0\t{coords['latitude']}\t{coords['longitude']}\t{altitude}\t1\n")
    
    print(f"\nMission file exported: {output_file}")
    print(f"Format: QGroundControl .waypoints")
    print(f"Altitude: {altitude}m (you can edit this in Mission Planner)")

def main():
    parser = argparse.ArgumentParser(
        description='Downsample GPS waypoints for autonomous navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Downsample to 1 Hz
  python3 downsample_waypoints.py recording.json --frequency 1.0
  
  # Downsample to 2 Hz and export mission
  python3 downsample_waypoints.py recording.json --frequency 2.0 --mission
  
  # Distance-based: keep points 5m apart
  python3 downsample_waypoints.py recording.json --distance 5.0
  
  # Adaptive: keep best 700 points preserving curves
  python3 downsample_waypoints.py recording.json --adaptive 700
        """
    )
    
    parser.add_argument('input', help='Input JSON file from ZED recording')
    parser.add_argument('--frequency', type=float, help='Target frequency in Hz (e.g., 1.0, 2.0)')
    parser.add_argument('--distance', type=float, help='Minimum distance between points in meters')
    parser.add_argument('--adaptive', type=int, help='Adaptive downsampling to N points (e.g., 700)')
    parser.add_argument('--output', default='downsampled.json', help='Output JSON file')
    parser.add_argument('--mission', action='store_true', help='Also export to Mission Planner format')
    parser.add_argument('--altitude', type=float, default=50.0, help='Mission altitude in meters (default: 50)')
    
    args = parser.parse_args()
    
    # Load input file
    print(f"\nLoading: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    waypoints = data.get('GNSS', [])
    if not waypoints:
        print("Error: No GNSS data found in file")
        return
    
    print(f"Loaded {len(waypoints)} waypoints\n")
    
    # Calculate route statistics
    if len(waypoints) > 1:
        first = waypoints[0]['coordinates']
        last = waypoints[-1]['coordinates']
        total_distance = calculate_distance(
            first['latitude'], first['longitude'],
            last['latitude'], last['longitude']
        )
        duration = (waypoints[-1]['ts'] - waypoints[0]['ts']) / 1e9
        print(f"Route statistics:")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  Straight-line distance: {total_distance:.1f} meters")
        print()
    
    # Apply downsampling method
    if args.adaptive:
        downsampled_waypoints = downsample_adaptive(waypoints, args.adaptive)
    elif args.distance:
        downsampled_waypoints = downsample_by_distance(waypoints, args.distance)
    elif args.frequency:
        downsampled_waypoints = downsample_by_frequency(waypoints, args.frequency)
    else:
        print("Error: Please specify --frequency, --distance, or --adaptive")
        return
    
    # Save downsampled JSON
    output_data = {'GNSS': downsampled_waypoints}
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved: {args.output}")
    
    # Export to mission format if requested
    if args.mission:
        mission_file = args.output.replace('.json', '.waypoints')
        export_to_mavlink_mission(downsampled_waypoints, mission_file, args.altitude)
        print(f"\nTo use with Orange Cube:")
        print(f"1. Open Mission Planner or QGroundControl")
        print(f"2. Load the waypoint file: {mission_file}")
        print(f"3. Upload mission to Orange Cube")

if __name__ == '__main__':
    main()