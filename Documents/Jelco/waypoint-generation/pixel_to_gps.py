import pyzed.sl as sl
import csv
import sys

def convert_waypoints_to_gps(waypoints_csv, output_csv):
    """
    Convert image waypoints to GPS coordinates using ZED X with GNSS fusion.
    
    Args:
        waypoints_csv: Path to CSV with columns 'x_col,y_row'
        output_csv: Path to output CSV with GPS coordinates
    """
    
    # Initialize ZED camera
    zed = sl.Camera()
    
    # Create initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Adjust to your resolution
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    
    # Enable positional tracking with GNSS fusion
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_minimum_distance = 0.3
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        sys.exit(1)
    
    # Enable positional tracking
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error enabling positional tracking: {err}")
        zed.close()
        sys.exit(1)
    
    # Check if GNSS fusion is available
    fusion_status = zed.get_sensors_data(sl.SensorsData(), sl.TIME_REFERENCE.CURRENT)
    gnss_data = sl.GNSSData()
    
    # Read waypoints from CSV
    waypoints = []
    with open(waypoints_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x_col'])
            y = float(row['y_row'])
            waypoints.append((x, y))
    
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_csv}")
    
    # Create runtime parameters
    runtime_params = sl.RuntimeParameters()
    
    # Capture a frame to get current camera state
    print("Capturing frame...")
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        
        # Get depth map
        depth_map = sl.Mat()
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        
        # Get camera pose with GNSS fusion
        camera_pose = sl.Pose()
        tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        
        # Get GNSS data
        sensors_data = sl.SensorsData()
        zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)
        gnss_data = sensors_data.get_gnss_data()
        
        print(f"Tracking state: {tracking_state}")
        print(f"GNSS fusion status: {gnss_data.gnss_fusion_status}")
        
        # Get calibration parameters
        calibration = zed.get_camera_information().camera_configuration.calibration_parameters
        fx = calibration.left_cam.fx
        fy = calibration.left_cam.fy
        cx = calibration.left_cam.cx
        cy = calibration.left_cam.cy
        
        # Convert waypoints to GPS coordinates
        gps_waypoints = []
        
        for i, (x_pixel, y_pixel) in enumerate(waypoints):
            # Get depth at this pixel
            err, depth_value = depth_map.get_value(int(x_pixel), int(y_pixel))
            
            if err == sl.ERROR_CODE.SUCCESS and depth_value > 0 and not sl.is_infinite(depth_value):
                # Convert pixel + depth to 3D point in camera frame
                x_cam = (x_pixel - cx) * depth_value / fx
                y_cam = (y_pixel - cy) * depth_value / fy
                z_cam = depth_value
                
                # Create point in camera frame
                point_camera = sl.float3(x_cam, y_cam, z_cam)
                
                # Transform to world frame
                point_world = sl.float3()
                camera_pose.pose_data.transform_point(point_camera, point_world)
                
                # Get GeoPose to convert world coordinates to GPS
                geo_pose = sl.GeoPose()
                zed.get_geo_pose(geo_pose)
                
                if geo_pose.latlng_coordinates.is_init():
                    # The geo_pose gives us the camera's GPS position
                    # We need to offset by the point's position relative to camera
                    # This requires converting the local offset to GPS offset
                    
                    # For accurate conversion, use the heading and position
                    cam_lat = geo_pose.latlng_coordinates.get_latitude()
                    cam_lon = geo_pose.latlng_coordinates.get_longitude()
                    cam_alt = geo_pose.latlng_coordinates.get_altitude()
                    
                    # Convert local ENU offset to GPS offset
                    # This is an approximation for small distances
                    lat_per_meter = 1.0 / 111320.0
                    lon_per_meter = 1.0 / (111320.0 * cos(cam_lat * 3.14159 / 180.0))
                    
                    # In ZED's RIGHT_HANDED_Z_UP_X_FWD:
                    # X is forward, Y is left, Z is up
                    waypoint_lat = cam_lat + (point_world[0] * lat_per_meter)
                    waypoint_lon = cam_lon + (point_world[1] * lon_per_meter)
                    waypoint_alt = cam_alt + point_world[2]
                    
                    gps_waypoints.append({
                        'waypoint_id': i,
                        'x_pixel': x_pixel,
                        'y_pixel': y_pixel,
                        'latitude': waypoint_lat,
                        'longitude': waypoint_lon,
                        'altitude': waypoint_alt,
                        'depth_m': depth_value
                    })
                    
                    print(f"Waypoint {i}: ({x_pixel:.1f}, {y_pixel:.1f}) -> "
                          f"GPS({waypoint_lat:.8f}, {waypoint_lon:.8f}, {waypoint_alt:.2f}m)")
                else:
                    print(f"Waypoint {i}: No GNSS data available")
                    gps_waypoints.append({
                        'waypoint_id': i,
                        'x_pixel': x_pixel,
                        'y_pixel': y_pixel,
                        'latitude': None,
                        'longitude': None,
                        'altitude': None,
                        'depth_m': depth_value,
                        'error': 'No GNSS data'
                    })
            else:
                print(f"Waypoint {i}: Invalid depth at pixel ({x_pixel}, {y_pixel})")
                gps_waypoints.append({
                    'waypoint_id': i,
                    'x_pixel': x_pixel,
                    'y_pixel': y_pixel,
                    'latitude': None,
                    'longitude': None,
                    'altitude': None,
                    'depth_m': None,
                    'error': 'Invalid depth'
                })
        
        # Write results to CSV
        if gps_waypoints:
            with open(output_csv, 'w', newline='') as f:
                fieldnames = ['waypoint_id', 'x_pixel', 'y_pixel', 'latitude', 
                             'longitude', 'altitude', 'depth_m', 'error']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for wp in gps_waypoints:
                    if 'error' not in wp:
                        wp['error'] = ''
                    writer.writerows([wp])
            
            print(f"\nResults saved to {output_csv}")
        
    else:
        print("Failed to grab frame from camera")
    
    # Cleanup
    zed.close()
    print("Camera closed")


if __name__ == "__main__":
    import math
    from math import cos
    
    # Example usage
    waypoints_file = "waypoints.csv"  # Your input file
    output_file = "waypoints_gps.csv"
    
    convert_waypoints_to_gps(waypoints_file, output_file)