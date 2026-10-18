########################################################################
# Configurable Recording Script for ZED Global Localization
# Configure what gets recorded using the settings below
########################################################################

from display.generic_display import GenericDisplay
import pyzed.sl as sl
from gnss_reader.gpsd_reader import GPSDReader
from exporter.gnss_saver import GNSSSaver, get_current_datetime
import exporter.KMLExporter as export
from signal import signal, SIGINT
import sys
import time

# ============================================================================
# RECORDING CONFIGURATION - Set what you want to record
# ============================================================================
ENABLE_SVO_RECORDING = False      # Record video/depth to SVO file (large files!)
ENABLE_JSON_EXPORT = True        # Export GNSS waypoints to JSON file
ENABLE_KML_EXPORT = False         # Export GNSS waypoints to KML file (Google Earth)

# Waypoint frequency control (for JSON/KML only, not SVO)
WAYPOINT_DECIMATION = 1          # Save every Nth waypoint (1=all, 2=every other, 10=every 10th)
                                 # For 1Hz from 10Hz GPS: use 10
                                 # For 2Hz from 10Hz GPS: use 5
# ============================================================================

is_running = True 

def handler(signal_received, frame):
    global is_running
    print("\nCtrl+C received, stopping gracefully...")
    is_running = False 
    
signal(SIGINT, handler)

def main():
    global is_running
    
    print("\n" + "="*70)
    print("ZED GLOBAL LOCALIZATION RECORDING")
    print("="*70)
    print(f"SVO Recording:    {'ENABLED' if ENABLE_SVO_RECORDING else 'DISABLED'}")
    print(f"JSON Export:      {'ENABLED' if ENABLE_JSON_EXPORT else 'DISABLED'}")
    print(f"KML Export:       {'ENABLED' if ENABLE_KML_EXPORT else 'DISABLED'}")
    if WAYPOINT_DECIMATION > 1:
        print(f"Waypoint Mode:    Saving every {WAYPOINT_DECIMATION} points")
    else:
        print(f"Waypoint Mode:    Saving all points (high frequency)")
    print("="*70 + "\n")
    
    # Open the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1 
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Set Region of Interest to exclude car hood
    print("Setting up Region of Interest to exclude car hood...")
    resolution = zed.get_camera_information().camera_configuration.resolution
    roi_height = int(resolution.height * 0.60)
    
    mask_roi = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C1)
    mask_data = mask_roi.get_data()
    mask_data[:] = 0
    mask_data[0:roi_height, :] = 255
    
    zed.set_region_of_interest(mask_roi, [sl.MODULE.POSITIONAL_TRACKING])

    # Enable positional tracking with 15-degree tilt compensation
    tracking_params = sl.PositionalTrackingParameters()
    initial_rotation = sl.Rotation()
    pitch = -0.262  # -15 degrees in radians
    yaw = 0.0
    roll = 0.0
    initial_rotation.set_rotation_vector(pitch, yaw, roll)
    
    initial_translation = sl.Translation(0, 0, 0)
    initial_transform = sl.Transform()
    initial_transform.init_rotation_translation(initial_rotation, initial_translation)
    tracking_params.set_initial_world_transform(initial_transform)
    
    print(f"Applying camera tilt compensation: 15° pitch")
    
    positional_init = zed.enable_positional_tracking(tracking_params)
    if positional_init != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print("[ZED][ERROR] Can't start tracking of camera : "+repr(positional_init)+". Exit program.")
        exit()
    
    # Enable SVO recording (optional)
    svo_path = None
    if ENABLE_SVO_RECORDING:
        svo_path = "ZED_SN"+str(zed.get_camera_information().serial_number)+"_"+get_current_datetime()+".svo"
        returned_state = zed.enable_recording(sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264_LOSSLESS))
        if returned_state != sl.ERROR_CODE.SUCCESS:
            print("Recording ZED : "+repr(returned_state)+". Exit program.")
            zed.close()
            exit()
        print(f"[INFO] SVO recording to: {svo_path}")
    else:
        print("[INFO] SVO recording disabled")
    
    # Create Fusion object
    fusion = sl.Fusion()
    init_fusion_param = sl.InitFusionParameters()
    init_fusion_param.coordinate_units = sl.UNIT.METER
    fusion_init_code = fusion.init(init_fusion_param)
    if fusion_init_code != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Failed to initialize fusion :"+repr(fusion_init_code)+". Exit program")
        exit()

    # Enable odometry publishing
    configuration = sl.CommunicationParameters()
    configuration.set_for_shared_memory()
    zed.start_publishing(configuration)

    # Enable GNSS data producing
    gnss_reader = GPSDReader()
    status_initialize = gnss_reader.initialize()
    if status_initialize == -1:
        gnss_reader.stop_thread()
        zed.close()
        exit()

    # Subscribe to odometry with GNSS antenna offset
    uuid = sl.CameraIdentifier(zed.get_camera_information().serial_number)
    antenna_transform = sl.Transform()
    antenna_x = -0.14  # 14cm behind camera
    antenna_y = -0.42  # 42cm to the right
    antenna_z = 0.0
    
    print(f"Using antenna offset: X={antenna_x}m, Y={antenna_y}m, Z={antenna_z}m")
    antenna_transform.set_translation(sl.Translation(antenna_x, antenna_y, antenna_z))
    fusion.subscribe(uuid, configuration, antenna_transform)

    # Enable positional tracking for Fusion object
    positional_tracking_fusion_parameters = sl.PositionalTrackingFusionParameters()
    positional_tracking_fusion_parameters.enable_GNSS_fusion = True 
    gnss_calibration_parameters = sl.GNSSCalibrationParameters()
    gnss_calibration_parameters.target_yaw_uncertainty = 0.1
    gnss_calibration_parameters.enable_translation_uncertainty_target = False
    gnss_calibration_parameters.enable_reinitialization = True
    gnss_calibration_parameters.gnss_vio_reinit_threshold = 5
    positional_tracking_fusion_parameters.gnss_calibration_parameters = gnss_calibration_parameters
    fusion.enable_positionnal_tracking(positional_tracking_fusion_parameters)

    # Setup viewer
    viewer = GenericDisplay()
    viewer.init(zed.get_camera_information().camera_model)
    
    print("\n" + "="*70)
    print("RECORDING IN PROGRESS")
    print("="*70)
    print("Press Ctrl-C to stop recording")
    print("="*70 + "\n")
    
    py_translation = sl.Translation()
    
    # Initialize GNSS saver (pass None if not recording to SVO)
    gnss_data_saver = GNSSSaver(zed if ENABLE_SVO_RECORDING else None)
    
    # Counters
    gnss_received_count = 0
    gnss_saved_to_json_count = 0
    gnss_saved_to_kml_count = 0
    fusion_success_count = 0
    fusion_initialized = False
    last_status_time = time.time()
    last_error_print = 0

    while is_running and viewer.isAvailable():
        # Grab camera
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed_pose = sl.Pose()
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)

        # Get GNSS data
        status, input_gnss = gnss_reader.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            gnss_received_count += 1
            
            # Display on viewer
            viewer.updateRawGeoPoseData(input_gnss)
            
            # Determine if we should save this waypoint (decimation)
            should_save_waypoint = (gnss_received_count % WAYPOINT_DECIMATION == 0)
            
            # Save to JSON (with decimation)
            if ENABLE_JSON_EXPORT and should_save_waypoint:
                gnss_data_saver.addGNSSData(input_gnss)
                gnss_saved_to_json_count += 1
            
            # Save to KML (with decimation)
            if ENABLE_KML_EXPORT and should_save_waypoint:
                latitude, longitude, altitude = input_gnss.get_coordinates(False)
                coordinates = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude,
                }
                export.saveKMLData("raw_gnss.kml", coordinates)
                gnss_saved_to_kml_count += 1

            # Try to publish GNSS data to Fusion (always, no decimation)
            ingest_error = fusion.ingest_gnss_data(input_gnss)
            if ingest_error == sl.FUSION_ERROR_CODE.SUCCESS:
                fusion_success_count += 1
            elif time.time() - last_error_print > 5:
                print(f"Fusion ingest status: {ingest_error}")
                last_error_print = time.time()

        # Process data and compute positions
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_position = sl.Pose()
            current_state = fusion.get_position(fused_position)

            # Display on OpenGL
            rotation = fused_position.get_rotation_vector()
            translation = fused_position.get_translation(py_translation)
            text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
            viewer.updatePoseData(fused_position.pose_data(), text_translation, text_rotation, current_state) 

            # Get position into the GNSS coordinate system
            current_geopose = sl.GeoPose()
            current_geopose_status = fusion.get_geo_pose(current_geopose)
            if current_geopose_status == sl.GNSS_FUSION_STATUS.OK:
                viewer.updateGeoPoseData(current_geopose)
                if not fusion_initialized:
                    print("[SUCCESS] Fusion initialized! GNSS and camera are now aligned.")
                    fusion_initialized = True
        
        # Print status update every 10 seconds
        current_time = time.time()
        if current_time - last_status_time > 10.0:
            print(f"\n[STATUS] GNSS received: {gnss_received_count}")
            if ENABLE_JSON_EXPORT:
                print(f"[STATUS] Saved to JSON: {gnss_saved_to_json_count}")
            if ENABLE_KML_EXPORT:
                print(f"[STATUS] Saved to KML: {gnss_saved_to_kml_count}")
            print(f"[STATUS] Fusion success: {fusion_success_count}")
            if fusion_initialized:
                print("[STATUS] Fusion: ACTIVE")
            else:
                print("[STATUS] Fusion: Initializing...")
            last_status_time = current_time

    # Cleanup
    print("\n" + "="*70)
    print("STOPPING RECORDING")
    print("="*70)
    
    print(f"\n[FINAL STATS]")
    print(f"Total GNSS points received: {gnss_received_count}")
    if ENABLE_JSON_EXPORT:
        print(f"Waypoints saved to JSON: {gnss_saved_to_json_count}")
    if ENABLE_KML_EXPORT:
        print(f"Waypoints saved to KML: {gnss_saved_to_kml_count}")
    print(f"Points successfully fused: {fusion_success_count}")
    
    gnss_reader.stop_thread()
    
    if ENABLE_SVO_RECORDING:
        zed.disable_recording()
        print(f"\n[INFO] SVO file saved: {svo_path}")
    
    if ENABLE_JSON_EXPORT:
        print("\n[INFO] Saving JSON file...")
        gnss_data_saver.saveAllData()
    
    # Give time for file writes to complete
    time.sleep(0.5)
    
    fusion.close()
    zed.close()
    
    print("\n[SUCCESS] Recording complete!")
    print("="*70 + "\n")
    
if __name__ == '__main__':
    main()