########################################################################
#
# Copyright (c) 2025, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE,
# DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample shows how to track the position of the ZED camera 
    and displays it in a OpenGL window.
"""

from display.generic_display import GenericDisplay
import pyzed.sl as sl
from gnss_reader.gpsd_reader import GPSDReader

def main():
    # Open the camera
    zed = sl.Camera() 
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1 
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Set Region of Interest to exclude car hood
    # This improves visual odometry by ignoring static objects in view
    print("Setting up Region of Interest to exclude car hood...")
    resolution = zed.get_camera_information().camera_configuration.resolution
    
    # Create ROI mask - use only top 60% of image (adjust if needed)
    roi_height = int(resolution.height * 0.60)
    
    # Create sl.Mat and initialize it
    mask_roi = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C1)
    
    # Get numpy view and set mask values (white = use, black = ignore)
    mask_data = mask_roi.get_data()
    mask_data[:] = 0  # All black initially
    mask_data[0:roi_height, :] = 255  # Top 60% is white (active)
    
    # Apply ROI to positional tracking module
    zed.set_region_of_interest(mask_roi, [sl.MODULE.POSITIONAL_TRACKING])

    # Enable positional tracking with 15-degree tilt compensation only
    # Don't rotate the camera frame - we'll handle antenna orientation separately
    tracking_params = sl.PositionalTrackingParameters()
    
    # Only compensate for the 15-degree downward camera tilt
    initial_rotation = sl.Rotation()
    pitch = -0.262  # -15 degrees in radians (camera tilted down)
    yaw = 0.0       # No yaw rotation
    roll = 0.0
    initial_rotation.set_rotation_vector(pitch, yaw, roll)
    
    initial_translation = sl.Translation(0, 0, 0)
    initial_transform = sl.Transform()
    initial_transform.init_rotation_translation(initial_rotation, initial_translation)
    
    tracking_params.set_initial_world_transform(initial_transform)
    
    print(f"Applying camera tilt compensation: 15° pitch")
    
    positional_init = zed.enable_positional_tracking(tracking_params)
    if positional_init != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Can't start tracking of camera : "+repr(positional_init)+". Exit program.")
        exit()

    # Create Fusion object:
    fusion = sl.Fusion()
    init_fusion_param = sl.InitFusionParameters()
    init_fusion_param.coordinate_units = sl.UNIT.METER
    fusion_init_code = fusion.init(init_fusion_param)
    if fusion_init_code != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Failed to initialize fusion :"+repr(fusion_init_code)+". Exit program")
        exit()

    # Enable odometry publishing:
    configuration = sl.CommunicationParameters()
    zed.start_publishing(configuration)

    # Enable GNSS data producing:
    gnss_reader = GPSDReader()
    status_initialize = gnss_reader.initialize()
    if status_initialize==-1:
        gnss_reader.stop_thread()
        zed.close()
        exit()

    # Subscribe to odometry with GNSS antenna offset:
    # CRITICAL: Measure the actual offset from camera to GNSS antenna on your vehicle
    # The transform is in the camera coordinate system:
    #   X: forward (positive) / backward (negative)
    #   Y: left (negative) / right (positive) 
    #   Z: up (positive) / down (negative)
    # 
    # Example: If antenna is 0.5m behind camera and 0.3m above:
    # antenna_transform.set_translation(sl.Translation(-0.5, 0.0, 0.3))
    #
    # TODO: MEASURE YOUR ACTUAL ANTENNA POSITION AND UPDATE THESE VALUES!
    uuid = sl.CameraIdentifier(zed.get_camera_information().serial_number)
    
    antenna_transform = sl.Transform()
    # Measured antenna offset: 14cm behind camera, 42cm to the right
    # Camera coordinate system: X=forward, Y=left, Z=up
    antenna_x = -0.14  # 14cm behind = negative X
    antenna_y = -0.42  # 42cm to the right = negative Y (left is positive)
    antenna_z = 0.0    # Assuming same height (adjust if different)
    
    print(f"Using measured antenna offset: X={antenna_x}m, Y={antenna_y}m, Z={antenna_z}m")
    print("(14cm behind camera, 42cm to the right)")
    
    antenna_transform.set_translation(sl.Translation(antenna_x, antenna_y, antenna_z))
    fusion.subscribe(uuid, configuration, antenna_transform)

    # Enable positional tracking for Fusion object:
    positional_tracking_fusion_parameters = sl.PositionalTrackingFusionParameters()
    positional_tracking_fusion_parameters.enable_GNSS_fusion = True 
    gnss_calibration_parameters = sl.GNSSCalibrationParameters()
    gnss_calibration_parameters.target_yaw_uncertainty = 0.1
    gnss_calibration_parameters.enable_translation_uncertainty_target = False
    gnss_calibration_parameters.enable_reinitialization = True
    gnss_calibration_parameters.gnss_vio_reinit_threshold = 5
    positional_tracking_fusion_parameters.gnss_calibration_parameters = gnss_calibration_parameters
    fusion.enable_positionnal_tracking(positional_tracking_fusion_parameters)
    
    py_translation = sl.Translation()

    # Setup viewer:
    viewer = GenericDisplay()
    viewer.init(zed.get_camera_information().camera_model)
    print("Start grabbing data ...")
    
    while viewer.isAvailable():
        # Grab camera:
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed_pose = sl.Pose()
            # You can still use the classical getPosition for your application, just not that the position returned by this method
            # is the position without any GNSS/cameras fusion
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)

        # Get GNSS data:
        status, input_gnss = gnss_reader.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            # Display it on the Live Server
            viewer.updateRawGeoPoseData(input_gnss)

            # Publish GNSS data to Fusion
            ingest_error = fusion.ingest_gnss_data(input_gnss)
            if ingest_error != sl.FUSION_ERROR_CODE.SUCCESS:
                print("Ingest error occurred when ingesting GNSSData: ",ingest_error)

        # Process data and compute positions:
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_position = sl.Pose()

            # Get position into the ZED CAMERA coordinate system:
            current_state = fusion.get_position(fused_position)

            # Display it on OpenGL:
            rotation = fused_position.get_rotation_vector()
            translation = fused_position.get_translation(py_translation)
            text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
            viewer.updatePoseData(fused_position.pose_data(),text_translation,text_rotation, current_state) 

            # Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            # and GNSS. When the initialization is finish the getGeoPose will return sl.POSITIONAL_TRACKING_STATE.OK
            current_geopose = sl.GeoPose()
            current_geopose_status = fusion.get_geo_pose(current_geopose)
            if current_geopose_status == sl.GNSS_FUSION_STATUS.OK:
                viewer.updateGeoPoseData(current_geopose)
            """
            else:
                GNSS coordinate system to ZED coordinate system is not initialize yet
                The initialization between the coordinates system is basically an optimization problem that
                Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
            """

    gnss_reader.stop_thread() 
    fusion.close()
    zed.close()
   
    
if __name__ == '__main__' : 
    main()