#!/usr/bin/env python3
"""
ZED camera handler - OPTIMIZED VERSION
Returns RGB images directly (no unnecessary BGR conversion)
"""

import cv2
import numpy as np
import pyzed.sl as sl
from typing import Tuple, Optional, Dict
from . import config


class ZEDCamera:
    """Manages ZED camera capture and pose - optimized for performance"""
    
    def __init__(self):
        self.zed = sl.Camera()
        self.image_zed = sl.Mat()
        self.depth_map = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        self.camera_width = 0
        self.camera_height = 0
        self.camera_calibration = {}
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ZED camera"""
        print("Initializing ZED camera...")
        
        init_params = sl.InitParameters()
        
        # Set resolution
        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "SVGA": sl.RESOLUTION.SVGA,
            "AUTO": sl.RESOLUTION.AUTO
        }
        init_params.camera_resolution = resolution_map.get(config.ZED_RESOLUTION, sl.RESOLUTION.HD720)
        
        init_params.camera_fps = config.ZED_FPS
        
        # Set depth mode
        depth_map = {
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "NONE": sl.DEPTH_MODE.NONE
        }
        init_params.depth_mode = depth_map.get(config.ZED_DEPTH_MODE, sl.DEPTH_MODE.NEURAL)
        self.depth_enabled = init_params.depth_mode != sl.DEPTH_MODE.NONE
        
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
        
        # Enable positional tracking (needs depth)
        if self.depth_enabled:
            tracking_params = sl.PositionalTrackingParameters()
            tracking_params.enable_imu_fusion = True
            err = self.zed.enable_positional_tracking(tracking_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"⚠ Positional tracking failed: {err}")
        
        # Get camera info
        camera_info = self.zed.get_camera_information()
        self.camera_width = camera_info.camera_configuration.resolution.width
        self.camera_height = camera_info.camera_configuration.resolution.height
        self.camera_fps = camera_info.camera_configuration.fps
        
        # Get calibration
        calibration = camera_info.camera_configuration.calibration_parameters
        self.camera_calibration = {
            'fx': calibration.left_cam.fx,
            'fy': calibration.left_cam.fy,
            'cx': calibration.left_cam.cx,
            'cy': calibration.left_cam.cy,
            'width': self.camera_width,
            'height': self.camera_height
        }
        
        print(f"✓ ZED Camera opened:")
        print(f"  Serial: {camera_info.serial_number}")
        print(f"  Resolution: {self.camera_width}x{self.camera_height}")
        print(f"  FPS: {camera_info.camera_configuration.fps}")
    
    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Grab frame from camera - OPTIMIZED VERSION
        Returns RGB image (not BGR) to avoid extra conversion for neural networks
        Returns: (success, image_rgb, depth_map)
        """
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return False, None, None
        
        # Get left image - retrieve as BGRA
        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        image = self.image_zed.get_data()
        
        # OPTIMIZED: Convert BGRA → RGB in one step (skip BGR intermediate)
        # This is faster than BGRA→BGR→RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Get depth map (None with ZED_DEPTH_MODE = "NONE")
        depth = None
        if self.depth_enabled:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            depth = self.depth_map.get_data()
        
        return True, image_rgb, depth
    
    def get_pose(self) -> Dict:
        """Get current camera pose"""
        camera_pose = sl.Pose()
        self.zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        
        rotation_vector = camera_pose.get_rotation_vector()
        
        return {
            'rotation': {
                'pitch': float(rotation_vector[0]),
                'yaw': float(rotation_vector[1]),
                'roll': float(rotation_vector[2])
            },
            'heading_deg': float(np.degrees(rotation_vector[2]))  # Roll = heading
        }
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution (width, height)"""
        return self.camera_width, self.camera_height

    def get_fps(self) -> float:
        """Get the camera frame rate"""
        return float(self.camera_fps)

    def get_timestamp_seconds(self) -> Optional[float]:
        """Get the capture timestamp for the most recently grabbed image."""
        try:
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            return float(timestamp.get_nanoseconds()) * 1e-9
        except Exception:
            return None
    
    def get_calibration(self) -> Dict:
        """Get camera calibration parameters"""
        return self.camera_calibration.copy()
    
    def enable_recording(self, svo_path) -> bool:
        """Record to an SVO2 file; only frames fetched with grab_frame() are recorded"""
        recording_params = sl.RecordingParameters()
        recording_params.video_filename = str(svo_path)
        recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H265
        err = self.zed.enable_recording(recording_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"⚠ SVO recording failed: {err}")
            return False
        self.recording = True
        print(f"✓ Recording SVO: {svo_path}")
        return True

    def close(self):
        """Close camera"""
        if hasattr(self, 'zed'):
            if getattr(self, 'recording', False):
                self.zed.disable_recording()
                self.recording = False
            self.zed.close()
            print("✓ ZED camera closed")