#!/usr/bin/env python3

import cv2
import numpy as np
import pyzed.sl as sl
from typing import Tuple, Optional, Dict
from . import config


class SVOReplayCamera:
    """
    Replay a recorded ZED SVO/SVO2 file with the same interface as ZEDCamera.

    Returns:
        success, image_rgb, depth_map
    """

    def __init__(self, svo_path: str):
        self.svo_path = svo_path

        self.zed = sl.Camera()
        self.image_zed = sl.Mat()
        self.depth_map = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()

        self.camera_width = 0
        self.camera_height = 0
        self.camera_calibration = {}

        self._initialize()

    def _initialize(self):
        print(f"Opening SVO replay: {self.svo_path}")

        init_params = sl.InitParameters()

        try:
            init_params.set_from_svo_file(self.svo_path)
        except AttributeError:
            input_type = sl.InputType()
            input_type.set_from_svo_file(self.svo_path)
            init_params.input = input_type

        try:
            init_params.svo_real_time_mode = config.SVO_REAL_TIME_MODE
        except Exception:
            pass

        depth_map = {
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "NONE": sl.DEPTH_MODE.NONE,
        }

        init_params.depth_mode = depth_map.get(
            config.camera_depth_mode(),
            sl.DEPTH_MODE.NEURAL
        )
        # Depth map + positional tracking only when asked for; the auto-stop only needs the point cloud
        self.depth_enabled = depth_map.get(
            config.ZED_DEPTH_MODE,
            sl.DEPTH_MODE.NEURAL
        ) != sl.DEPTH_MODE.NONE

        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO file: {err}")

        if self.depth_enabled:  # positional tracking needs depth
            tracking_params = sl.PositionalTrackingParameters()
            tracking_params.enable_imu_fusion = True
            err = self.zed.enable_positional_tracking(tracking_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Warning: positional tracking failed during SVO replay: {err}")

        camera_info = self.zed.get_camera_information()
        self.camera_width = camera_info.camera_configuration.resolution.width
        self.camera_height = camera_info.camera_configuration.resolution.height
        self.camera_fps = camera_info.camera_configuration.fps

        calibration = camera_info.camera_configuration.calibration_parameters
        self.camera_calibration = {
            "fx": calibration.left_cam.fx,
            "fy": calibration.left_cam.fy,
            "cx": calibration.left_cam.cx,
            "cy": calibration.left_cam.cy,
            "width": self.camera_width,
            "height": self.camera_height,
        }

        print("✓ SVO opened")
        print(f"  Resolution: {self.camera_width}x{self.camera_height}")
        print(f"  FPS: {camera_info.camera_configuration.fps}")

        # Replay window (SVO_START_SECONDS / SVO_END_SECONDS), converted to frames
        total_frames = self.zed.get_svo_number_of_frames()
        fps = float(self.camera_fps) if self.camera_fps else 30.0
        start_seconds = max(0.0, float(config.SVO_START_SECONDS or 0.0))
        self.start_frame = min(int(round(start_seconds * fps)), max(total_frames - 1, 0))
        self.end_frame = total_frames
        if config.SVO_END_SECONDS is not None:
            self.end_frame = min(total_frames, int(round(float(config.SVO_END_SECONDS) * fps)))
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"SVO_END_SECONDS ({config.SVO_END_SECONDS}) must be after "
                f"SVO_START_SECONDS ({config.SVO_START_SECONDS})"
            )
        if self.start_frame > 0:
            self.zed.set_svo_position(self.start_frame)
        self._next_frame = self.start_frame
        print(f"  Length: {total_frames / fps:.1f} s ({total_frames} frames)")
        print(
            f"  Playing: {self.start_frame / fps:.1f} s to {self.end_frame / fps:.1f} s "
            f"(frames {self.start_frame}-{self.end_frame - 1})"
        )

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if self._next_frame >= self.end_frame:
            return False, None, None
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return False, None, None
        # After a grab the position is the frame just grabbed (also after a set_svo_position skip)
        self._next_frame = self.zed.get_svo_position() + 1

        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        image = self.image_zed.get_data()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        depth = None  # None with ZED_DEPTH_MODE = "NONE"
        if self.depth_enabled:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            depth = self.depth_map.get_data()

        return True, image_rgb, depth

    def get_pose(self) -> Dict:
        camera_pose = sl.Pose()
        self.zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)

        rotation_vector = camera_pose.get_rotation_vector()

        return {
            "rotation": {
                "pitch": float(rotation_vector[0]),
                "yaw": float(rotation_vector[1]),
                "roll": float(rotation_vector[2]),
            },
            "heading_deg": float(np.degrees(rotation_vector[2])),
        }

    def get_resolution(self) -> Tuple[int, int]:
        return self.camera_width, self.camera_height

    def get_fps(self) -> float:
        return float(self.camera_fps)

    def get_timestamp_seconds(self) -> Optional[float]:
        try:
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            return float(timestamp.get_nanoseconds()) * 1e-9
        except Exception:
            return None

    def get_calibration(self) -> Dict:
        return self.camera_calibration.copy()

    def close(self):
        if hasattr(self, "zed"):
            self.zed.close()
            print("✓ SVO replay closed")
