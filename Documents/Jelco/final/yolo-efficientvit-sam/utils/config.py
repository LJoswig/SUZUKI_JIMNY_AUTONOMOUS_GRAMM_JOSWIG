#!/usr/bin/env python3
"""
Configuration file for ZED + YOLO + SAM real-time system using TensorRT
OPTIMIZED VERSION
"""

import numpy as np
import os

# ============================================================================
# Model Paths (TensorRT Engines)
# ============================================================================
YOLO_ENGINE_PATH = "engines/yolov8-dirt-roads_he.engine"
ENCODER_ENGINE_PATH = "engines/sam_l1_encoder.engine"
DECODER_ENGINE_PATH = "engines/sam_l1_decoder.engine"

# ============================================================================
# YOLO Configuration
# ============================================================================
YOLO_INPUT_SIZE = 416
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
YOLO_CLASS_NAMES = {0: "dirt road"}

# Input image cropping (to avoid detecting car hood)
# Crop to only use the top X% of the image before detection
# 0.6 = top 60%, 0.7 = top 70%, etc.
# This prevents the hood from ever being seen by the detector
INPUT_CROP_VERTICAL_LIMIT = 0.8  # Adjust this value to fine-tune

# ============================================================================
# SAM Configuration
# ============================================================================
ENCODER_SIZE = 512
SAM_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# ============================================================================
# ZED Camera Parameters
# ============================================================================
ZED_RESOLUTION = "SVGA"  # Options: "HD2K", "HD1080", "HD720", "SVGA", "AUTO"
ZED_FPS = 30
ZED_DEPTH_MODE = "NEURAL"  # Options: "NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"

# Select "svo" to replay a ZED recording, or "live" to use the camera.
INPUT_MODE = "svo"
SVO_FILE_PATH = "/home/user/Documents/testfahrt_20261021_181107.svo2"  # Matches CENTERLINE_MOTION_ROSBAG_URI
SVO_REAL_TIME_MODE =  True # False processes as fast as inference allows

# ============================================================================
# Processing Parameters - OPTIMIZED
# ============================================================================
# Processing resolution for both YOLO and SAM
# This is the resolution images are downscaled to before any neural network processing
# Smaller = faster processing, larger = better quality
# 416×260 maintains aspect ratio and ensures largest dimension ≤ 416
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 400

# Histogram equalization (improves detection in varying lighting)
USE_HISTOGRAM_EQUALIZATION = True

# Fit a quadratic curve before temporal filtering.
CENTERLINE_USE_CURVE_FIT = True
CENTERLINE_POLYNOMIAL_DEGREE = 2  # Allowed: 1, 2, or 3
CENTERLINE_USE_TEMPORAL_FILTER = True
CENTERLINE_TEMPORAL_FILTER_TYPE = "kalman"  # "ema" or "kalman"
# Kalman sensor mode:
# - "visual_only": centerline-only Kalman (no motion topics required)
# - "motion_aided": Kalman prediction is conditioned on motion topics
CENTERLINE_KALMAN_SENSOR_MODE = "visual_only"  # "visual_only" or "motion_aided"
CENTERLINE_KALMAN_VISUAL_ONLY = False # Backward-compatible alias for visual_only mode
CENTERLINE_TEMPORAL_ALPHA = 0.35  # default 0.35 EMA only: lower is smoother, higher responds faster
# Kalman state: centerline x at three anchor rows (fractions of image height),
# placed inside the band the road usually occupies (rows ~180-300 of 400).
# Baseline tuned on testfahrt_20261021_181107 at ~5 Hz.
CENTERLINE_KALMAN_ANCHOR_ROWS = (0.475, 0.6, 0.725)
CENTERLINE_KALMAN_SHARED_PROCESS_NOISE = 200.0  # Common sideways drift of all anchors, pixels^2 / second
CENTERLINE_KALMAN_INDEPENDENT_PROCESS_NOISE = 80.0  # Per-anchor drift (curvature changes), pixels^2 / second
CENTERLINE_KALMAN_MEASUREMENT_NOISE = 16.0  # Per-row centerline variance, pixels^2
CENTERLINE_KALMAN_MAX_COAST_SECONDS = 1.0
CENTERLINE_KALMAN_MAX_POSITION_STD = 35.0  # Maximum predicted x uncertainty, pixels
CENTERLINE_KALMAN_INNOVATION_GATE = 11.34  # 99% chi-square gate for 3 anchors
CENTERLINE_KALMAN_RELOCK_AFTER_REJECTIONS = 3  # Consecutive agreeing gate rejections before re-locking
CENTERLINE_KALMAN_MIN_CONFIDENCE = 0.25
CENTERLINE_KALMAN_MIN_VERTICAL_SPAN = 0.25
CENTERLINE_KALMAN_MIN_POINTS = 8

# Motion-aided mode only: yaw rate shifts all anchors sideways by gain * yaw_rate * dt.
# Fitted on testfahrt_20261021_181107 (IMU angular_velocity.z), processing-resolution pixels / rad.
CENTERLINE_KALMAN_YAW_RATE_GAIN = -60.0

# Motion input source (motion-aided Kalman and path memory odometry):
# - "none": disable motion provider
# - "rosbag": consume motion signals from ROS2 bag replay
# - "live": subscribe to live ROS2 topics
CENTERLINE_MOTION_SOURCE = "rosbag"  # "none", "rosbag", or "live"
CENTERLINE_MOTION_ROSBAG_URI = "/home/user/Documents/testfahrt_20261021_181107_rosbag"
CENTERLINE_MOTION_MAX_SIGNAL_AGE_SECONDS = 0.15
CENTERLINE_MOTION_LOOKAHEAD_SECONDS = 0.02

# Topic names used by both rosbag and live providers.
CENTERLINE_TOPIC_ACKERMANN_CMD = "/ackermann_cmd"
CENTERLINE_TOPIC_STEERING_FEEDBACK = "/steering_feedback"
CENTERLINE_TOPIC_IMU_ATTITUDE = "/imu_attitude"
CENTERLINE_TOPIC_FEEDBACK_DATA = "/feedback_data"  # data[1] = measured speed, m/s

# Published for the road follower: the path the follower uses (fused, in metres around
# the rear axle at image time, header.stamp = image time) and the centerline status.
# Needs ROS sourced; nothing listens to these topics yet.
CENTERLINE_ROS_PUBLISH_ENABLED = True
CENTERLINE_TOPIC_PATH = "/centerline/path"  # nav_msgs/Path, empty while lost
CENTERLINE_TOPIC_STATUS = "/centerline/status"  # std_msgs/String: observed, coasting, remembered or lost
CENTERLINE_PATH_FRAME_ID = "base_link"

# Path memory: keep the last observed centerline in an odometry frame and replay it
# once the Kalman filter has lost the road. Needs CENTERLINE_MOTION_SOURCE "rosbag"
# (with a bag recorded alongside the SVO) or "live".
CENTERLINE_PATH_MEMORY_ENABLED = True
CENTERLINE_PATH_MEMORY_MAX_SECONDS = 3.0
CENTERLINE_PATH_MEMORY_MAX_DISTANCE = 8.0  # Metres driven since the road was last seen
CENTERLINE_PATH_MEMORY_MIN_REMAINING = 5.0  # Remembered path must still reach this far ahead of the rear axle, metres
# Path fusion: blend each new observation into the stored path in the odom frame.
CENTERLINE_PATH_FUSION_WEIGHT = 0.5  # Weight of a new observation; 1.0 = no fusion (newest replaces)
CENTERLINE_PATH_FUSION_MISMATCH_DISTANCE = 0.8  # Metres; a larger median lateral difference is a mismatch
CENTERLINE_PATH_FUSION_MISMATCH_FRAMES = 4  # Consecutive agreeing mismatches before the path is replaced (~0.8 s at 5 Hz)

# Wheel/IMU odometry (validated against GPS on testfahrt_20261021_181107).
CENTERLINE_ODOM_YAW_RATE_SIGN = -1.0  # The cube IMU reports positive yaw rate for right turns
CENTERLINE_ODOM_MAX_SPEED_AGE_SECONDS = 0.5  # /feedback_data arrives at ~5 Hz

# Camera mounting for flat-ground projection. base_link = rear axle centre on the ground.
CENTERLINE_CAMERA_HEIGHT = 1.78  # Metres above ground; measured, matches the ZED depth fit
CENTERLINE_CAMERA_PITCH_DOWN_DEG = 8.5  # Fitted from ZED depth
CENTERLINE_CAMERA_X = 1.27  # Metres ahead of the rear axle (measured; ~1.0 m behind the front axle)
CENTERLINE_CAMERA_Y = 0.0  # Metres left of the vehicle centreline
CENTERLINE_CAMERA_YAW_DEG = 0.0  # Camera heading relative to the vehicle, positive = left
CENTERLINE_GROUND_MAX_RANGE = 25.0  # Ignore centerline points further away, metres

# ============================================================================
# Road Follower (pure pursuit, shadow mode first)
# ============================================================================
# Steering output uses the /ackermann_cmd convention: steering-wheel degrees, positive = right.
FOLLOWER_WHEELBASE = 2.27  # Metres; measured camera-to-axle distances (Jimny spec 2.25)
FOLLOWER_STEERING_RATIO = 20.0  # Steering-wheel deg per road-wheel deg; fitted on testfahrt_20261021_181107 (nominal 14)
FOLLOWER_STEERING_SIGN = -1.0  # A left turn (positive curvature) needs a negative steering_angle
FOLLOWER_LOOKAHEAD_DISTANCE = 6.0  # Metres from the rear axle; must exceed CENTERLINE_CAMERA_X + ~4 m
FOLLOWER_MAX_STEERING_WHEEL_DEG = 620.0  # Same clamp as the steering node
FOLLOWER_MAX_STEERING_RATE_DEG_S = 120.0  # The steering actuator is slow; matches the RC rate scale
FOLLOWER_TARGET_SPEED = 1.39  # m/s (5 km/h)
FOLLOWER_MAX_PATH_AGE_SECONDS = 1.0  # Stop when the newest path is older; tune from live message intervals
FOLLOWER_RATE_HZ = 20.0

# ============================================================================
# Visualization - OPTIMIZED
# ============================================================================
MASK_OPACITY = 0.4
BOX_THICKNESS = 2
BOX_COLOR = (0, 255, 0)  # Green in RGB
MASK_COLOR = (0, 255, 0)  # Green in RGB
SHOW_LABELS = True
SHOW_MASKS = True
SHOW_BOXES = True
# Shadow road follower in the preview: lookahead point (yellow), arc the steering
# command would drive (magenta) and the command itself. Nothing is sent to the car.
SHOW_FOLLOWER_OVERLAY = True

# Display at native camera resolution (no scaling needed)
# If you want to display smaller, change this value (e.g., 0.8 for 80% size)
DISPLAY_SCALE = 1.0

# ============================================================================
# Output Parameters
# ============================================================================
OUTPUT_DIR = "road_captures"
SAVE_DEPTH_MAPS = True
SAVE_FULL_RESOLUTION = True  # Save at camera's native resolution
SAVE_ANNOTATED_VIDEO = False
ANNOTATED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "road_segmentation_new_kalman.mp4")

# Live runs only: record the camera (SVO2, every processed frame) and a matching rosbag
# into RECORD_RUN_DIR/run_<date>_<time>/, so the drive can be replayed later with
# tools/shadow_follower.py. Needs ROS sourced for the bag.
RECORD_RUN_ENABLED = False
RECORD_RUN_DIR = "/home/user/recordings"

# SSH-friendly preview; forward this port in VS Code and open the printed URL.
ENABLE_REMOTE_PREVIEW = True
REMOTE_PREVIEW_HOST = "127.0.0.1"
REMOTE_PREVIEW_PORT = 8765
REMOTE_PREVIEW_JPEG_QUALITY = 75
SHOW_LOCAL_WINDOW = False

# ============================================================================
# Warmup Settings
# ============================================================================
WARMUP_ITERATIONS = 5

# ============================================================================
# Runtime Options
# ============================================================================
VERBOSE = True
