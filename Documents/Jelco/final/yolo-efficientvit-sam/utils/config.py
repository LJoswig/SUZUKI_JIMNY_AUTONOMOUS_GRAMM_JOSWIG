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

# Centerline curve fit: False uses the existing row-wise Gaussian smoothing.
CENTERLINE_USE_CURVE_FIT = False
CENTERLINE_POLYNOMIAL_DEGREE = 2  # Allowed: 1, 2, or 3
CENTERLINE_USE_TEMPORAL_FILTER = False
CENTERLINE_TEMPORAL_FILTER_TYPE = "ema"  # "ema" or "kalman"
CENTERLINE_TEMPORAL_ALPHA = 0.35  # EMA only: lower is smoother, higher responds faster
CENTERLINE_KALMAN_PROCESS_NOISE = 2500.0  # Acceleration variance, pixels^2 / second^4
CENTERLINE_KALMAN_MEASUREMENT_NOISE = 25.0  # Position variance, pixels^2
CENTERLINE_KALMAN_MAX_GAP_SECONDS = 0.5  # Reset rows unseen longer than this

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

# Display at native camera resolution (no scaling needed)
# If you want to display smaller, change this value (e.g., 0.8 for 80% size)
DISPLAY_SCALE = 1.0

# ============================================================================
# Output Parameters
# ============================================================================
OUTPUT_DIR = "road_captures"
SAVE_DEPTH_MAPS = True
SAVE_FULL_RESOLUTION = True  # Save at camera's native resolution

# ============================================================================
# Warmup Settings
# ============================================================================
WARMUP_ITERATIONS = 5

# ============================================================================
# Runtime Options
# ============================================================================
VERBOSE = True