"""
YOLO detection utilities - OPTIMIZED VERSION
Preprocessing, postprocessing, and NMS functions
Works with RGB input directly
"""

import cv2
import numpy as np


def crop_image_vertical(img: np.ndarray, vertical_limit: float) -> tuple:
    """
    Crop image to only keep the top portion (to exclude car hood).
    
    Args:
        img: Input RGB image
        vertical_limit: Fraction of image height to keep (e.g., 0.6 = top 60%)
        
    Returns:
        Tuple of (cropped image, crop_info dict)
    """
    orig_h, orig_w = img.shape[:2]
    
    if vertical_limit >= 1.0:
        # No cropping needed
        return img, {'cropped': False, 'orig_h': orig_h, 'orig_w': orig_w}
    
    # Calculate crop height
    crop_h = int(orig_h * vertical_limit)
    
    # Crop from top
    cropped = img[:crop_h, :, :]
    
    crop_info = {
        'cropped': True,
        'orig_h': orig_h,
        'orig_w': orig_w,
        'crop_h': crop_h,
        'crop_w': orig_w
    }
    
    return cropped, crop_info


def apply_histogram_equalization(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast
    Works on RGB input
    
    Args:
        image_rgb: Input RGB image
        
    Returns:
        Equalized RGB image
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return image_rgb
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    # Convert back to RGB
    image_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)
    
    return image_equalized


def letterbox_image(img: np.ndarray, target_size: int, crop_info: dict = None) -> tuple:
    """
    Resize image with aspect ratio preservation using letterboxing
    
    Args:
        img: Input RGB image (possibly cropped)
        target_size: Target size (square)
        crop_info: Optional crop information from crop_image_vertical
        
    Returns:
        Tuple of (letterboxed image, transform info dict)
    """
    orig_h, orig_w = img.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    letterboxed = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    letterboxed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    transform_info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'scale': scale,
        'new_h': new_h,
        'new_w': new_w,
        'pad_top': pad_top,
        'pad_left': pad_left
    }
    
    # Include crop info if provided
    if crop_info is not None:
        transform_info['crop_info'] = crop_info
    
    return letterboxed, transform_info


def preprocess_for_yolo(img_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess RGB image for YOLO inference
    
    Args:
        img_rgb: Input RGB image (uint8)
        
    Returns:
        Preprocessed tensor (1, 3, H, W) float32
    """
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    img_batch = img_chw.reshape(1, *img_chw.shape)
    return img_batch


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    
    Args:
        boxes: Array of shape (N, 4) in xywh format
        
    Returns:
        Array of shape (N, 4) in xyxy format
    """
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return boxes_xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def postprocess_yolo_output(output: np.ndarray, 
                            conf_threshold: float, 
                            iou_threshold: float, 
                            transform_info: dict) -> tuple:
    """
    Postprocess YOLO output to get final bounding boxes
    
    Args:
        output: Raw YOLO output tensor
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        transform_info: Transform info from letterbox_image (includes crop_info if cropped)
        
    Returns:
        Tuple of (boxes, confidences) in original image coordinates
    """
    predictions = np.squeeze(output)
    
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    # Convert to xyxy format
    boxes_xyxy = xywh2xyxy(boxes)
    
    # Apply NMS
    keep = nms(boxes_xyxy, confidences, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    
    # Transform back to cropped image coordinates
    scale = transform_info['scale']
    pad_left = transform_info['pad_left']
    pad_top = transform_info['pad_top']
    orig_w = transform_info['orig_w']
    orig_h = transform_info['orig_h']
    
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale
    
    # Clip to cropped image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
    
    # If image was cropped, transform boxes back to full original image coordinates
    if 'crop_info' in transform_info:
        crop_info = transform_info['crop_info']
        if crop_info['cropped']:
            # Boxes are currently in cropped image coordinates
            # They're already in the correct X coordinates
            # Y coordinates are already correct relative to crop
            # No transformation needed - boxes are in top portion of original image
            
            # Just need to make sure we clip to the actual original dimensions
            full_orig_h = crop_info['orig_h']
            full_orig_w = crop_info['orig_w']
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, full_orig_w)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, crop_info['crop_h'])
    
    return boxes_xyxy, confidences