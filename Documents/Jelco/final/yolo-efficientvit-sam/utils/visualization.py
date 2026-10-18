"""
Visualization utilities
Create annotated output images with masks and bounding boxes
"""

import cv2
import numpy as np


def create_output_image(img: np.ndarray, 
                       masks: list, 
                       boxes: np.ndarray, 
                       confidences: np.ndarray,
                       mask_color: tuple = (0, 255, 0),
                       box_color: tuple = (0, 255, 0),
                       alpha: float = 0.4) -> np.ndarray:
    """
    Create visualization with masks and bounding boxes
    
    Args:
        img: Input image
        masks: List of binary masks
        boxes: Array of bounding boxes in xyxy format
        confidences: Array of confidence scores
        mask_color: RGB color for mask overlay
        box_color: RGB color for bounding boxes
        alpha: Transparency for mask overlay (0-1)
        
    Returns:
        Annotated output image
    """
    output_img = img.copy()
    h, w = img.shape[:2]
    
    # Combine all masks
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    
    # Apply mask overlay
    mask_overlay = np.zeros_like(output_img)
    mask_overlay[combined_mask > 0] = mask_color
    output_img = cv2.addWeighted(output_img, 1-alpha, mask_overlay, alpha, 0)
    
    # Draw bounding boxes and labels
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw label
        label = f"Road {i+1}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label background
        cv2.rectangle(output_img, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), box_color, -1)
        
        # Label text
        cv2.putText(output_img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return output_img


def print_detection_summary(boxes: np.ndarray, confidences: np.ndarray):
    """
    Print summary of detected objects
    
    Args:
        boxes: Array of bounding boxes
        confidences: Array of confidence scores
    """
    print(f"Detected {len(boxes)} road(s)")
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        print(f"  Road {i+1}: bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], confidence={conf:.3f}")


def print_segmentation_summary(masks: list, iou_scores: list):
    """
    Print summary of segmentation results
    
    Args:
        masks: List of binary masks
        iou_scores: List of IoU scores
    """
    for i, (mask, iou) in enumerate(zip(masks, iou_scores)):
        mask_pixels = mask.sum()
        print(f"  Mask {i+1}: {mask_pixels} pixels, IoU={iou:.3f}")


def print_performance_summary(yolo_time: float, 
                             encode_time: float, 
                             decode_times: list):
    """
    Print performance summary
    
    Args:
        yolo_time: YOLO inference time in seconds
        encode_time: SAM encoder inference time in seconds
        decode_times: List of SAM decoder inference times in seconds
    """
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    total_decode_time = sum(decode_times)
    avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0
    
    print(f"YOLO Detection:           {yolo_time*1000:7.2f} ms")
    print(f"SAM Encoder:              {encode_time*1000:7.2f} ms")
    print(f"SAM Decoder (avg):        {avg_decode_time*1000:7.2f} ms")
    print(f"SAM Decoder (total):      {total_decode_time*1000:7.2f} ms")
    print("-"*70)
    
    total_time = yolo_time + encode_time + total_decode_time
    print(f"TOTAL PIPELINE TIME:      {total_time*1000:7.2f} ms")
    print(f"THROUGHPUT:               {1/total_time:7.1f} FPS")
    print("="*70)