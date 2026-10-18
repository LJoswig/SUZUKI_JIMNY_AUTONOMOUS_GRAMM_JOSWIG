"""
SAM (Segment Anything Model) utilities
Preprocessing, prompt generation, mask selection, and postprocessing
"""

import cv2
import numpy as np
from .config import SAM_NORM_MEAN, SAM_NORM_STD


def preprocess_for_encoder(img: np.ndarray, encoder_size: int) -> tuple:
    """
    Preprocess image for SAM encoder with letterbox resizing
    
    Args:
        img: Input BGR image
        encoder_size: Target encoder size (square)
        
    Returns:
        Tuple of (preprocessed tensor, transform info dict)
    """
    h, w = img.shape[:2]
    scale = encoder_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((encoder_size, encoder_size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized.astype(np.float32)
    
    input_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_image = (input_image - SAM_NORM_MEAN) / SAM_NORM_STD
    input_tensor = input_image.transpose(2, 0, 1)[None, :, :, :]
    
    transform_info = {
        'scale': scale,
        'new_h': new_h,
        'new_w': new_w
    }
    
    return input_tensor.astype(np.float32), transform_info


def box_to_sam_prompt(box: np.ndarray, 
                      img_w: int, 
                      img_h: int, 
                      encoder_size: int, 
                      sam_transform_info: dict) -> tuple:
    """
    Convert bounding box to SAM point prompt
    
    Args:
        box: Bounding box in xyxy format
        img_w: Image width
        img_h: Image height
        encoder_size: Encoder input size
        sam_transform_info: Transform info from preprocess_for_encoder
        
    Returns:
        Tuple of (input_point, input_label) for SAM decoder
    """
    x1, y1, x2, y2 = box
    
    # Get box center in original image
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Apply letterbox transformation
    scale = sam_transform_info['scale']
    prompt_x = center_x * scale
    prompt_y = center_y * scale
    
    # Scale to 1024×1024 (SAM's native coordinate system)
    scale_to_1024 = 1024.0 / encoder_size
    final_prompt_x = prompt_x * scale_to_1024
    final_prompt_y = prompt_y * scale_to_1024
    
    input_point = np.array([[[final_prompt_x, final_prompt_y]]], dtype=np.float32)
    input_label = np.array([[1]], dtype=np.float32)  # 1 = foreground point
    
    return input_point, input_label


def select_best_mask(masks_logits: np.ndarray, iou_predictions: np.ndarray) -> int:
    """
    Select the most selective mask with reasonable IoU
    
    Strategy:
    - Filter masks with IoU > 0.5 (reasonable quality)
    - Among valid masks, choose one with lowest mean (most selective)
    
    Args:
        masks_logits: Mask logits of shape (1, N, H, W)
        iou_predictions: IoU predictions of shape (1, N)
        
    Returns:
        Index of the best mask
    """
    num_masks = masks_logits.shape[1]
    
    mask_means = []
    for idx in range(num_masks):
        mask_logit = masks_logits[0, idx, :, :]
        mask_means.append(mask_logit.mean())
    
    # Filter masks with reasonable IoU (> 0.5)
    valid_masks = [(idx, mask_means[idx]) for idx in range(num_masks) 
                   if iou_predictions[0, idx] > 0.5]
    
    if not valid_masks:
        # Fallback: use highest IoU
        best_mask_idx = np.argmax(iou_predictions[0])
    else:
        # Choose the one with LOWEST mean (most selective)
        best_mask_idx = min(valid_masks, key=lambda x: x[1])[0]
    
    return best_mask_idx


def postprocess_mask(mask_256: np.ndarray, 
                     img_h: int, 
                     img_w: int, 
                     encoder_size: int, 
                     sam_transform_info: dict, 
                     box: np.ndarray = None) -> np.ndarray:
    """
    Postprocess SAM mask to original image size with optional box constraint
    
    Args:
        mask_256: Binary mask of shape (256, 256)
        img_h: Target image height
        img_w: Target image width
        encoder_size: Encoder input size
        sam_transform_info: Transform info from preprocess_for_encoder
        box: Optional bounding box to constrain mask (xyxy format)
        
    Returns:
        Binary mask of shape (img_h, img_w)
    """
    # Upscale from 256×256 to encoder size (512×512)
    mask_512 = cv2.resize(mask_256, (encoder_size, encoder_size), 
                         interpolation=cv2.INTER_LINEAR)
    
    # Crop to actual image region (remove padding)
    new_h = sam_transform_info['new_h']
    new_w = sam_transform_info['new_w']
    mask_cropped = mask_512[:new_h, :new_w]
    
    # Resize back to target size
    mask_target = cv2.resize(mask_cropped, (img_w, img_h), 
                            interpolation=cv2.INTER_LINEAR)
    
    mask_binary = (mask_target > 0.5).astype(np.uint8)
    
    # Optionally constrain to box
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        box_mask = np.zeros_like(mask_binary)
        box_mask[y1:y2, x1:x2] = 1
        mask_binary = mask_binary * box_mask
    
    return mask_binary