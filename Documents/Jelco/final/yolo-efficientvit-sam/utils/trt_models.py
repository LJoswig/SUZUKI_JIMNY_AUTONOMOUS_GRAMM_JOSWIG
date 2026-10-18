# ============================================================================
# 2. trt_models.py
# (New file: TensorRT model wrappers for YOLO and SAM)
# ============================================================================
"""
TensorRT Engine model wrappers for YOLO and SAM2 inference
Provides a high-level API for the real-time pipeline.
"""
import numpy as np
import cv2
from typing import Tuple, List

from . import config
from .trt_engine import TRT_Engine
from .yolo_utils import letterbox_image, preprocess_for_yolo, postprocess_yolo_output, apply_histogram_equalization
from .sam_utils import preprocess_for_encoder, box_to_sam_prompt, select_best_mask, postprocess_mask

# Final mask threshold (used in SAM post-processing)
FINAL_MASK_THRESHOLD = 0.45 


class YOLOTrtInference:
    """YOLO inference with TensorRT Engine"""
    
    def __init__(self, engine_path=config.YOLO_ENGINE_PATH):
        if config.VERBOSE:
            print(f"\nLoading YOLO TRT Engine: {engine_path}")
        self.engine = TRT_Engine(engine_path)
        self.input_name = list(self.engine.inputs.keys())[0]
        self.output_name = list(self.engine.outputs.keys())[0]
        self.input_size = config.YOLO_INPUT_SIZE
        self.conf_threshold = config.CONF_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD

    def predict(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO detection on a BGR image.
        
        Args:
            img_bgr: BGR image at PROCESSING_WIDTH x PROCESSING_HEIGHT
            
        Returns:
            Tuple of (boxes_xyxy, confidences, class_ids) at the input image resolution.
        """
        
        # 1. Preprocess
        if config.USE_HISTOGRAM_EQUALIZATION:
            img_bgr = apply_histogram_equalization(img_bgr)
        
        letterboxed_img, transform_info = letterbox_image(img_bgr, self.input_size)
        yolo_input_tensor = preprocess_for_yolo(letterboxed_img)
        
        # 2. Inference
        output = self.engine.inference({self.input_name: yolo_input_tensor})
        
        # 3. Postprocess
        boxes, confidences = postprocess_yolo_output(
            output[self.output_name], 
            self.conf_threshold, 
            self.iou_threshold, 
            transform_info
        )
        
        # Class IDs are always 0 for "dirt road" in this single-class model
        class_ids = np.zeros(len(boxes), dtype=np.int32)

        return boxes, confidences, class_ids


class SAMTrtPredictor:
    """SAM (EfficientViT) inference with TensorRT Engines"""
    
    def __init__(self, encoder_path=config.ENCODER_ENGINE_PATH, decoder_path=config.DECODER_ENGINE_PATH):
        if config.VERBOSE:
            print(f"Loading SAM Encoder TRT Engine: {encoder_path}")
        self.encoder = TRT_Engine(encoder_path)
        self.encoder_input_name = list(self.encoder.inputs.keys())[0]
        self.encoder_output_name = list(self.encoder.outputs.keys())[0]
        
        if config.VERBOSE:
            print(f"Loading SAM Decoder TRT Engine: {decoder_path}")
        self.decoder = TRT_Engine(decoder_path)
        self.decoder_output_names = list(self.decoder.outputs.keys())
        
        self.encoder_size = config.ENCODER_SIZE
        self.original_image = None
        self.image_embeddings = None
        self.sam_transform_info = None

    def set_image(self, image_rgb_full: np.ndarray):
        """Precompute image embedding for a new image"""
        self.original_image = image_rgb_full
        
        # 1. Preprocess for encoder
        encoder_input_tensor, self.sam_transform_info = preprocess_for_encoder(
            cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR), # SAM util expects BGR
            self.encoder_size
        )
        
        # 2. Encoder Inference
        encoder_output = self.encoder.inference({self.encoder_input_name: encoder_input_tensor})
        self.image_embeddings = encoder_output[self.encoder_output_name]
        
    def predict(self, boxes: np.ndarray) -> List[np.ndarray]:
        """
        Run SAM segmentation using precomputed embeddings and YOLO boxes.
        
        Args:
            boxes: Bounding boxes in xyxy format at full resolution.
            
        Returns:
            List of binary masks (np.ndarray) at full resolution.
        """
        if self.image_embeddings is None or self.original_image is None:
            raise ValueError("Must call set_image() before calling predict()")
        
        all_masks = []
        img_h, img_w = self.original_image.shape[:2]
        
        for box in boxes:
            # 1. Prepare prompt
            input_point, input_label = box_to_sam_prompt(
                box, img_w, img_h, self.encoder_size, self.sam_transform_info
            )
            
            # Decoder inputs
            dummy_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            dummy_has_mask_input = np.zeros((1,), dtype=np.float32)
            
            decoder_inputs = {
                "image_embeddings": self.image_embeddings,
                "point_coords": input_point,
                "point_labels": input_label,
                "mask_input": dummy_mask_input,
                "has_mask_input": dummy_has_mask_input
            }
            
            # 2. Decoder Inference
            outputs = self.decoder.inference(decoder_inputs)
            mask_output = outputs[self.decoder_output_names[0]]
            iou_output = outputs[self.decoder_output_names[1]]
            
            # 3. Postprocess
            mask = select_best_mask(mask_output, iou_output)
            
            # Postprocess and resize the best mask to original image size
            binary_mask = postprocess_mask(
                mask.squeeze(), 
                img_h, 
                img_w, 
                self.encoder_size, 
                self.sam_transform_info,
                box=box
            )
            all_masks.append(binary_mask)
            
        return all_masks