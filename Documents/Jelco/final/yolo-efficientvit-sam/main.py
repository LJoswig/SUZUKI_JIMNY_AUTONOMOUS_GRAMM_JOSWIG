#!/usr/bin/env python3
"""
Real-time ZED + YOLO + SAM road segmentation pipeline
FINAL OPTIMIZED VERSION
- RGB throughout (BGR only at display)
- Processing at 416×260 for both YOLO and SAM
- Crop-first mask pipeline
- Display at native 960×600
- Pre-allocated buffers
"""

import cv2
import numpy as np
import time
import json
import os
from collections import deque

# Import from our modules
from utils.config import *
from utils.trt_engine import TRT_Engine
from utils.yolo_utils import (
    crop_image_vertical,
    apply_histogram_equalization,
    letterbox_image,
    preprocess_for_yolo,
    postprocess_yolo_output
)
from utils.sam_utils import (
    preprocess_for_encoder,
    box_to_sam_prompt,
    select_best_mask,
    postprocess_mask
)
from utils.warmup import warmup_engines
from utils.visualization import create_output_image
from utils.zed_camera import ZEDCamera


class PerformanceMonitor:
    """Track FPS and performance metrics"""
    
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
    
    def update(self):
        current_time = time.perf_counter()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if len(self.times) == 0:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


class DetailedInferenceTimer:
    """Track detailed timing for all pipeline components"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        # Timings
        self.frame_grab = deque(maxlen=window_size)
        self.downscale = deque(maxlen=window_size)
        self.yolo_preprocess = deque(maxlen=window_size)
        self.yolo_inference = deque(maxlen=window_size)
        self.yolo_postprocess = deque(maxlen=window_size)
        
        self.sam_preprocess = deque(maxlen=window_size)
        self.sam_encoder = deque(maxlen=window_size)
        self.sam_decoder = deque(maxlen=window_size)
        self.sam_postprocess = deque(maxlen=window_size)
        
        self.visualization = deque(maxlen=window_size)
        
        self.total_times = deque(maxlen=window_size)
    
    def add_measurement(self, frame_grab, downscale, yolo_pre, yolo_inf, yolo_post,
                       sam_pre, sam_enc, sam_dec, sam_post, viz):
        self.frame_grab.append(frame_grab)
        self.downscale.append(downscale)
        self.yolo_preprocess.append(yolo_pre)
        self.yolo_inference.append(yolo_inf)
        self.yolo_postprocess.append(yolo_post)
        
        self.sam_preprocess.append(sam_pre)
        self.sam_encoder.append(sam_enc)
        self.sam_decoder.append(sam_dec)
        self.sam_postprocess.append(sam_post)
        
        self.visualization.append(viz)
        
        total = (frame_grab + downscale + yolo_pre + yolo_inf + yolo_post + 
                sam_pre + sam_enc + sam_dec + sam_post + viz)
        self.total_times.append(total)
    
    def get_averages(self):
        """Return average times in milliseconds"""
        def avg(deq):
            return (sum(deq) / len(deq) * 1000) if deq else 0
        
        return {
            'frame_grab': avg(self.frame_grab),
            'downscale': avg(self.downscale),
            'yolo_preprocess': avg(self.yolo_preprocess),
            'yolo_inference': avg(self.yolo_inference),
            'yolo_postprocess': avg(self.yolo_postprocess),
            'yolo_total': avg(self.yolo_preprocess) + avg(self.yolo_inference) + avg(self.yolo_postprocess),
            
            'sam_preprocess': avg(self.sam_preprocess),
            'sam_encoder': avg(self.sam_encoder),
            'sam_decoder': avg(self.sam_decoder),
            'sam_postprocess': avg(self.sam_postprocess),
            'sam_total': avg(self.sam_preprocess) + avg(self.sam_encoder) + avg(self.sam_decoder) + avg(self.sam_postprocess),
            
            'visualization': avg(self.visualization),
            'total': avg(self.total_times)
        }
    
    def get_inference_hz(self):
        """Get actual inference rate"""
        if len(self.total_times) == 0:
            return 0.0
        avg_time = sum(self.total_times) / len(self.total_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


class RealtimeRoadSegmentation:
    """Main real-time segmentation system - OPTIMIZED"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("  ZED + YOLO + SAM Real-Time Road Segmentation [OPTIMIZED]")
        print("="*70 + "\n")
        
        # Initialize ZED camera
        print("Initializing camera...")
        self.camera = ZEDCamera()
        cam_w, cam_h = self.camera.get_resolution()
        
        # Processing size (downscaled for speed)
        self.processing_width = PROCESSING_WIDTH
        self.processing_height = PROCESSING_HEIGHT
        
        # Scale factors (for converting boxes back to camera resolution)
        self.scale_x = cam_w / self.processing_width
        self.scale_y = cam_h / self.processing_height
        
        # Display size (at native camera resolution)
        self.display_width = int(cam_w * DISPLAY_SCALE)
        self.display_height = int(cam_h * DISPLAY_SCALE)
        
        # Load TensorRT engines
        print("\nLoading TensorRT engines...")
        self.yolo_engine = TRT_Engine(YOLO_ENGINE_PATH)
        self.encoder_engine = TRT_Engine(ENCODER_ENGINE_PATH)
        self.decoder_engine = TRT_Engine(DECODER_ENGINE_PATH)
        print("✓ Engines loaded successfully")
        
        # Warmup engines
        print("\nWarming up engines...")
        warmup_engines(
            self.yolo_engine,
            self.encoder_engine,
            self.decoder_engine,
            num_iterations=WARMUP_ITERATIONS
        )
        
        # Setup output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Performance monitoring
        self.fps_monitor = PerformanceMonitor()
        self.inference_timer = DetailedInferenceTimer(window_size=10)
        
        # Detection state
        self.frame_counter = 0
        self.current_detections = None
        
        # PRE-ALLOCATE BUFFERS (avoid repeated allocations)
        self.processing_buffer = np.zeros((self.processing_height, self.processing_width, 3), dtype=np.uint8)
        if DISPLAY_SCALE != 1.0:
            self.display_buffer = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        else:
            self.display_buffer = None  # Display at native resolution
        
        print(f"\nConfiguration:")
        print(f"  Camera: {cam_w}x{cam_h} @ {ZED_FPS} FPS")
        print(f"  Processing: {self.processing_width}x{self.processing_height} (both YOLO & SAM)")
        print(f"  Display: {self.display_width}x{self.display_height}")
        print(f"  OPTIMIZATIONS:")
        print(f"    - RGB pipeline (no unnecessary conversions)")
        print(f"    - Single downscale to {self.processing_width}x{self.processing_height}")
        print(f"    - Crop-first mask pipeline")
        print(f"    - Pre-allocated buffers")
        
        self.running = False
    
    def run_yolo_detection(self, img_rgb):
        """Run YOLO detection on RGB image with detailed timing"""
        # Crop image to exclude car hood (top portion only)
        cropped_img, crop_info = crop_image_vertical(img_rgb, INPUT_CROP_VERTICAL_LIMIT)
        
        # Preprocess
        start = time.perf_counter()
        if USE_HISTOGRAM_EQUALIZATION:
            cropped_img = apply_histogram_equalization(cropped_img)
        letterboxed_img, transform_info = letterbox_image(cropped_img, YOLO_INPUT_SIZE, crop_info)
        yolo_input = preprocess_for_yolo(letterboxed_img)
        preprocess_time = time.perf_counter() - start
        
        # Inference
        input_name = list(self.yolo_engine.inputs.keys())[0]
        start = time.perf_counter()
        output = self.yolo_engine.inference({input_name: yolo_input})
        inference_time = time.perf_counter() - start
        
        # Postprocess (boxes will be in original full image coordinates)
        start = time.perf_counter()
        output_name = list(output.keys())[0]
        boxes, confidences = postprocess_yolo_output(
            output[output_name],
            CONF_THRESHOLD,
            IOU_THRESHOLD,
            transform_info
        )
        postprocess_time = time.perf_counter() - start
        
        return boxes, confidences, preprocess_time, inference_time, postprocess_time
    
    def run_sam_segmentation(self, img_rgb, boxes):
        """Run SAM segmentation on RGB image with detailed timing"""
        h, w = img_rgb.shape[:2]
        
        # Preprocess for encoder
        start = time.perf_counter()
        encoder_input, sam_transform = preprocess_for_encoder(img_rgb, ENCODER_SIZE)
        preprocess_time = time.perf_counter() - start
        
        # Encode image once
        encoder_input_name = list(self.encoder_engine.inputs.keys())[0]
        start = time.perf_counter()
        encoder_results = self.encoder_engine.inference({encoder_input_name: encoder_input})
        encoder_time = time.perf_counter() - start
        
        image_embedding = encoder_results[list(encoder_results.keys())[0]]
        
        # Decode each box
        all_masks = []
        decoder_times = []
        postprocess_times = []
        
        for box in boxes:
            # Generate prompt
            input_point, input_label = box_to_sam_prompt(
                box, w, h, ENCODER_SIZE, sam_transform
            )
            
            # Prepare decoder inputs
            dummy_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros((1,), dtype=np.float32)
            
            decoder_input = {
                "image_embeddings": image_embedding,
                "point_coords": input_point,
                "point_labels": input_label,
                "mask_input": dummy_mask_input,
                "has_mask_input": has_mask_input,
            }
            
            # Decode
            start = time.perf_counter()
            decoder_results = self.decoder_engine.inference(decoder_input)
            decoder_times.append(time.perf_counter() - start)
            
            # Postprocess mask
            start = time.perf_counter()
            masks_logits = decoder_results['masks']
            iou_predictions = decoder_results['iou_predictions']
            
            best_mask_idx = select_best_mask(masks_logits, iou_predictions)
            mask_logit = masks_logits[0, best_mask_idx, :, :]
            
            binary_mask_256 = (mask_logit > 0.0).astype(np.float32)
            mask_target = postprocess_mask(
                binary_mask_256, h, w, ENCODER_SIZE, sam_transform, box=box
            )
            postprocess_times.append(time.perf_counter() - start)
            
            all_masks.append(mask_target)
        
        # Average times
        avg_decoder_time = sum(decoder_times) / len(decoder_times) if decoder_times else 0
        avg_postprocess_time = sum(postprocess_times) / len(postprocess_times) if postprocess_times else 0
        
        return np.array(all_masks).astype(bool), preprocess_time, encoder_time, avg_decoder_time, avg_postprocess_time
    
    def process_frame(self, processing_rgb):
        """
        Run YOLO + SAM detection on downscaled RGB image
        
        Args:
            processing_rgb: Downscaled RGB image (416×260)
        
        Returns:
            (detections, timing_tuple) where detections are at CAMERA resolution
        """
        # YOLO detection on processing resolution
        boxes, confidences, yolo_pre, yolo_inf, yolo_post = self.run_yolo_detection(processing_rgb)
        
        if len(boxes) == 0:
            return None, (yolo_pre, yolo_inf, yolo_post, 0, 0, 0, 0)
        
        # SAM segmentation on same processing resolution
        masks, sam_pre, sam_enc, sam_dec, sam_post = self.run_sam_segmentation(processing_rgb, boxes)
        
        # Scale boxes and masks to camera resolution
        cam_w, cam_h = self.camera.get_resolution()
        
        # Scale boxes
        full_res_boxes = boxes * np.array([self.scale_x, self.scale_y,
                                           self.scale_x, self.scale_y])
        
        # Scale masks
        full_res_masks = np.array([
            cv2.resize(
                m.astype(np.uint8),
                (cam_w, cam_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for m in masks
        ])
        
        # Return detections at camera resolution
        detections = {
            'boxes': full_res_boxes,
            'masks': full_res_masks,
            'confidences': confidences
        }
        
        return detections, (yolo_pre, yolo_inf, yolo_post, sam_pre, sam_enc, sam_dec, sam_post)
    
    def draw_detections_fast(self, image_rgb, detections):
        """Draw detections on RGB image - optimized version"""
        if detections is None:
            return image_rgb, 0
        
        start = time.perf_counter()
        try:
            result = create_output_image(
                image_rgb,
                detections['masks'],
                detections['boxes'],
                detections['confidences'],
                mask_color=MASK_COLOR,
                box_color=BOX_COLOR,
                alpha=MASK_OPACITY
            )
            viz_time = time.perf_counter() - start
            return result, viz_time
        except Exception as e:
            print(f"Drawing error: {e}")
            return image_rgb, 0
    
    def save_capture(self, full_res_image_rgb, full_res_detections, depth_map):
        """Save capture with all metadata"""
        if full_res_detections is None:
            print("No detections to save")
            return
        
        timestamp = int(time.time())
        
        # Get camera pose
        pose_info = self.camera.get_pose()
        
        # Prepare metadata
        metadata = {
            'timestamp': timestamp,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            'camera_pose': pose_info,
            'camera_calibration': self.camera.get_calibration(),
            'num_detections': len(full_res_detections['boxes']),
            'detection_scores': full_res_detections['confidences'].tolist(),
            'processing_resolution': {
                'width': self.processing_width,
                'height': self.processing_height
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Save annotated image (convert RGB to BGR for saving)
        if SAVE_FULL_RESOLUTION:
            annotated_rgb, _ = self.draw_detections_fast(full_res_image_rgb, full_res_detections)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            image_path = os.path.join(OUTPUT_DIR, f"image_{timestamp}.jpg")
            cv2.imwrite(image_path, annotated_bgr)
            print(f"✓ Saved image: {image_path}")
        
        # Save depth
        if SAVE_DEPTH_MAPS:
            depth_path = os.path.join(OUTPUT_DIR, f"depth_{timestamp}.npy")
            np.save(depth_path, depth_map)
            
            # Depth visualization
            depth_vis = depth_map.copy()
            depth_vis[depth_vis > 20] = 20
            depth_vis = (depth_vis / 20.0 * 255).astype(np.uint8)
            depth_vis_path = os.path.join(OUTPUT_DIR, f"depth_vis_{timestamp}.png")
            cv2.imwrite(depth_vis_path, depth_vis)
            print(f"✓ Saved depth: {depth_path}")
        
        # Save individual masks
        for i, mask in enumerate(full_res_detections['masks']):
            mask_img = (mask.astype(np.uint8) * 255)
            mask_path = os.path.join(OUTPUT_DIR, f"mask_{i}_{timestamp}.png")
            cv2.imwrite(mask_path, mask_img)
        
        # Save combined mask
        if len(full_res_detections['masks']) > 0:
            combined = np.maximum.reduce(full_res_detections['masks']).astype(np.uint8) * 255
            combined_path = os.path.join(OUTPUT_DIR, f"mask_combined_{timestamp}.png")
            cv2.imwrite(combined_path, combined)
            print(f"✓ Saved combined mask")
            print(f"  Heading: {pose_info['heading_deg']:.1f}°")
    
    def run(self):
        """Main processing loop - OPTIMIZED"""
        print("\n" + "="*70)
        print("Starting real-time processing...")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame + masks + depth")
        print("  'p' - Show camera pose")
        print("  't' - Toggle timing display")
        print("="*70 + "\n")
        
        self.running = True
        show_timing = True
        
        try:
            while self.running:
                # Grab frame - TIME THIS
                start = time.perf_counter()
                success, full_res_image_rgb, depth_map = self.camera.grab_frame()
                frame_grab_time = time.perf_counter() - start
                
                if not success:
                    print("Failed to grab frame")
                    break
                
                # Downscale ONCE to processing resolution - TIME THIS
                start = time.perf_counter()
                cv2.resize(full_res_image_rgb,
                          (self.processing_width, self.processing_height),
                          dst=self.processing_buffer,
                          interpolation=cv2.INTER_AREA)
                downscale_time = time.perf_counter() - start
                
                # Run detection on processing resolution
                self.current_detections, timing = self.process_frame(self.processing_buffer)
                yolo_pre, yolo_inf, yolo_post, sam_pre, sam_enc, sam_dec, sam_post = timing
                
                # Prepare display image
                if self.display_buffer is not None:
                    # Need to resize for display
                    display_image = cv2.resize(full_res_image_rgb,
                                              (self.display_width, self.display_height),
                                              interpolation=cv2.INTER_AREA)
                    
                    # Scale detections for display
                    if self.current_detections is not None:
                        cam_w, cam_h = self.camera.get_resolution()
                        display_scale_x = self.display_width / cam_w
                        display_scale_y = self.display_height / cam_h
                        
                        display_detections = {
                            'boxes': self.current_detections['boxes'] * np.array([
                                display_scale_x, display_scale_y,
                                display_scale_x, display_scale_y
                            ]),
                            'masks': np.array([
                                cv2.resize(
                                    m.astype(np.uint8),
                                    (self.display_width, self.display_height),
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                                for m in self.current_detections['masks']
                            ]),
                            'confidences': self.current_detections['confidences']
                        }
                    else:
                        display_detections = None
                else:
                    # Display at native resolution (no resize needed!)
                    display_image = full_res_image_rgb
                    display_detections = self.current_detections
                
                # Draw detections
                display_image, viz_time = self.draw_detections_fast(display_image, display_detections)
                
                # Record all timings
                self.inference_timer.add_measurement(
                    frame_grab_time, downscale_time,
                    yolo_pre, yolo_inf, yolo_post,
                    sam_pre, sam_enc, sam_dec, sam_post,
                    viz_time
                )
                
                # Get timing info
                fps = self.fps_monitor.get_fps()
                inference_hz = self.inference_timer.get_inference_hz()
                avg_times = self.inference_timer.get_averages()
                
                if show_timing:
                    # Draw comprehensive timing overlay
                    y = 25
                    spacing = 22
                    
                    # Overall metrics
                    cv2.putText(display_image, f"Display FPS: {fps:.1f}", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += spacing
                    cv2.putText(display_image, f"Inference: {inference_hz:.1f} Hz", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += spacing
                    
                    if self.current_detections is not None:
                        num_roads = len(self.current_detections['boxes'])
                        cv2.putText(display_image, f"Roads: {num_roads}", (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y += spacing
                    
                    y += 10  # Extra space
                    
                    # Frame operations
                    cv2.putText(display_image, "Frame Ops:", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y += spacing
                    cv2.putText(display_image, f"  Grab: {avg_times['frame_grab']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Downscale: {avg_times['downscale']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing + 5
                    
                    # YOLO breakdown
                    cv2.putText(display_image, "YOLO Pipeline:", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y += spacing
                    cv2.putText(display_image, f"  Preprocess: {avg_times['yolo_preprocess']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Inference: {avg_times['yolo_inference']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Postprocess: {avg_times['yolo_postprocess']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Total: {avg_times['yolo_total']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y += spacing + 5
                    
                    # SAM breakdown
                    cv2.putText(display_image, "SAM Pipeline:", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y += spacing
                    cv2.putText(display_image, f"  Preprocess: {avg_times['sam_preprocess']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Encoder: {avg_times['sam_encoder']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Decoder: {avg_times['sam_decoder']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Postprocess: {avg_times['sam_postprocess']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y += spacing
                    cv2.putText(display_image, f"  Total: {avg_times['sam_total']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y += spacing + 5
                    
                    # Visualization and total
                    cv2.putText(display_image, f"Visualization: {avg_times['visualization']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y += spacing + 5
                    cv2.putText(display_image, f"TOTAL: {avg_times['total']:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Convert RGB to BGR for display
                display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("ZED Road Segmentation", display_image_bgr)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_capture(full_res_image_rgb, self.current_detections, depth_map)
                elif key == ord('p'):
                    self.print_pose()
                elif key == ord('t'):
                    show_timing = not show_timing
                    print(f"Timing display: {'ON' if show_timing else 'OFF'}")
                
                self.fps_monitor.update()
                self.frame_counter += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def print_pose(self):
        """Print current camera pose"""
        print("\n" + "="*50)
        print("CURRENT CAMERA POSE:")
        
        pose_info = self.camera.get_pose()
        
        print(f"  Heading: {pose_info['heading_deg']:.1f}°")
        print(f"  Roll/Pitch/Yaw: {pose_info['rotation']['roll']:.2f}°, "
              f"{pose_info['rotation']['pitch']:.2f}°, {pose_info['rotation']['yaw']:.2f}°")
        
        print("="*50 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.running = False
        self.camera.close()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """Entry point"""
    try:
        system = RealtimeRoadSegmentation()
        system.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()