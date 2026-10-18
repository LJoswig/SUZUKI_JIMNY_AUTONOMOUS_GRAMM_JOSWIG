"""
Engine warmup utilities
Warm up TensorRT engines for optimal performance
"""

import numpy as np
import time
from .config import YOLO_INPUT_SIZE, ENCODER_SIZE


def warmup_engines(yolo_engine, encoder_engine, decoder_engine, num_iterations=5):
    """
    Warm up all TensorRT engines to ensure optimal performance.
    The first inference is always slower due to GPU initialization.
    
    Args:
        yolo_engine: YOLO TRT_Engine instance
        encoder_engine: SAM encoder TRT_Engine instance
        decoder_engine: SAM decoder TRT_Engine instance
        num_iterations: Number of warmup iterations
    """
    print("\n" + "="*70)
    print("WARMING UP ENGINES")
    print("="*70)
    
    # Warmup YOLO
    print(f"Warming up YOLO engine ({num_iterations} iterations)...")
    dummy_yolo_input = np.random.rand(1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).astype(np.float32)
    yolo_input_name = list(yolo_engine.inputs.keys())[0]
    
    yolo_times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = yolo_engine.inference({yolo_input_name: dummy_yolo_input})
        elapsed = time.perf_counter() - start
        yolo_times.append(elapsed * 1000)
    
    print(f"  Average: {np.mean(yolo_times):.2f} ms, Min: {np.min(yolo_times):.2f} ms")
    
    # Warmup Encoder
    print(f"Warming up SAM Encoder ({num_iterations} iterations)...")
    dummy_encoder_input = np.random.rand(1, 3, ENCODER_SIZE, ENCODER_SIZE).astype(np.float32)
    encoder_input_name = list(encoder_engine.inputs.keys())[0]
    
    encoder_times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        encoder_results = encoder_engine.inference({encoder_input_name: dummy_encoder_input})
        elapsed = time.perf_counter() - start
        encoder_times.append(elapsed * 1000)
    
    print(f"  Average: {np.mean(encoder_times):.2f} ms, Min: {np.min(encoder_times):.2f} ms")
    
    # Get dummy embedding for decoder warmup
    dummy_embedding = encoder_results[list(encoder_results.keys())[0]]
    
    # Warmup Decoder
    print(f"Warming up SAM Decoder ({num_iterations} iterations)...")
    dummy_point_coords = np.array([[[512.0, 512.0]]], dtype=np.float32)
    dummy_point_labels = np.array([[1]], dtype=np.float32)
    dummy_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    dummy_has_mask_input = np.zeros((1,), dtype=np.float32)
    
    decoder_input = {
        "image_embeddings": dummy_embedding,
        "point_coords": dummy_point_coords,
        "point_labels": dummy_point_labels,
        "mask_input": dummy_mask_input,
        "has_mask_input": dummy_has_mask_input,
    }
    
    decoder_times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = decoder_engine.inference(decoder_input)
        elapsed = time.perf_counter() - start
        decoder_times.append(elapsed * 1000)
    
    print(f"  Average: {np.mean(decoder_times):.2f} ms, Min: {np.min(decoder_times):.2f} ms")
    print("="*70 + "\n")