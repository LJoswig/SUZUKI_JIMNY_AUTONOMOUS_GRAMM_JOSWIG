import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d 
from skimage.morphology import remove_small_objects
import os

# --- CONFIGURATION ---
INPUT_MASK_PATH = '../road_data/mask_road_combined_1759914794.png' 
OUTPUT_DIR = 'centerline_output'
GAUSSIAN_SIGMA = 50  # Higher = smoother (good for autonomous nav)
DOWNSAMPLING_FACTOR = 75  # Lower = more waypoints (10 is good for testing)
HOOD_EXCLUSION_HEIGHT_PCT = 0.30  # Exclude bottom 30% (car hood area)

# --- HELPER FUNCTIONS ---

def save_image_and_plot(data, filename, title, is_distance_map=False):
    """Saves a 2D array/image and a Matplotlib plot of the result."""
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    data_to_save = data
    if is_distance_map and data.max() > 0:
        data_to_save = (data / data.max() * 255).astype(np.uint8)

    plt.figure(figsize=(10, 8))
    if data_to_save.ndim == 3:
        plt.imshow(cv2.cvtColor(data_to_save, cv2.COLOR_BGR2RGB))
    elif data_to_save.ndim == 2:
        plt.imshow(data_to_save, cmap='magma' if is_distance_map else 'gray')

    plt.title(title)
    plt.axis('off')
    plt.savefig(full_path.replace('.png', '.pdf'))
    plt.close()

    cv2.imwrite(full_path, data_to_save)
    print(f"Saved: {full_path}")


def refine_mask(mask_image):
    """Applies morphological closing and safely removes small objects."""
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    
    boolean_mask = closed_mask > 0
    cleaned_mask = remove_small_objects(boolean_mask, min_size=100) 
    
    refined_mask_img = (cleaned_mask * 255).astype(np.uint8)
    return refined_mask_img


def extract_centerline_midpoint(refined_mask):
    """Finds left/right boundaries and calculates centerlines as the midpoint average."""
    
    if np.all(refined_mask == 0):
        print("Warning: Refined mask is entirely empty (black).")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("Warning: cv2.findContours returned no contours.")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    main_contour = max(contours, key=cv2.contourArea)
    points = main_contour.squeeze()
    
    if points.ndim != 2 or points.shape[0] < 2:
        print("Warning: Main contour is too short or malformed.")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    height = refined_mask.shape[0]
    min_x_at_y = np.full(height, refined_mask.shape[1]) 
    max_x_at_y = np.full(height, -1)                  

    for X, Y in points:
        min_x_at_y[Y] = min(min_x_at_y[Y], X)
        max_x_at_y[Y] = max(max_x_at_y[Y], X)

    valid_rows = np.where(min_x_at_y <= max_x_at_y)[0]
    
    if len(valid_rows) == 0:
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    midpoint_X = (min_x_at_y[valid_rows] + max_x_at_y[valid_rows]) / 2.0
    midpoint_Y = valid_rows.astype(np.float64)
    
    midpoint_coords_full = np.vstack((midpoint_X, midpoint_Y)).T 
    
    # Visualization
    midpoint_coords_vis = midpoint_coords_full[::DOWNSAMPLING_FACTOR]
    centerline_image = np.zeros_like(refined_mask, dtype=np.uint8)
    for X, Y in midpoint_coords_vis:
        cv2.circle(centerline_image, (int(X), int(Y)), 1, 255, -1)
        
    print(f"Midpoint Extraction: Extracted {midpoint_coords_full.shape[0]} full points. {midpoint_coords_vis.shape[0]} used for visualization.")
    return midpoint_coords_full, centerline_image


def smooth_centerline_gaussian(centerline_coords_full, mask_shape):
    """Applies a 1D Gaussian filter for simple, non-destructive smoothing."""
    if centerline_coords_full.shape[0] < 5:
        return None, None
    
    X_raw = centerline_coords_full[:, 0]
    Y_raw = centerline_coords_full[:, 1]
    
    X_smooth = gaussian_filter1d(X_raw, sigma=GAUSSIAN_SIGMA)
    Y_smooth = gaussian_filter1d(Y_raw, sigma=GAUSSIAN_SIGMA)
    
    smooth_centerline_coords = np.vstack((X_smooth, Y_smooth)).T
    
    final_image = np.zeros(mask_shape, dtype=np.uint8)
    return smooth_centerline_coords, final_image 


def overlay_line_on_mask(mask_image, coords, color=(255, 0, 0), thickness=2, style='line'):
    """Creates an RGB image with the mask and overlays the line/points in the specified color."""
    overlay_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    
    if style == 'points':
        for X, Y in coords:
            cv2.circle(overlay_image, (int(X), int(Y)), 3, color, -1)
    else:
        for i in range(1, len(coords)):
            pt1 = (int(coords[i-1][0]), int(coords[i-1][1]))
            pt2 = (int(coords[i][0]), int(coords[i][1]))
            cv2.line(overlay_image, pt1, pt2, color, thickness)
            
    return overlay_image


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("CENTERLINE EXTRACTION FROM ROAD MASK")
    print("="*70)
    print(f"Input: {INPUT_MASK_PATH}")
    print(f"Gaussian Sigma: {GAUSSIAN_SIGMA}")
    print(f"Downsampling: Every {DOWNSAMPLING_FACTOR}th point")
    print(f"Hood Exclusion: Bottom {HOOD_EXCLUSION_HEIGHT_PCT*100}%")
    print("="*70 + "\n")
    
    # Load and prepare the input mask 
    if os.path.exists(INPUT_MASK_PATH):
        mask_raw = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_COLOR)
        
        if mask_raw is None:
            print(f"Error: Could not load image from {INPUT_MASK_PATH}")
            exit()
             
        mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    else:
        print(f"Error: Input file {INPUT_MASK_PATH} not found.")
        exit()

    # Hood exclusion masking
    height, width = mask_binary.shape
    exclusion_start_row = int(height * (1.0 - HOOD_EXCLUSION_HEIGHT_PCT))
    mask_binary[exclusion_start_row:, :] = 0
    print(f"Applied hood exclusion: ignoring pixels below row {exclusion_start_row}")

    # STEP 1: Pre-processing
    print("\n[STEP 1] Refining mask...")
    refined_mask = refine_mask(mask_binary)
    save_image_and_plot(refined_mask, '01_refined_mask.png', 'Refined Road Mask')

    # STEP 2: Extraction
    print("\n[STEP 2] Extracting centerline...")
    centerline_coords_full, centerline_image_raw = extract_centerline_midpoint(refined_mask)
    save_image_and_plot(centerline_image_raw, '02_raw_midpoints.png', 'Raw Midpoints')

    if centerline_coords_full is not None:
        
        # STEP 3: Smoothing
        print("\n[STEP 3] Applying Gaussian smoothing...")
        smooth_coords, _ = smooth_centerline_gaussian(centerline_coords_full, mask_binary.shape)

        if smooth_coords is not None:
            
            # Filter smoothed points in excluded area
            valid_smooth_indices = smooth_coords[:, 1] < exclusion_start_row
            smooth_coords_filtered = smooth_coords[valid_smooth_indices]

            if len(smooth_coords_filtered) < 5:
                print("\n❌ Error: All smoothed coordinates were filtered out")
                exit()
            
            # Downsample for final waypoints
            centerline_coords_smooth_downsampled = smooth_coords_filtered[::DOWNSAMPLING_FACTOR]
            
            # CRITICAL: Save in correct format (x_col, y_row) for GPS converter
            df_coords_smooth = pd.DataFrame(centerline_coords_smooth_downsampled, columns=['x_col', 'y_row'])
            coords_path_smooth = os.path.join(OUTPUT_DIR, 'waypoints_for_gps_conversion.csv')
            df_coords_smooth.to_csv(coords_path_smooth, index=False)
            print(f"\n✅ Saved {len(centerline_coords_smooth_downsampled)} waypoints to: {coords_path_smooth}")
            print(f"   → Use this file with waypoint_to_gps.py")

            # Also save full smooth line for visualization
            df_coords_full_smooth = pd.DataFrame(smooth_coords_filtered, columns=['x_col', 'y_row'])
            coords_path_full_smooth = os.path.join(OUTPUT_DIR, 'centerline_full_smooth.csv')
            df_coords_full_smooth.to_csv(coords_path_full_smooth, index=False)
            
            # Visualizations
            print("\n[STEP 4] Creating visualizations...")
            GREEN = (0, 255, 0)
            overlay_waypoints = overlay_line_on_mask(refined_mask, centerline_coords_smooth_downsampled, color=GREEN, style='points')
            save_image_and_plot(overlay_waypoints, '03_waypoints_overlay.png', f'Waypoints (Green, n={len(centerline_coords_smooth_downsampled)})')
            
            BLUE = (255, 0, 0)
            overlay_smooth_line = overlay_line_on_mask(refined_mask, smooth_coords_filtered, color=BLUE, style='line')
            save_image_and_plot(overlay_smooth_line, '04_smooth_centerline_overlay.png', 'Smooth Centerline (Blue)')
            
            print("\n" + "="*70)
            print("✅ CENTERLINE EXTRACTION COMPLETE")
            print("="*70)
            print(f"\nNext step:")
            print(f"  python3 waypoint_to_gps.py {coords_path_smooth} --snapshot --output mission_waypoints.json")
            print("="*70 + "\n")
        else:
            print("\n❌ Error: Gaussian smoothing failed")
    else:
        print("\n❌ Error: Contour extraction failed")