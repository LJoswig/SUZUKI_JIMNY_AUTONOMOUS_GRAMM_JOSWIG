import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d 
from skimage.morphology import remove_small_objects
import os

# --- CONFIGURATION ---
INPUT_MASK_PATH = './road_masks/mask_road_idx1_1759330846.png' 
OUTPUT_DIR = 'centerline_output'
GAUSSIAN_SIGMA = 50 # Set to 50 for highly smooth centerline, as requested
DOWNSAMPLING_FACTOR = 150 # Use every Nth point to make waypoints sparser
# NEW CONFIG: Exclude the bottom percentage of the image (where the car hood is)
# Increased to 0.20 (20%) to ensure the line is clearly off the car hood
HOOD_EXCLUSION_HEIGHT_PCT = 0.30 # E.g., 0.20 means the bottom 20% of rows are ignored

# --- HELPER FUNCTIONS ---

def save_image_and_plot(data, filename, title, is_distance_map=False):
    """Saves a 2D array/image and a Matplotlib plot of the result."""
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    data_to_save = data
    if is_distance_map and data.max() > 0:
        data_to_save = (data / data.max() * 255).astype(np.uint8)

    plt.figure(figsize=(10, 8))
    # Matplotlib will now use the default backend for display
    if data_to_save.ndim == 3:
        # Matplotlib expects RGB, OpenCV is BGR
        plt.imshow(cv2.cvtColor(data_to_save, cv2.COLOR_BGR2RGB))
    elif data_to_save.ndim == 2:
        plt.imshow(data_to_save, cmap='magma' if is_distance_map else 'gray')

    plt.title(title)
    plt.axis('off')
    plt.savefig(full_path.replace('.png', '.pdf'))
    plt.close()

    cv2.imwrite(full_path, data_to_save)
    print(f"Saved: {full_path}")


# 1. Pre-processing: Mask Refinement (Less Aggressive Filtering)
def refine_mask(mask_image):
    """Applies morphological closing and safely removes small objects."""
    # Closing operation to fill small gaps
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    
    boolean_mask = closed_mask > 0
    # Restored min_size=100 for safer noise removal.
    cleaned_mask = remove_small_objects(boolean_mask, min_size=100) 
    
    refined_mask_img = (cleaned_mask * 255).astype(np.uint8)
    return refined_mask_img


# 2. Contour-based Extraction: Midpoint Averaging (Generates the full set of raw points)
def extract_centerline_midpoint(refined_mask):
    """Finds left/right boundaries and calculates centerlines as the midpoint average."""
    
    # Check if the mask is completely black before contour finding
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
    # NOTE: This logic relies on the fact that the mask has been zeroed out below the hood line.
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
    
    # Coordinates format: (X_col, Y_row) - the full set of raw midpoints
    midpoint_coords_full = np.vstack((midpoint_X, midpoint_Y)).T 
    
    # Visualization of raw midpoints (downsampled just for display)
    midpoint_coords_vis = midpoint_coords_full[::DOWNSAMPLING_FACTOR]
    centerline_image = np.zeros_like(refined_mask, dtype=np.uint8)
    for X, Y in midpoint_coords_vis:
        cv2.circle(centerline_image, (int(X), int(Y)), 1, 255, -1)
        
    print(f"Midpoint Extraction: Extracted {midpoint_coords_full.shape[0]} full points. {midpoint_coords_vis.shape[0]} used for visualization.")
    # Return the full set of coordinates to ensure the Gaussian smoothing has enough data points
    return midpoint_coords_full, centerline_image


# 3. Post-processing: Gaussian Smoothing (For smooth trajectory path)
def smooth_centerline_gaussian(centerline_coords_full, mask_shape):
    """Applies a 1D Gaussian filter for simple, non-destructive smoothing."""
    if centerline_coords_full.shape[0] < 5:
        return None, None
    
    # Coordinates are (X_col, Y_row)
    X_raw = centerline_coords_full[:, 0]
    Y_raw = centerline_coords_full[:, 1]
    
    # Apply 1D Gaussian filter to both X and Y arrays independently
    X_smooth = gaussian_filter1d(X_raw, sigma=GAUSSIAN_SIGMA)
    Y_smooth = gaussian_filter1d(Y_raw, sigma=GAUSSIAN_SIGMA)
    
    smooth_centerline_coords = np.vstack((X_smooth, Y_smooth)).T
    
    final_image = np.zeros(mask_shape, dtype=np.uint8)
    return smooth_centerline_coords, final_image 


# 4. Final Image Overlay Helpers

def overlay_line_on_mask(mask_image, coords, color=(255, 0, 0), thickness=2, style='line'):
    """Creates an RGB image with the mask and overlays the line/points in the specified color."""
    overlay_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    
    if style == 'points':
        # Overlay points (used for waypoints)
        for X, Y in coords:
            cv2.circle(overlay_image, (int(X), int(Y)), 3, color, -1)
    else: # style == 'line'
        # Overlay line segments (used for smoothed path)
        for i in range(1, len(coords)):
            pt1 = (int(coords[i-1][0]), int(coords[i-1][1]))
            pt2 = (int(coords[i][0]), int(coords[i][1]))
            cv2.line(overlay_image, pt1, pt2, color, thickness)
            
    return overlay_image


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare the input mask 
    if os.path.exists(INPUT_MASK_PATH):
        # Load in COLOR mode
        mask_raw = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_COLOR)
        
        if mask_raw is None:
             print(f"Error: Could not load image from {INPUT_MASK_PATH}. Please ensure the file exists and is readable.")
             exit()
             
        # Convert to binary
        mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        # Threshold: anything with a grayscale value > 10 is considered the mask (white or green area)
        _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    else:
        print(f"Error: Input file {INPUT_MASK_PATH} not found.")
        exit()

    
    # --- HOOD EXCLUSION MASKING ---
    height, width = mask_binary.shape
    # Calculate the starting row index for the exclusion zone
    exclusion_start_row = int(height * (1.0 - HOOD_EXCLUSION_HEIGHT_PCT))
    
    # Set the area from the exclusion_start_row down to the bottom to black (0)
    mask_binary[exclusion_start_row:, :] = 0
    print(f"Applied hood exclusion mask: ignoring all waypoints below row {exclusion_start_row} ({HOOD_EXCLUSION_HEIGHT_PCT*100}% from bottom).")
    

    # --- STEP 1: PRE-PROCESSING (Noise Filter - min_size=100) ---
    refined_mask = refine_mask(mask_binary)
    save_image_and_plot(refined_mask, '01_refined_mask_final.png', 'Step 1: Refined Road Mask (Noise Filtered)')

    # --- STEP 2: EXTRACTION (Contour Midpoints - FULL RAW Waypoints) ---
    centerline_coords_full, centerline_image_raw = extract_centerline_midpoint(refined_mask)
    save_image_and_plot(centerline_image_raw, '02_extracted_midpoints_raw_downsampled_vis.png', 'Step 2: Raw Waypoints (Midpoints - Downsampled Visualization)')

    if centerline_coords_full is not None:
        
        # --- STEP 3: POST-PROCESSING (Gaussian Smoothing - To define the smooth trajectory) ---
        smooth_coords, final_image_smooth_placeholder = smooth_centerline_gaussian(centerline_coords_full, mask_binary.shape)

        if smooth_coords is not None:
            
            # --- CRITICAL: FILTER OUT SMOOTHED POINTS THAT FELL INTO THE EXCLUDED AREA ---
            # Even though the raw points respected the exclusion, Gaussian smoothing can interpolate 
            # points back down into the masked-out area near the exclusion boundary.
            valid_smooth_indices = smooth_coords[:, 1] < exclusion_start_row
            smooth_coords_filtered = smooth_coords[valid_smooth_indices]

            if len(smooth_coords_filtered) < 5:
                print("\n❌ Pipeline stopped: Smoothed coordinates were completely filtered out by hood exclusion.")
                exit()
            
            # Downsample the SMOOTHED and FILTERED coordinates for the final waypoints
            centerline_coords_smooth_downsampled = smooth_coords_filtered[::DOWNSAMPLING_FACTOR]
            
            # Save the smoothed and downsampled coordinates as the final waypoints
            df_coords_smooth = pd.DataFrame(centerline_coords_smooth_downsampled, columns=['x_col', 'y_row'])
            coords_path_smooth = os.path.join(OUTPUT_DIR, '02_waypoint_smooth_downsampled.csv') 
            df_coords_smooth.to_csv(coords_path_smooth, index=False)
            print(f"Saved smoothed and downsampled waypoints to {coords_path_smooth}")

            # Save the full smoothed coordinates (high resolution for continuous path display)
            df_coords_full_smooth = pd.DataFrame(smooth_coords_filtered, columns=['x_col', 'y_row']) # Use the FILTERED coords
            coords_path_full_smooth = os.path.join(OUTPUT_DIR, '03_final_smooth_centerline_full.csv')
            df_coords_full_smooth.to_csv(coords_path_full_smooth, index=False)
            print(f"Saved full smoothed centerline coordinates (high resolution) to {coords_path_full_smooth}")
            
            # --- STEP 4: OVERLAY VISUALIZATION ---
            
            # 4a: Overlay SMOOTHED Downsampled Waypoints (Green Dots)
            GREEN = (0, 255, 0)
            overlay_waypoints = overlay_line_on_mask(refined_mask, centerline_coords_smooth_downsampled, color=GREEN, style='points')
            save_image_and_plot(overlay_waypoints, '04a_mask_with_smooth_waypoints_downsampled_overlay.png', f'Step 4a: Smoothed Waypoints Overlay (Green Dots - Downsampled by {DOWNSAMPLING_FACTOR}, Sigma={GAUSSIAN_SIGMA})')
            
            # 4b: Overlay SMOOTHED Line (Blue Line) - This is the continuous line path
            BLUE = (255, 0, 0)
            overlay_smooth_line = overlay_line_on_mask(refined_mask, smooth_coords_filtered, color=BLUE, style='line') # Use the FILTERED coords
            save_image_and_plot(overlay_smooth_line, '04b_mask_with_smooth_centerline_overlay_smoother.png', f'Step 4b: Smoothed Line Overlay (Blue Line - Sigma={GAUSSIAN_SIGMA})')
            
            print("\n✅ Pipeline complete. Waypoints are now generated from the smoothed line.")
        else:
            print("\n❌ Pipeline stopped: Gaussian smoothing failed.")
    else:
        print("\n❌ Pipeline stopped: Contour extraction failed.")
