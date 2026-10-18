# Revert imports for the new approach
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from skimage.morphology import remove_small_objects
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

# --- CONFIGURATION ---
INPUT_MASK_PATH = '0_input/road_segmented_mask.jpg' 
OUTPUT_DIR = 'centerline_output'

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


# 1. Pre-processing: Mask Refinement (Ultra-Aggressive Filtering)
def refine_mask(mask_image):
    """Applies morphological closing and aggressively removes small objects."""
    # Morphological closing to fill small gaps
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    
    boolean_mask = closed_mask > 0
    
    # FIX: Increased min_size to 10000. This should ensure only the main road component remains.
    cleaned_mask = remove_small_objects(boolean_mask, min_size=10000) 
    
    refined_mask_img = (cleaned_mask * 255).astype(np.uint8)
    return refined_mask_img

# 2. Contour-based Extraction: Midpoint Averaging
def extract_centerline_midpoint(refined_mask):
    """Finds left/right boundaries and calculates centerlines as the midpoint average."""
    
    # Find the main contour (only external contours)
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("Midpoint Extraction: No main contour found.")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    # Assume the largest contour is the road
    main_contour = max(contours, key=cv2.contourArea)
    
    # Convert contour points to (X, Y) list
    points = main_contour.squeeze()
    if points.ndim != 2 or points.shape[0] < 2:
        print("Midpoint Extraction: Contour has too few points.")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    # Prepare array to hold the boundary X coordinates for each Y row
    # We use the full height of the image
    height = refined_mask.shape[0]
    min_x_at_y = np.full(height, refined_mask.shape[1]) # Initialize to max width
    max_x_at_y = np.full(height, -1)                  # Initialize to -1

    # Map all contour points to their Y row, tracking the min/max X (left/right)
    for X, Y in points:
        min_x_at_y[Y] = min(min_x_at_y[Y], X)
        max_x_at_y[Y] = max(max_x_at_y[Y], X)

    # Find rows that actually have a mask (i.e., min_x < max_x)
    valid_rows = np.where(min_x_at_y <= max_x_at_y)[0]
    
    if len(valid_rows) == 0:
        print("Midpoint Extraction: No valid centerline rows found.")
        return None, np.zeros_like(refined_mask, dtype=np.uint8)

    # Calculate midpoint X for each valid Y row
    midpoint_X = (min_x_at_y[valid_rows] + max_x_at_y[valid_rows]) / 2.0
    midpoint_Y = valid_rows.astype(np.float64)
    
    # Combine coordinates: (row, col) format for RANSAC
    midpoint_coords = np.vstack((midpoint_Y, midpoint_X)).T 

    # Visualization of midpoints (optional, useful for debugging)
    centerline_image = np.zeros_like(refined_mask, dtype=np.uint8)
    for Y, X in midpoint_coords:
        cv2.circle(centerline_image, (int(X), int(Y)), 1, 255, -1)
        
    print(f"Midpoint Extraction: Extracted {midpoint_coords.shape[0]} raw points.")
    return midpoint_coords, centerline_image


# 3. Post-processing: RANSAC and Spline Fitting (remains the same but uses dense midpoint input)
def smooth_centerline_ransac_spline(centerline_coords, mask_shape):
    """Applies RANSAC for outlier rejection, then fits a smooth cubic spline."""
    if centerline_coords.shape[0] < 5:
        print("Not enough centerline points for RANSAC/Spline.")
        return None, None

    # NOTE: centerline_coords is (Y_row, X_col) from midpoint extraction
    y_raw = centerline_coords[:, 0].reshape(-1, 1) 
    X_raw = centerline_coords[:, 1]               
    
    poly_features = PolynomialFeatures(degree=3) 
    y_poly = poly_features.fit_transform(y_raw)

    # RANSAC will now clean up the midpoints, removing spurious points from the top
    ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, 
                             residual_threshold=2.0, max_trials=100) 
    
    with np.errstate(invalid='ignore'):
        ransac.fit(y_poly, X_raw)
    
    inlier_mask = ransac.inlier_mask_ 
    inlier_y = y_raw[inlier_mask].flatten()
    inlier_X = X_raw[inlier_mask]

    # ... [rest of the spline logic remains identical] ...
    
    # Sort and remove duplicates for CubicSpline robustness
    sort_indices = np.argsort(inlier_y)
    inlier_y_sorted = inlier_y[sort_indices]
    inlier_X_sorted = inlier_X[sort_indices]
    
    unique_y_sorted, indices, counts = np.unique(inlier_y_sorted, return_index=True, return_counts=True)
    
    unique_X_list = []
    current_index = 0
    for count in counts:
        mean_X = np.mean(inlier_X_sorted[current_index:current_index + count])
        unique_X_list.append(mean_X)
        current_index += count
        
    unique_X_sorted = np.array(unique_X_list)
    
    if len(unique_y_sorted) < 4: 
        print(f"Not enough unique inlier points ({len(unique_y_sorted)}) after cleaning for CubicSpline.")
        return None, None

    # Cubic Spline
    cs = CubicSpline(unique_y_sorted, unique_X_sorted, bc_type='natural')
    
    y_new = np.linspace(unique_y_sorted.min(), unique_y_sorted.max(), 50)
    X_new = cs(y_new)
    
    final_image = np.zeros(mask_shape, dtype=np.uint8)
    for X, y in zip(X_new, y_new):
        cv2.circle(final_image, (int(X), int(y)), 3, 255, -1)

    # Output as a list of (x_col, y_row) coordinates
    smooth_centerline_coords = np.vstack((X_new, y_new)).T 

    return smooth_centerline_coords, final_image

# 4. Final Image Overlay
def overlay_centerline_on_mask(mask_image, smooth_centerline_coords):
    """Creates an RGB image with the mask in white and the centerline in blue."""
    overlay_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    
    BLUE = (255, 0, 0) 

    for i in range(1, len(smooth_centerline_coords)):
        pt1 = (int(smooth_centerline_coords[i-1][0]), int(smooth_centerline_coords[i-1][1]))
        pt2 = (int(smooth_centerline_coords[i][0]), int(smooth_centerline_coords[i][1]))
        cv2.line(overlay_image, pt1, pt2, BLUE, 2)
        
    return overlay_image

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_MASK_PATH):
        mask_raw = cv2.imread('road_segmented_mask.jpg', cv2.IMREAD_COLOR)
        if mask_raw is None:
             print("Error: road_segmented_mask.jpg not found or could not be loaded.")
             exit()
        mask_binary = cv2.inRange(mask_raw, (0, 200, 0), (50, 255, 50)) 
    else:
        mask_raw = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        _, mask_binary = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)


    # --- STEP 1: PRE-PROCESSING (Ultra-Aggressive Filter) ---
    refined_mask = refine_mask(mask_binary)
    save_image_and_plot(refined_mask, '01_refined_mask_ultra_aggressive.png', 'Step 1: Refined Road Mask (Ultra-Aggressive Filter)')
    
    # --- STEP 2: EXTRACTION (Contour Midpoints) ---
    centerline_coords, centerline_image = extract_centerline_midpoint(refined_mask)
    save_image_and_plot(centerline_image, '02_extracted_midpoints.png', 'Step 2: Raw Centerline Points via Contour Midpoints')

    # --- STEP 3 & 4: POST-PROCESSING & OVERLAY (remain unchanged) ---
    if centerline_coords is not None:
        smooth_coords, final_image = smooth_centerline_ransac_spline(centerline_coords, mask_binary.shape)
        
        if smooth_coords is not None:
            save_image_and_plot(final_image, '03_final_smooth_centerline_midpoint.png', 'Step 3: Final Centerline (Midpoint + RANSAC + Spline)')
            
            # Save coordinates and overlay image...
            df_coords = pd.DataFrame(smooth_coords, columns=['x_col', 'y_row'])
            coords_path = os.path.join(OUTPUT_DIR, '03_final_centerline_coords_midpoint.csv')
            df_coords.to_csv(coords_path, index=False)

            overlay_image = overlay_centerline_on_mask(refined_mask, smooth_coords)
            save_image_and_plot(overlay_image, '04_mask_with_centerline_overlay_midpoint.png', 'Step 4: Final Centerline Overlay (Contour Midpoint Fixed)')
            
            print("\n✅ Pipeline complete. Switching to Contour Midpoints should provide a perfectly centered line.")
        else:
            print("\n❌ Pipeline stopped: RANSAC/Spline failed on midpoint data.")
    else:
        print("\n❌ Pipeline stopped: Contour extraction failed.")