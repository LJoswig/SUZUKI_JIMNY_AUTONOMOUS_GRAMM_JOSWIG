import cv2
import numpy as np


def extract_centerline(mask, min_component_area=100):
    """Return spatially smoothed row-wise center points and continuity confidence."""
    if mask is None:
        return None, 0.0

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Centerline extraction expects a 2D road mask")

    binary_mask = (mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return None, 0.0

    kernel = np.ones((7, 7), dtype=np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    cleaned_mask = np.zeros_like(binary_mask)
    for component_id in range(1, component_count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned_mask[labels == component_id] = 1

    rows, columns = np.nonzero(cleaned_mask)
    if rows.size == 0:
        return None, 0.0

    height, width = cleaned_mask.shape
    left_edges = np.full(height, width, dtype=np.int32)
    right_edges = np.full(height, -1, dtype=np.int32)
    np.minimum.at(left_edges, rows, columns)
    np.maximum.at(right_edges, rows, columns)

    valid_rows = np.flatnonzero(right_edges >= left_edges)
    row_midpoints = (left_edges[valid_rows] + right_edges[valid_rows]) / 2.0
    row_range = np.arange(valid_rows[0], valid_rows[-1] + 1)
    interpolated_midpoints = np.interp(row_range, valid_rows, row_midpoints)
    smoothed_midpoints = cv2.GaussianBlur(
        interpolated_midpoints.astype(np.float32).reshape(-1, 1),
        (1, 5),
        sigmaX=0.0,
        sigmaY=1.0,
    ).reshape(-1)
    points = np.column_stack((
        smoothed_midpoints[valid_rows - valid_rows[0]],
        valid_rows.astype(np.float32),
    )).astype(np.float32)

    covered_rows = valid_rows[-1] - valid_rows[0] + 1
    confidence = len(valid_rows) / covered_rows
    return points, float(confidence)