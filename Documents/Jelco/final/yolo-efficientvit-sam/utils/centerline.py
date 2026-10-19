import cv2
import numpy as np
import time


class TemporalCenterlineSmoother:
    """Apply an EMA to centerline x-coordinates matched by image row."""

    def __init__(self, alpha=0.35):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Temporal smoothing alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.previous_points = None

    def reset(self):
        self.previous_points = None

    def update(self, points):
        if points is None or len(points) == 0:
            self.reset()
            return None

        current_points = np.asarray(points, dtype=np.float32).copy()
        if self.previous_points is not None:
            previous_x = np.interp(
                current_points[:, 1],
                self.previous_points[:, 1],
                self.previous_points[:, 0],
                left=np.nan,
                right=np.nan,
            )
            overlapping_rows = np.isfinite(previous_x)
            current_points[overlapping_rows, 0] = (
                self.alpha * current_points[overlapping_rows, 0]
                + (1.0 - self.alpha) * previous_x[overlapping_rows]
            )

        self.previous_points = current_points.copy()
        return current_points


class TemporalCenterlineKalmanFilter:
    """Track lateral position and velocity independently at each image row."""

    def __init__(
        self,
        process_noise=2500.0,
        measurement_noise=25.0,
        max_gap_seconds=0.5,
    ):
        if process_noise < 0.0:
            raise ValueError("Kalman process noise must be non-negative")
        if measurement_noise <= 0.0:
            raise ValueError("Kalman measurement noise must be positive")
        if max_gap_seconds <= 0.0:
            raise ValueError("Kalman max gap must be positive")

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_gap_seconds = float(max_gap_seconds)
        self.initialized = np.zeros(0, dtype=bool)
        self.position = np.zeros(0, dtype=np.float64)
        self.velocity = np.zeros(0, dtype=np.float64)
        self.covariance_position = np.zeros(0, dtype=np.float64)
        self.covariance_cross = np.zeros(0, dtype=np.float64)
        self.covariance_velocity = np.zeros(0, dtype=np.float64)
        self.last_update_times = np.zeros(0, dtype=np.float64)

    def reset(self):
        self.initialized.fill(False)

    def _ensure_capacity(self, row_count):
        current_size = len(self.initialized)
        if row_count <= current_size:
            return

        extension = row_count - current_size
        self.initialized = np.pad(self.initialized, (0, extension))
        self.position = np.pad(self.position, (0, extension))
        self.velocity = np.pad(self.velocity, (0, extension))
        self.covariance_position = np.pad(self.covariance_position, (0, extension))
        self.covariance_cross = np.pad(self.covariance_cross, (0, extension))
        self.covariance_velocity = np.pad(self.covariance_velocity, (0, extension))
        self.last_update_times = np.pad(self.last_update_times, (0, extension))

    def _initialize_rows(self, rows, measured_x, measurement_noise, timestamp):
        self.position[rows] = measured_x
        self.velocity[rows] = 0.0
        self.covariance_position[rows] = measurement_noise
        self.covariance_cross[rows] = 0.0
        self.covariance_velocity[rows] = 2500.0
        self.last_update_times[rows] = timestamp
        self.initialized[rows] = True

    def update(self, points, confidence=1.0):
        if points is None or len(points) == 0:
            self.reset()
            return None

        current_points = np.asarray(points, dtype=np.float32).copy()
        rows = np.rint(current_points[:, 1]).astype(np.intp)
        if np.any(rows < 0):
            raise ValueError("Centerline rows must be non-negative")
        self._ensure_capacity(int(rows.max()) + 1)

        confidence = float(np.clip(confidence, 0.05, 1.0))
        measurement_noise = self.measurement_noise / confidence
        timestamp = time.perf_counter()

        measured_x = current_points[:, 0].astype(np.float64)
        is_new = ~self.initialized[rows]
        new_rows = rows[is_new]
        if new_rows.size:
            self._initialize_rows(
                new_rows, measured_x[is_new], measurement_noise, timestamp
            )

        existing_indices = np.flatnonzero(~is_new)
        existing_rows = rows[existing_indices]
        delta_time = timestamp - self.last_update_times[existing_rows]
        expired = delta_time > self.max_gap_seconds
        if np.any(expired):
            expired_rows = existing_rows[expired]
            expired_indices = existing_indices[expired]
            self._initialize_rows(
                expired_rows,
                measured_x[expired_indices],
                measurement_noise,
                timestamp,
            )

        active_indices = existing_indices[~expired]
        active_rows = rows[active_indices]
        if active_rows.size:
            delta_time = np.maximum(delta_time[~expired], 1e-3)
            delta_time_2 = delta_time * delta_time
            delta_time_3 = delta_time_2 * delta_time
            delta_time_4 = delta_time_2 * delta_time_2

            predicted_position = (
                self.position[active_rows]
                + delta_time * self.velocity[active_rows]
            )
            predicted_covariance_position = (
                self.covariance_position[active_rows]
                + 2.0 * delta_time * self.covariance_cross[active_rows]
                + delta_time_2 * self.covariance_velocity[active_rows]
                + self.process_noise * delta_time_4 / 4.0
            )
            predicted_covariance_cross = (
                self.covariance_cross[active_rows]
                + delta_time * self.covariance_velocity[active_rows]
                + self.process_noise * delta_time_3 / 2.0
            )
            predicted_covariance_velocity = (
                self.covariance_velocity[active_rows]
                + self.process_noise * delta_time_2
            )

            innovation = measured_x[active_indices] - predicted_position
            innovation_variance = predicted_covariance_position + measurement_noise
            position_gain = predicted_covariance_position / innovation_variance
            velocity_gain = predicted_covariance_cross / innovation_variance

            self.position[active_rows] = predicted_position + position_gain * innovation
            self.velocity[active_rows] = self.velocity[active_rows] + velocity_gain * innovation
            self.covariance_position[active_rows] = (
                (1.0 - position_gain) * predicted_covariance_position
            )
            self.covariance_cross[active_rows] = (
                (1.0 - position_gain) * predicted_covariance_cross
            )
            self.covariance_velocity[active_rows] = (
                predicted_covariance_velocity
                - velocity_gain * predicted_covariance_cross
            )
            self.last_update_times[active_rows] = timestamp

        current_points[:, 0] = self.position[rows].astype(np.float32)

        return current_points


def extract_centerline(
    mask, min_component_area=100, use_curve_fit=False, fit_degree=2
):
    """Return center points and continuity confidence, optionally fitting a curve."""
    if fit_degree not in (1, 2, 3):
        raise ValueError("Centerline polynomial degree must be 1, 2, or 3")

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
    covered_rows = valid_rows[-1] - valid_rows[0] + 1
    confidence = len(valid_rows) / covered_rows

    if use_curve_fit and len(valid_rows) >= fit_degree + 1:
        row_positions = valid_rows.astype(np.float64)
        row_scale = max(np.ptp(row_positions) / 2.0, 1.0)
        normalized_rows = (row_positions - row_positions.mean()) / row_scale
        inliers = np.ones(len(valid_rows), dtype=bool)

        for _ in range(3):
            coefficients = np.polyfit(
                normalized_rows[inliers], row_midpoints[inliers], deg=fit_degree
            )
            residuals = row_midpoints - np.polyval(coefficients, normalized_rows)
            residual_center = np.median(residuals[inliers])
            residual_mad = np.median(
                np.abs(residuals[inliers] - residual_center)
            )
            tolerance = max(3.0, 3.0 * 1.4826 * residual_mad)
            refined_inliers = np.abs(residuals - residual_center) <= tolerance
            if refined_inliers.sum() < 5 or np.array_equal(refined_inliers, inliers):
                break
            inliers = refined_inliers

        coefficients = np.polyfit(
            normalized_rows[inliers], row_midpoints[inliers], deg=fit_degree
        )
        center_x = np.polyval(coefficients, normalized_rows)
        confidence *= float(np.mean(inliers))
    else:
        row_range = np.arange(valid_rows[0], valid_rows[-1] + 1)
        interpolated_midpoints = np.interp(row_range, valid_rows, row_midpoints)
        smoothed_midpoints = cv2.GaussianBlur(
            interpolated_midpoints.astype(np.float32).reshape(-1, 1),
            (1, 5),
            sigmaX=0.0,
            sigmaY=1.0,
        ).reshape(-1)
        center_x = smoothed_midpoints[valid_rows - valid_rows[0]]

    points = np.column_stack((
        center_x,
        valid_rows.astype(np.float32),
    )).astype(np.float32)
    return points, float(confidence)