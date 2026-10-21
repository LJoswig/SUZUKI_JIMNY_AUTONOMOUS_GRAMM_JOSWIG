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
    """Track a quadratic image-space centerline and its temporal change."""

    OBSERVED = "observed"
    COASTING = "coasting"
    LOST = "lost"

    def __init__(
        self,
        process_noise=2500.0,
        measurement_noise=25.0,
        max_coast_seconds=0.25,
        max_position_std=35.0,
        innovation_gate_threshold=11.34,
        min_confidence=0.25,
        min_vertical_span_fraction=0.15,
        min_points=8,
        initial_velocity_variance=2500.0,
    ):
        if process_noise < 0.0:
            raise ValueError("Kalman process noise must be non-negative")
        if measurement_noise <= 0.0:
            raise ValueError("Kalman measurement noise must be positive")
        if max_coast_seconds <= 0.0:
            raise ValueError("Kalman max coast duration must be positive")
        if max_position_std <= 0.0:
            raise ValueError("Kalman maximum position uncertainty must be positive")
        if innovation_gate_threshold <= 0.0:
            raise ValueError("Kalman innovation gate threshold must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Kalman minimum confidence must be in [0, 1]")
        if not 0.0 < min_vertical_span_fraction <= 1.0:
            raise ValueError("Kalman minimum vertical span must be in (0, 1]")
        if min_points < 3:
            raise ValueError("Kalman minimum points must be at least 3")
        if initial_velocity_variance <= 0.0:
            raise ValueError("Kalman initial velocity variance must be positive")

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_coast_seconds = float(max_coast_seconds)
        self.max_position_std = float(max_position_std)
        self.innovation_gate_threshold = float(innovation_gate_threshold)
        self.min_confidence = float(min_confidence)
        self.min_vertical_span_fraction = float(min_vertical_span_fraction)
        self.min_points = int(min_points)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.reset()

    def reset(self):
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.zeros((6, 6), dtype=np.float64)
        self.initialized = False
        self.last_timestamp = None
        self.last_measurement_timestamp = None
        self.image_height = None
        self.support_rows = None
        self.status = self.LOST
        self.last_innovation_distance = None
        self.last_rejection_reason = None

    @staticmethod
    def _design_matrix(normalized_rows):
        return np.column_stack((normalized_rows ** 2, normalized_rows, np.ones_like(normalized_rows)))

    def _fit_measurement(self, points, confidence, image_height, image_width):
        if points is None or len(points) < self.min_points:
            return None
        if confidence < self.min_confidence:
            self.last_rejection_reason = "low_confidence"
            return None

        current_points = np.asarray(points, dtype=np.float64)
        if current_points.ndim != 2 or current_points.shape[1] != 2:
            self.last_rejection_reason = "invalid_point_shape"
            return None
        if not np.all(np.isfinite(current_points)):
            self.last_rejection_reason = "non_finite_points"
            return None

        x_values = current_points[:, 0]
        rows = current_points[:, 1]
        if np.any(rows < 0.0) or np.any(rows >= image_height):
            self.last_rejection_reason = "rows_outside_image"
            return None
        if np.ptp(rows) < self.min_vertical_span_fraction * image_height:
            self.last_rejection_reason = "insufficient_vertical_span"
            return None

        row_center = (image_height - 1.0) / 2.0
        row_scale = max(row_center, 1.0)
        normalized_rows = (rows - row_center) / row_scale
        design = self._design_matrix(normalized_rows)
        inliers = np.ones(len(rows), dtype=bool)

        for _ in range(3):
            if inliers.sum() < self.min_points:
                self.last_rejection_reason = "too_few_fit_inliers"
                return None
            coefficients, _, _, _ = np.linalg.lstsq(
                design[inliers], x_values[inliers], rcond=None
            )
            residuals = x_values - design @ coefficients
            residual_center = np.median(residuals[inliers])
            residual_mad = np.median(
                np.abs(residuals[inliers] - residual_center)
            )
            tolerance = max(3.0, 3.0 * 1.4826 * residual_mad)
            refined_inliers = np.abs(residuals - residual_center) <= tolerance
            if refined_inliers.sum() < self.min_points:
                break
            if np.array_equal(refined_inliers, inliers):
                break
            inliers = refined_inliers

        coefficients, _, _, _ = np.linalg.lstsq(
            design[inliers], x_values[inliers], rcond=None
        )
        residuals = x_values[inliers] - design[inliers] @ coefficients
        residual_variance = float(np.mean(residuals ** 2))
        if not np.all(np.isfinite(coefficients)):
            self.last_rejection_reason = "invalid_curve_fit"
            return None
        if image_width is not None:
            support_rows = np.linspace(np.min(rows), np.max(rows), 16)
            normalized_support = (support_rows - row_center) / row_scale
            fitted_x = self._design_matrix(normalized_support) @ coefficients
            margin = 0.05 * image_width
            if np.any(fitted_x < -margin) or np.any(fitted_x > image_width - 1 + margin):
                self.last_rejection_reason = "curve_outside_image"
                return None

        confidence = float(np.clip(confidence, self.min_confidence, 1.0))
        variance = self.measurement_noise / confidence + residual_variance
        measurement_covariance = np.eye(3, dtype=np.float64) * variance
        return coefficients, measurement_covariance, rows, image_height

    def _predict(self, timestamp):
        delta_time = timestamp - self.last_timestamp
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3) * delta_time

        dt_squared = delta_time * delta_time
        dt_cubed = dt_squared * delta_time
        dt_fourth = dt_squared * dt_squared
        identity = np.eye(3, dtype=np.float64)
        process_covariance = self.process_noise * np.block([
            [identity * dt_fourth / 4.0, identity * dt_cubed / 2.0],
            [identity * dt_cubed / 2.0, identity * dt_squared],
        ])

        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + process_covariance
        )
        self.last_timestamp = timestamp

    def _maximum_position_std(self):
        maximum_variance = 0.0
        for normalized_row in (-1.0, 0.0, 1.0):
            observation = np.array(
                [normalized_row ** 2, normalized_row, 1.0, 0.0, 0.0, 0.0]
            )
            variance = float(observation @ self.covariance @ observation.T)
            maximum_variance = max(maximum_variance, variance)
        return float(np.sqrt(max(maximum_variance, 0.0)))

    def _initialize(self, measurement, timestamp):
        coefficients, measurement_covariance, rows, image_height = measurement
        self.state[:3] = coefficients
        self.state[3:] = 0.0
        self.covariance.fill(0.0)
        self.covariance[:3, :3] = measurement_covariance
        self.covariance[3:, 3:] = (
            np.eye(3, dtype=np.float64) * self.initial_velocity_variance
        )
        self.initialized = True
        self.last_timestamp = timestamp
        self.last_measurement_timestamp = timestamp
        self.image_height = image_height
        self._set_support_rows(rows)
        self.status = self.OBSERVED
        self.last_innovation_distance = None
        self.last_rejection_reason = None

    def _set_support_rows(self, rows):
        first_row = float(np.min(rows))
        last_row = float(np.max(rows))
        count = max(2, min(100, int(np.ceil(last_row - first_row)) + 1))
        self.support_rows = np.linspace(first_row, last_row, count)

    def _get_points(self):
        if not self.initialized or self.support_rows is None:
            return None
        row_center = (self.image_height - 1.0) / 2.0
        row_scale = max(row_center, 1.0)
        normalized_rows = (self.support_rows - row_center) / row_scale
        x_values = self._design_matrix(normalized_rows) @ self.state[:3]
        return np.column_stack((x_values, self.support_rows)).astype(np.float32)

    def _update_measurement(self, measurement, timestamp):
        coefficients, measurement_covariance, rows, image_height = measurement
        if image_height != self.image_height:
            self._initialize(measurement, timestamp)
            return True

        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        innovation = coefficients - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        try:
            weighted_innovation = np.linalg.solve(
                innovation_covariance, innovation
            )
        except np.linalg.LinAlgError:
            self.last_rejection_reason = "singular_innovation_covariance"
            return False

        distance = float(innovation @ weighted_innovation)
        self.last_innovation_distance = distance
        if not np.isfinite(distance) or distance > self.innovation_gate_threshold:
            self.last_rejection_reason = "innovation_gate"
            return False

        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T
        identity = np.eye(6, dtype=np.float64)
        residual_projection = identity - gain @ observation
        self.state += gain @ innovation
        self.covariance = (
            residual_projection @ self.covariance @ residual_projection.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.last_measurement_timestamp = timestamp
        self._set_support_rows(rows)
        self.status = self.OBSERVED
        self.last_rejection_reason = None
        return True

    def update(
        self,
        points,
        confidence=1.0,
        timestamp=None,
        image_height=None,
        image_width=None,
    ):
        """Update from a curve measurement or coast briefly when it is missing."""
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Kalman timestamp must be finite")

        if image_height is None:
            if points is None or len(points) == 0:
                image_height = self.image_height
            else:
                image_height = int(np.ceil(np.max(np.asarray(points)[:, 1]))) + 1
        if image_height is None or image_height <= 1:
            self.last_rejection_reason = "invalid_image_height"
            self.status = self.LOST
            return None

        self.last_rejection_reason = None
        measurement = self._fit_measurement(
            points, confidence, int(image_height), image_width
        )

        if self.initialized and timestamp <= self.last_timestamp:
            self.reset()

        if self.initialized:
            self._predict(timestamp)
            elapsed_since_measurement = timestamp - self.last_measurement_timestamp
            expired = elapsed_since_measurement > self.max_coast_seconds
            too_uncertain = self._maximum_position_std() > self.max_position_std
            if expired or too_uncertain:
                self.reset()

        if not self.initialized:
            if measurement is None:
                self.status = self.LOST
                return None
            self._initialize(measurement, timestamp)
            return self._get_points()

        accepted = False
        if measurement is not None:
            accepted = self._update_measurement(measurement, timestamp)

        elapsed_since_measurement = timestamp - self.last_measurement_timestamp
        if elapsed_since_measurement > self.max_coast_seconds:
            self.reset()
            return None
        if self._maximum_position_std() > self.max_position_std:
            self.reset()
            return None

        self.status = self.OBSERVED if accepted else self.COASTING
        return self._get_points()


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