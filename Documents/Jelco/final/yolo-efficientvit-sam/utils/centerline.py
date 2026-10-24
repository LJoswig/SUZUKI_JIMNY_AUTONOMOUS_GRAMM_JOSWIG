import cv2
import numpy as np
import time
from typing import Optional


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
    """Track the centerline as x-positions at fixed anchor rows.

    The three anchors define a quadratic, so the state describes the same curve as
    polynomial coefficients but stays well conditioned: each state is a pixel position
    inside the band of rows the road usually occupies. The measurement covariance is
    derived from the least-squares geometry of each fit, so anchors outside the rows
    that were actually observed automatically receive large uncertainty.
    """

    OBSERVED = "observed"
    COASTING = "coasting"
    LOST = "lost"

    def __init__(
        self,
        anchor_row_fractions=(0.475, 0.6, 0.725),
        shared_process_noise=200.0,
        independent_process_noise=80.0,
        measurement_noise=16.0,
        max_coast_seconds=1.0,
        max_position_std=35.0,
        innovation_gate_threshold=11.34,
        relock_after_rejections=3,
        min_confidence=0.25,
        min_vertical_span_fraction=0.25,
        min_points=8,
        use_motion_model=False,
        yaw_rate_gain=-60.0,
    ):
        anchor_row_fractions = np.asarray(anchor_row_fractions, dtype=np.float64)
        if anchor_row_fractions.shape != (3,):
            raise ValueError("Kalman anchor rows must contain exactly three fractions")
        if np.any(anchor_row_fractions < 0.0) or np.any(anchor_row_fractions > 1.0):
            raise ValueError("Kalman anchor rows must be fractions in [0, 1]")
        if np.any(np.diff(anchor_row_fractions) <= 0.0):
            raise ValueError("Kalman anchor rows must be strictly increasing")
        if shared_process_noise < 0.0 or independent_process_noise < 0.0:
            raise ValueError("Kalman process noise must be non-negative")
        if shared_process_noise + independent_process_noise <= 0.0:
            raise ValueError("Kalman process noise must not be zero")
        if measurement_noise <= 0.0:
            raise ValueError("Kalman measurement noise must be positive")
        if max_coast_seconds <= 0.0:
            raise ValueError("Kalman max coast duration must be positive")
        if max_position_std <= 0.0:
            raise ValueError("Kalman maximum position uncertainty must be positive")
        if innovation_gate_threshold <= 0.0:
            raise ValueError("Kalman innovation gate threshold must be positive")
        if relock_after_rejections < 1:
            raise ValueError("Kalman relock count must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Kalman minimum confidence must be in [0, 1]")
        if not 0.0 < min_vertical_span_fraction <= 1.0:
            raise ValueError("Kalman minimum vertical span must be in (0, 1]")
        if min_points < 3:
            raise ValueError("Kalman minimum points must be at least 3")

        self.anchor_row_fractions = anchor_row_fractions
        self.shared_process_noise = float(shared_process_noise)
        self.independent_process_noise = float(independent_process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_coast_seconds = float(max_coast_seconds)
        self.max_position_std = float(max_position_std)
        self.innovation_gate_threshold = float(innovation_gate_threshold)
        self.relock_after_rejections = int(relock_after_rejections)
        self.min_confidence = float(min_confidence)
        self.min_vertical_span_fraction = float(min_vertical_span_fraction)
        self.min_points = int(min_points)
        self.use_motion_model = bool(use_motion_model)
        # Horizontal image shift per radian of yaw, in processing-resolution pixels.
        self.yaw_rate_gain = float(yaw_rate_gain)
        self.reset()

    def reset(self):
        self.state = np.zeros(3, dtype=np.float64)
        self.covariance = np.zeros((3, 3), dtype=np.float64)
        self.initialized = False
        self.last_timestamp = None
        self.last_measurement_timestamp = None
        self.image_height = None
        self.anchor_rows = None
        self.anchor_design = None
        self.support_rows = None
        self.status = self.LOST
        self.last_innovation_distance = None
        self.last_rejection_reason = None
        self.relocked = False
        self._rejected_measurements = []

    def _normalize_rows(self, rows, image_height):
        row_center = (image_height - 1.0) / 2.0
        return (np.asarray(rows, dtype=np.float64) - row_center) / max(row_center, 1.0)

    @staticmethod
    def _design_matrix(normalized_rows):
        return np.column_stack((normalized_rows ** 2, normalized_rows, np.ones_like(normalized_rows)))

    def _anchor_design(self, image_height):
        anchor_rows = self.anchor_row_fractions * (image_height - 1.0)
        return anchor_rows, self._design_matrix(
            self._normalize_rows(anchor_rows, image_height)
        )

    def _set_geometry(self, image_height):
        self.image_height = image_height
        self.anchor_rows, self.anchor_design = self._anchor_design(image_height)

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

        design = self._design_matrix(self._normalize_rows(rows, image_height))
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

        inlier_design = design[inliers]
        coefficients, _, _, _ = np.linalg.lstsq(
            inlier_design, x_values[inliers], rcond=None
        )
        if not np.all(np.isfinite(coefficients)):
            self.last_rejection_reason = "invalid_curve_fit"
            return None
        residuals = x_values[inliers] - inlier_design @ coefficients
        residual_variance = float(np.mean(residuals ** 2))
        if image_width is not None:
            support_rows = np.linspace(np.min(rows), np.max(rows), 16)
            fitted_x = self._design_matrix(
                self._normalize_rows(support_rows, image_height)
            ) @ coefficients
            margin = 0.05 * image_width
            if np.any(fitted_x < -margin) or np.any(fitted_x > image_width - 1 + margin):
                self.last_rejection_reason = "curve_outside_image"
                return None

        _, anchor_design = self._anchor_design(image_height)

        # Per-row noise mapped onto the anchors through the fit geometry: about one
        # row's variance at anchors inside the observed span, much more outside it.
        try:
            normalized_information = inlier_design.T @ inlier_design / len(inlier_design)
            anchor_shape = (
                anchor_design
                @ np.linalg.inv(normalized_information)
                @ anchor_design.T
            )
        except np.linalg.LinAlgError:
            self.last_rejection_reason = "singular_curve_fit"
            return None

        confidence = float(np.clip(confidence, self.min_confidence, 1.0))
        row_variance = self.measurement_noise / confidence + residual_variance
        anchor_x = anchor_design @ coefficients
        measurement_covariance = row_variance * (anchor_shape + anchor_shape.T) / 2.0
        return anchor_x, measurement_covariance, rows, image_height

    def _process_covariance(self, delta_time):
        return delta_time * (
            self.shared_process_noise * np.ones((3, 3), dtype=np.float64)
            + self.independent_process_noise * np.eye(3, dtype=np.float64)
        )

    def _predict(self, timestamp, motion_sample=None):
        delta_time = timestamp - self.last_timestamp
        if self.use_motion_model and motion_sample is not None:
            yaw_rate = getattr(motion_sample, "yaw_rate", None)
            if yaw_rate is not None:
                # Yaw shifts every row of the image sideways by the same amount.
                self.state += self.yaw_rate_gain * float(yaw_rate) * delta_time
        self.covariance = self.covariance + self._process_covariance(delta_time)
        self.last_timestamp = timestamp

    def _maximum_position_std(self):
        return float(np.sqrt(max(np.max(np.diag(self.covariance)), 0.0)))

    def _initialize(self, measurement, timestamp):
        anchor_x, measurement_covariance, rows, image_height = measurement
        if image_height != self.image_height:
            self._set_geometry(image_height)
        self.state = anchor_x.copy()
        self.covariance = measurement_covariance.copy()
        self.initialized = True
        self.last_timestamp = timestamp
        self.last_measurement_timestamp = timestamp
        self._set_support_rows(rows)
        self.status = self.OBSERVED
        self.last_innovation_distance = None
        self.last_rejection_reason = None
        self._rejected_measurements = []

    def _set_support_rows(self, rows):
        first_row = float(np.min(rows))
        last_row = float(np.max(rows))
        count = max(2, min(100, int(np.ceil(last_row - first_row)) + 1))
        self.support_rows = np.linspace(first_row, last_row, count)

    def _get_points(self):
        if not self.initialized or self.support_rows is None:
            return None
        coefficients = np.linalg.solve(self.anchor_design, self.state)
        x_values = self._design_matrix(
            self._normalize_rows(self.support_rows, self.image_height)
        ) @ coefficients
        return np.column_stack((x_values, self.support_rows)).astype(np.float32)

    def _mahalanobis(self, difference, covariance):
        try:
            return float(difference @ np.linalg.solve(covariance, difference))
        except np.linalg.LinAlgError:
            return np.inf

    def _agrees_with_rejected(self, measurement, timestamp):
        """Whether a rejected measurement matches the previous rejected one."""
        if not self._rejected_measurements:
            return True
        previous, previous_timestamp = self._rejected_measurements[-1]
        covariance = (
            previous[1]
            + measurement[1]
            + self._process_covariance(max(timestamp - previous_timestamp, 0.0))
        )
        distance = self._mahalanobis(measurement[0] - previous[0], covariance)
        return distance <= self.innovation_gate_threshold

    def _update_measurement(self, measurement, timestamp):
        anchor_x, measurement_covariance, rows, image_height = measurement
        if image_height != self.image_height:
            self._initialize(measurement, timestamp)
            return True

        innovation = anchor_x - self.state
        innovation_covariance = self.covariance + measurement_covariance
        distance = self._mahalanobis(innovation, innovation_covariance)
        self.last_innovation_distance = distance
        if not np.isfinite(distance) or distance > self.innovation_gate_threshold:
            self.last_rejection_reason = "innovation_gate"
            # Several rejected measurements that agree with each other mean the road
            # really moved; re-lock onto it instead of coasting until the track expires.
            if not self._agrees_with_rejected(measurement, timestamp):
                self._rejected_measurements = []
            self._rejected_measurements.append((measurement, timestamp))
            if len(self._rejected_measurements) >= self.relock_after_rejections:
                self._initialize(measurement, timestamp)
                self.relocked = True
                return True
            return False

        gain = np.linalg.solve(innovation_covariance, self.covariance).T
        residual_projection = np.eye(3, dtype=np.float64) - gain @ np.eye(3)
        self.state = self.state + gain @ innovation
        self.covariance = (
            residual_projection @ self.covariance @ residual_projection.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.last_measurement_timestamp = timestamp
        self._set_support_rows(rows)
        self.status = self.OBSERVED
        self.last_rejection_reason = None
        self._rejected_measurements = []
        return True

    def update(
        self,
        points,
        confidence=1.0,
        timestamp=None,
        image_height=None,
        image_width=None,
        motion_sample=None,
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
        self.relocked = False
        measurement = self._fit_measurement(
            points, confidence, int(image_height), image_width
        )

        if self.initialized and timestamp <= self.last_timestamp:
            self.reset()

        if self.initialized:
            self._predict(timestamp, motion_sample=motion_sample)
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