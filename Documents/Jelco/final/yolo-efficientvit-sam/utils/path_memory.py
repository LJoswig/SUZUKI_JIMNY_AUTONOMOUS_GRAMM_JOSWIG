"""Ground projection of the centerline and odom-frame memory to bridge detection dropouts.

Frames used here:
- camera:    ZED RIGHT_HANDED_Z_UP_X_FWD (x forward, y left, z up), tilted with the camera
- level:     camera position, rotated so z is vertical and x is the camera heading
- base_link: rear axle centre on the ground (x forward, y left)
- odom:      fixed local frame in which the vehicle pose is dead-reckoned
"""

import math

import numpy as np


def _rotation_2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points(points, pose):
    """Map (N, 2) points from a frame located at pose=(x, y, yaw) into its parent frame."""
    x, y, yaw = pose
    return points @ _rotation_2d(yaw).T + np.array([x, y])


def inverse_transform_points(points, pose):
    x, y, yaw = pose
    return (points - np.array([x, y])) @ _rotation_2d(yaw)


def tilt_from_angles(pitch_down_deg, roll_deg=0.0):
    pitch = math.radians(pitch_down_deg)
    roll = math.radians(roll_deg)
    pitch_matrix = np.array([
        [math.cos(pitch), 0.0, math.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch), 0.0, math.cos(pitch)],
    ])
    roll_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])
    return pitch_matrix @ roll_matrix


class GroundProjector:
    """Project processing-resolution pixels onto a flat ground plane around base_link."""

    def __init__(
        self,
        calibration,
        processing_size,
        camera_height,
        camera_x,
        camera_y=0.0,
        camera_yaw_deg=0.0,
        fallback_pitch_down_deg=10.0,
        fallback_roll_deg=0.0,
        max_range=25.0,
    ):
        if camera_height <= 0.0:
            raise ValueError("Camera height above ground must be positive")
        if max_range <= 0.0:
            raise ValueError("Ground projection range must be positive")
        processing_width, processing_height = processing_size
        scale_x = calibration["width"] / float(processing_width)
        scale_y = calibration["height"] / float(processing_height)
        self.fx = calibration["fx"] / scale_x
        self.fy = calibration["fy"] / scale_y
        self.cx = calibration["cx"] / scale_x
        self.cy = calibration["cy"] / scale_y
        self.width = processing_width
        self.height = processing_height
        self.camera_height = float(camera_height)
        self.camera_offset = np.array([camera_x, camera_y], dtype=np.float64)
        self.camera_yaw = math.radians(camera_yaw_deg)
        self.fallback_tilt = tilt_from_angles(fallback_pitch_down_deg, fallback_roll_deg)
        self.max_range = float(max_range)

    def image_to_ground(self, points, tilt=None):
        """Return base_link ground points (M, 2), nearest first, for image points (N, 2)."""
        if points is None or len(points) == 0:
            return None
        tilt = self.fallback_tilt if tilt is None else tilt
        points = np.asarray(points, dtype=np.float64)
        rays = np.column_stack((
            np.ones(len(points)),
            -(points[:, 0] - self.cx) / self.fx,
            -(points[:, 1] - self.cy) / self.fy,
        ))
        level_rays = rays @ tilt.T
        below_horizon = level_rays[:, 2] < -1e-3
        if not np.any(below_horizon):
            return None
        level_rays = level_rays[below_horizon]
        level_points = level_rays[:, :2] * (self.camera_height / -level_rays[:, 2:3])
        in_range = np.hypot(level_points[:, 0], level_points[:, 1]) <= self.max_range
        level_points = level_points[in_range]
        if len(level_points) == 0:
            return None
        ground = transform_points(level_points, (*self.camera_offset, self.camera_yaw))
        return ground[np.argsort(ground[:, 0])]

    def ground_to_image(self, ground_points, tilt=None):
        """Return processing-resolution pixels (M, 2) for base_link ground points (N, 2)."""
        if ground_points is None or len(ground_points) == 0:
            return None
        tilt = self.fallback_tilt if tilt is None else tilt
        level_xy = inverse_transform_points(
            np.asarray(ground_points, dtype=np.float64),
            (*self.camera_offset, self.camera_yaw),
        )
        level_points = np.column_stack((level_xy, np.full(len(level_xy), -self.camera_height)))
        camera_points = level_points @ tilt
        in_front = camera_points[:, 0] > 0.1
        camera_points = camera_points[in_front]
        if len(camera_points) == 0:
            return None
        u = self.cx - self.fx * camera_points[:, 1] / camera_points[:, 0]
        v = self.cy - self.fy * camera_points[:, 2] / camera_points[:, 0]
        margin = 0.25 * self.width
        visible = (u > -margin) & (u < self.width + margin) & (v >= 0.0) & (v < self.height)
        if np.count_nonzero(visible) < 2:
            return None
        return np.column_stack((u[visible], v[visible])).astype(np.float32)


class CenterlinePathMemory:
    """Keep the observed centerline in the odom frame, fuse new observations into it,
    and replay it while the road is not seen.

    Fusion works in the odom frame, so vehicle motion between frames is removed before
    observations are compared. A new observation that disagrees with the stored path is
    treated as a glitch, unless several consecutive ones agree with each other (the road
    really changed, e.g. after a fork), in which case the stored path is replaced.
    """

    OBSERVED = "observed"
    REMEMBERED = "remembered"
    LOST = "lost"

    def __init__(
        self,
        max_memory_seconds=3.0,
        max_memory_distance=8.0,
        min_remaining_length=3.0,
        fusion_weight=1.0,
        mismatch_distance=0.8,
        mismatch_frames=2,
        fusion_step=0.5,
    ):
        if max_memory_seconds <= 0.0:
            raise ValueError("Path memory duration must be positive")
        if max_memory_distance <= 0.0:
            raise ValueError("Path memory distance must be positive")
        if min_remaining_length < 0.0:
            raise ValueError("Path memory remaining length must be non-negative")
        if not 0.0 < fusion_weight <= 1.0:
            raise ValueError("Path fusion weight must be in (0, 1]")
        if mismatch_distance <= 0.0 or fusion_step <= 0.0:
            raise ValueError("Path fusion distances must be positive")
        if mismatch_frames < 1:
            raise ValueError("Path fusion mismatch frames must be at least 1")
        self.max_memory_seconds = float(max_memory_seconds)
        self.max_memory_distance = float(max_memory_distance)
        self.min_remaining_length = float(min_remaining_length)
        # Weight of a new observation; 1.0 replaces the stored path (no fusion).
        self.fusion_weight = float(fusion_weight)
        self.mismatch_distance = float(mismatch_distance)
        self.mismatch_frames = int(mismatch_frames)
        self.fusion_step = float(fusion_step)
        self.reset()

    def reset(self):
        self.odom_points = None
        self.stored_pose = None
        self.stored_epoch = None
        self.stored_timestamp = None
        self.last_recall_age = None
        self.rejected_observations = 0
        self._mismatched = []

    def _lateral_difference(self, path_a, path_b):
        """Median lateral distance of two base_link paths over their overlap, or None."""
        start = max(path_a[0, 0], path_b[0, 0])
        end = min(path_a[-1, 0], path_b[-1, 0])
        if end - start < 2.0 * self.fusion_step:
            return None
        xs = np.arange(start, end, self.fusion_step)
        difference = np.interp(xs, path_b[:, 0], path_b[:, 1]) - np.interp(xs, path_a[:, 0], path_a[:, 1])
        return float(np.median(np.abs(difference)))

    def _fuse(self, stored, observed, pose, epoch):
        """Blend an observation into the stored path (both base_link, sorted by x)."""
        difference = self._lateral_difference(stored, observed)
        if difference is not None and difference > self.mismatch_distance:
            previous = self._mismatched[-1] if self._mismatched else None
            if previous is not None and previous[1] == epoch:
                agreement = self._lateral_difference(inverse_transform_points(previous[0], pose), observed)
                if agreement is None or agreement > self.mismatch_distance:
                    self._mismatched = []
            self._mismatched.append((transform_points(observed, pose), epoch))
            if len(self._mismatched) < self.mismatch_frames:
                self.rejected_observations += 1
                return None
            self._mismatched = []
            return observed
        self._mismatched = []

        xs = np.arange(min(stored[0, 0], observed[0, 0]), max(stored[-1, 0], observed[-1, 0]), self.fusion_step)
        in_stored = (xs >= stored[0, 0]) & (xs <= stored[-1, 0])
        in_observed = (xs >= observed[0, 0]) & (xs <= observed[-1, 0])
        stored_y = np.interp(xs, stored[:, 0], stored[:, 1])
        observed_y = np.interp(xs, observed[:, 0], observed[:, 1])
        fused_y = np.where(
            in_stored & in_observed,
            stored_y + self.fusion_weight * (observed_y - stored_y),
            np.where(in_observed, observed_y, stored_y),
        )
        keep = in_stored | in_observed
        return np.column_stack((xs[keep], fused_y[keep]))

    def store(self, ground_points, pose, epoch, timestamp):
        """Fuse base_link ground points observed at the given odom pose into the stored path."""
        if ground_points is None or len(ground_points) < 2 or pose is None:
            return False
        observed = np.asarray(ground_points, dtype=np.float64)
        observed = observed[np.argsort(observed[:, 0])]
        if self.fusion_weight < 1.0 and self._is_current(pose, epoch, timestamp):
            stored = inverse_transform_points(self.odom_points, pose)
            observed = self._fuse(stored[np.argsort(stored[:, 0])], observed, pose, epoch)
            if observed is None:
                return False
        self.odom_points = transform_points(observed, pose)
        self.stored_pose = tuple(pose)
        self.stored_epoch = epoch
        self.stored_timestamp = float(timestamp)
        return True

    def _is_current(self, pose, epoch, timestamp):
        if self.odom_points is None or pose is None or epoch != self.stored_epoch:
            return False
        age = float(timestamp) - self.stored_timestamp
        travelled = math.hypot(pose[0] - self.stored_pose[0], pose[1] - self.stored_pose[1])
        return 0.0 <= age <= self.max_memory_seconds and travelled <= self.max_memory_distance

    def recall(self, pose, epoch, timestamp):
        """Return the stored path ahead of the vehicle in base_link, or None."""
        self.last_recall_age = None
        if not self._is_current(pose, epoch, timestamp):
            return None
        ground = inverse_transform_points(self.odom_points, pose)
        ahead = ground[ground[:, 0] > 0.0]
        if len(ahead) < 2 or np.max(ahead[:, 0]) < self.min_remaining_length:
            return None
        self.last_recall_age = float(timestamp) - self.stored_timestamp
        return ahead

    def update(self, ground_points, status, pose, epoch, timestamp):
        """Feed one frame's centerline (base_link) and filter status; return (path, status).

        Observed frames are stored (fused). With fusion on, the fused path is used for every
        frame; without it, the stored path only replaces a lost centerline.
        """
        if status == self.OBSERVED:
            self.store(ground_points, pose, epoch, timestamp)
        if status == self.LOST or self.fusion_weight < 1.0:
            path = self.recall(pose, epoch, timestamp)
            if path is not None:
                return path, (self.REMEMBERED if status == self.LOST else status)
        return ground_points, status
