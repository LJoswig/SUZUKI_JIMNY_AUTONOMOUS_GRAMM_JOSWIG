#!/usr/bin/env python3
"""
shadow_follower.py

Offline shadow test of the pure pursuit road follower against a recorded drive.

Phase 1 (perception, cached): replays the SVO in real time like the live pipeline and
stores the filtered centerline per frame plus the measured processing delay.
Phase 2 (analysis): dead-reckons the vehicle from the rosbag, applies path memory,
runs the follower at FOLLOWER_RATE_HZ with each path arriving after its real delay,
and compares the commanded steering with the human RC steering from the bag.

Usage (ROS must be sourced for reading the bag):
  source /opt/ros/humble/setup.bash
  .venv_yolo_efficientvit_sam/bin/python tools/shadow_follower.py \
      --svo /home/user/Documents/testfahrt_20261021_181107.svo2 \
      --bag /home/user/Documents/testfahrt_20261021_181107_rosbag \
      --steering-offset-deg -7

Outputs (in --out-dir): <run>_shadow.csv, <run>_shadow.png, <run>_summary.txt
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import utils.config as cfg  # noqa: E402
from utils.follower import PurePursuitFollower  # noqa: E402
from utils.motion_inputs import PlanarOdometry  # noqa: E402
from utils.path_memory import CenterlinePathMemory, GroundProjector  # noqa: E402

STEERING_LOSS_DEG = 90.0  # Measured vs commanded steering apart by more than this ...
STEERING_LOSS_SECONDS = 1.0  # ... for this long marks a steering tracking loss (excluded)
TURN_THRESHOLD_DEG = 60.0  # Human steering beyond this counts as a curve for the sign check
MIN_MOVING_SPEED = 0.5  # m/s


# ---------------------------------------------------------------------------
# Phase 1: perception
# ---------------------------------------------------------------------------
def run_perception(svo_path, cache_path):
    os.chdir(REPO)
    cfg.INPUT_MODE = "svo"
    cfg.SVO_FILE_PATH = svo_path
    cfg.SVO_REAL_TIME_MODE = True  # Skip frames like the live pipeline does
    cfg.ENABLE_REMOTE_PREVIEW = False
    cfg.SHOW_LOCAL_WINDOW = False
    cfg.SAVE_ANNOTATED_VIDEO = False
    cfg.CENTERLINE_KALMAN_SENSOR_MODE = "visual_only"
    cfg.CENTERLINE_MOTION_SOURCE = "none"
    cfg.CENTERLINE_PATH_MEMORY_ENABLED = False  # Applied in phase 2 with bag odometry

    import cv2
    import main as pipeline
    from utils.centerline import extract_centerline

    app = pipeline.RealtimeRoadSegmentation()
    records = []
    started = time.time()
    try:
        while True:
            ok, image, _ = app.camera.grab_frame()
            if not ok:
                break
            grabbed = time.perf_counter()
            timestamp = app.camera.get_timestamp_seconds()
            cv2.resize(image, (app.processing_width, app.processing_height),
                       dst=app.processing_buffer, interpolation=cv2.INTER_AREA)
            detections, _ = app.process_frame(app.processing_buffer)
            points, confidence = None, 0.0
            if detections is not None:
                points, confidence = extract_centerline(
                    detections["centerline_mask"],
                    use_curve_fit=cfg.CENTERLINE_USE_CURVE_FIT,
                    fit_degree=cfg.CENTERLINE_POLYNOMIAL_DEGREE,
                )
            filtered = app.centerline_kalman_filter.update(
                points, confidence, timestamp=timestamp,
                image_height=app.processing_height, image_width=app.processing_width,
            )
            records.append(dict(
                timestamp=timestamp,
                points=None if filtered is None else np.asarray(filtered),
                status=app.centerline_kalman_filter.status,
                latency=time.perf_counter() - grabbed,
            ))
            if len(records) % 100 == 0:
                print(f"[perception] {len(records)} frames, {time.time() - started:.0f} s")
    finally:
        app.cleanup()

    cache = dict(
        svo=svo_path,
        records=records,
        calibration=app.camera.get_calibration(),
        processing_size=(app.processing_width, app.processing_height),
    )
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[perception] cached {len(records)} frames to {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Phase 2: analysis
# ---------------------------------------------------------------------------
def read_bag(bag_uri):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = {
        cfg.CENTERLINE_TOPIC_IMU_ATTITUDE: "imu",
        cfg.CENTERLINE_TOPIC_FEEDBACK_DATA: "speed",
        cfg.CENTERLINE_TOPIC_STEERING_FEEDBACK: "steering_meas",
        cfg.CENTERLINE_TOPIC_ACKERMANN_CMD: "steering_cmd",
    }
    classes = {name: get_message(types[name]) for name in topics if name in types}
    missing = [name for name in topics if name not in types]
    if missing:
        raise RuntimeError(f"Bag is missing topics: {missing}")
    data = {key: [] for key in topics.values()}
    messages = []  # time-ordered (t, kind, value) for odometry
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in classes:
            continue
        msg = deserialize_message(payload, classes[topic])
        t = stamp_ns * 1e-9
        key = topics[topic]
        if key == "imu":
            value = msg.angular_velocity.z
            messages.append((t, "imu", value))
        elif key == "speed":
            if len(msg.data) < 2:
                continue
            value = msg.data[1]
            messages.append((t, "speed", value))
        elif key == "steering_meas":
            if len(msg.data) < 1:
                continue
            value = msg.data[0]
        else:
            value = msg.steering_angle
        data[key].append((t, value))
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}, messages


class OdometryTrack:
    """Dead-reckoned poses over the whole bag, queried by time like the live provider."""

    def __init__(self, messages):
        odometry = PlanarOdometry(
            yaw_rate_sign=cfg.CENTERLINE_ODOM_YAW_RATE_SIGN,
            max_speed_age_seconds=cfg.CENTERLINE_ODOM_MAX_SPEED_AGE_SECONDS,
        )
        times, poses, epochs = [], [], []
        for t, kind, value in messages:
            if kind == "speed":
                odometry.update_speed(value, t)
                continue
            odometry.update_yaw_rate(value, t)
            pose, epoch = odometry.pose(t)
            times.append(t)
            poses.append(pose)
            epochs.append(epoch)
        self.times = np.array(times)
        self.poses = poses
        self.epochs = epochs
        self.max_step = odometry.max_step_seconds

    def at(self, t):
        index = np.searchsorted(self.times, t, side="right") - 1
        if index < 0 or t - self.times[index] > self.max_step:
            return None, None
        return self.poses[index], self.epochs[index]


def value_at(series, t):
    index = np.searchsorted(series[:, 0], t, side="right") - 1
    return np.nan if index < 0 else series[index, 1]


def build_path_events(cache, odometry, projector, memory, fixed_latency):
    """Per perception frame: the ground path the follower would receive, and when."""
    events = []
    for record in cache["records"]:
        t = record["timestamp"]
        pose, epoch = odometry.at(t)
        status = record["status"]
        ground = None
        if record["points"] is not None:
            ground = projector.image_to_ground(record["points"])
        if status == "observed":
            memory.store(ground, pose, epoch, t)
        elif status == "lost":
            ground = memory.recall(pose, epoch, t)
            status = CenterlinePathMemory.REMEMBERED if ground is not None else status
        latency = record["latency"] if fixed_latency is None else fixed_latency
        events.append((t + latency, t, ground, pose, epoch, status))
    return events


def run_follower(events, odometry, bag, follower, rate_hz):
    start, end = events[0][0], events[-1][0]
    grid = np.arange(start, end, 1.0 / rate_hz)
    rows = []
    next_event = 0
    for t in grid:
        while next_event < len(events) and events[next_event][0] <= t:
            _, stamp, ground, pose, epoch, status = events[next_event]
            if ground is not None:
                follower.update_path(ground, pose, epoch, stamp, status)
            next_event += 1
        pose, epoch = odometry.at(t)
        command = follower.compute(pose, epoch, t)
        rows.append((
            t, command.steering_wheel_deg, command.speed, command.tracking, command.reason,
            command.path_status or "", np.nan if command.path_age is None else command.path_age,
            value_at(bag["steering_cmd"], t), value_at(bag["steering_meas"], t), value_at(bag["speed"], t),
        ))
    return rows


def steering_loss_mask(measured, commanded, rate_hz):
    apart = np.abs(measured - commanded) > STEERING_LOSS_DEG
    mask = np.zeros_like(apart)
    min_samples = int(STEERING_LOSS_SECONDS * rate_hz)
    start = None
    for i, value in enumerate(np.append(apart, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_samples:
                mask[start:i] = True
            start = None
    return mask


def summarize(rows, offset_deg, rate_hz):
    t = np.array([r[0] for r in rows])
    follower = np.array([r[1] for r in rows])
    tracking = np.array([r[3] for r in rows])
    reasons = np.array([r[4] for r in rows])
    path_status = np.array([r[5] for r in rows])
    human_cmd = np.array([r[7] for r in rows]) - offset_deg
    human_meas = np.array([r[8] for r in rows]) - offset_deg
    speed = np.array([r[9] for r in rows])

    moving = np.nan_to_num(speed) > MIN_MOVING_SPEED
    loss = steering_loss_mask(human_meas, human_cmd, rate_hz)
    valid = moving & ~loss & np.isfinite(human_cmd)
    compare = valid & tracking
    lines = []
    add = lines.append
    add(f"duration analysed: {t[-1] - t[0]:.0f} s, moving {moving.mean():.0%}, "
        f"excluded for steering tracking loss {loss.mean():.1%}")
    add(f"steering offset applied to human steering: {offset_deg:+.1f} deg")
    add("")
    add(f"follower tracking while moving: {tracking[valid].mean():.1%}")
    not_tracking = valid & ~tracking
    if not_tracking.any():
        values, counts = np.unique(reasons[not_tracking], return_counts=True)
        add("  not tracking because: " + ", ".join(f"{v} {c / valid.sum():.1%}" for v, c in zip(values, counts)))
    longest, current = 0, 0
    for value in not_tracking[valid]:
        current = current + 1 if value else 0
        longest = max(longest, current)
    add(f"  longest stretch without tracking: {longest / rate_hz:.1f} s")
    values, counts = np.unique(path_status[compare], return_counts=True)
    add("  path source while tracking: " + ", ".join(f"{v} {c / compare.sum():.1%}" for v, c in zip(values, counts)))
    add("")
    if compare.sum() < 10:
        add("too few samples to compare steering")
        return lines
    for name, human in (("human command", human_cmd), ("human measured", human_meas)):
        compare = valid & tracking & np.isfinite(human)
        if compare.sum() < 10:
            add(f"follower vs {name}: too few samples")
            continue
        diff = follower[compare] - human[compare]
        best_lag, best_corr = 0.0, -2.0
        for lag_samples in range(-int(1.5 * rate_hz), int(1.5 * rate_hz) + 1):
            shifted = np.roll(follower, lag_samples)
            corr = np.corrcoef(shifted[compare], human[compare])[0, 1]
            if corr > best_corr:
                best_lag, best_corr = lag_samples / rate_hz, corr
        turns = compare & (np.abs(human) > TURN_THRESHOLD_DEG)
        sign_ok = np.mean(np.sign(follower[turns]) == np.sign(human[turns])) if turns.any() else np.nan
        add(f"follower vs {name}:")
        add(f"  correlation {np.corrcoef(follower[compare], human[compare])[0, 1]:.2f} "
            f"(best {best_corr:.2f} with follower shifted {best_lag:+.2f} s; positive = follower later)")
        add(f"  |difference| median {np.median(np.abs(diff)):.0f} deg, p90 {np.percentile(np.abs(diff), 90):.0f} deg, "
            f"> 90 deg {np.mean(np.abs(diff) > 90):.1%} of the time")
        add(f"  mean difference {np.mean(diff):+.1f} deg (a constant bias here hints at a steering offset)")
        add(f"  same turn direction in curves (|human| > {TURN_THRESHOLD_DEG:.0f} deg): {sign_ok:.1%} of {turns.sum()} samples")
    return lines


def write_outputs(rows, summary, out_dir, name, offset_deg):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_shadow.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "follower_steering_deg", "follower_speed", "tracking", "reason", "path_status",
                         "path_age_s", "human_cmd_deg", "human_meas_deg", "vehicle_speed"])
        writer.writerows(rows)
    summary_path = out_dir / f"{name}_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r[0] for r in rows]); t = t - t[0]
    tracking = np.array([r[3] for r in rows])
    fig, (ax, ax_speed) = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax.plot(t, np.array([r[7] for r in rows]) - offset_deg, color="0.55", lw=1, label="human command")
    ax.plot(t, np.array([r[8] for r in rows]) - offset_deg, color="0.2", lw=1, label="human measured")
    follower = np.array([r[1] for r in rows], dtype=float)
    ax.plot(t, np.where(tracking, follower, np.nan), color="tab:blue", lw=1.3, label="follower (tracking)")
    ax.plot(t, np.where(tracking, np.nan, follower), color="tab:red", lw=1.3, label="follower (holding, speed 0)")
    ax.set_ylabel("steering wheel [deg], + = right")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(alpha=0.3)
    ax_speed.plot(t, [r[9] for r in rows], color="0.3", lw=1)
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_xlabel("time [s]")
    ax_speed.grid(alpha=0.3)
    fig.suptitle(f"{name}: shadow follower vs human steering")
    fig.tight_layout()
    plot_path = out_dir / f"{name}_shadow.png"
    fig.savefig(plot_path, dpi=110)
    return csv_path, summary_path, plot_path


def main():
    parser = argparse.ArgumentParser(description="Offline shadow test of the road follower.")
    parser.add_argument("--svo", default=cfg.SVO_FILE_PATH)
    parser.add_argument("--bag", default=cfg.CENTERLINE_MOTION_ROSBAG_URI)
    parser.add_argument("--cache", default=None, help="perception cache (default: <out-dir>/<run>_perception.pkl)")
    parser.add_argument("--rerun-perception", action="store_true", help="ignore an existing perception cache")
    parser.add_argument("--out-dir", default=str(REPO / "road_captures" / "shadow"))
    parser.add_argument("--steering-offset-deg", type=float, default=0.0,
                        help="measured steering-wheel angle when driving straight (about -7 with the current "
                             "encoder); subtracted from the human steering for the comparison only")
    parser.add_argument("--latency", type=float, default=None,
                        help="fixed perception delay in seconds instead of the measured per-frame delay")
    parser.add_argument("--lookahead", type=float, default=cfg.FOLLOWER_LOOKAHEAD_DISTANCE)
    parser.add_argument("--steering-ratio", type=float, default=cfg.FOLLOWER_STEERING_RATIO)
    parser.add_argument("--max-steering-rate", type=float, default=cfg.FOLLOWER_MAX_STEERING_RATE_DEG_S)
    args = parser.parse_args()

    name = Path(args.svo).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_dir / f"{name}_perception.pkl"
    if cache_path.exists() and not args.rerun_perception:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[perception] using cache {cache_path} ({len(cache['records'])} frames)")
    else:
        cache = run_perception(args.svo, cache_path)

    bag, messages = read_bag(args.bag)
    odometry = OdometryTrack(messages)
    projector = GroundProjector(
        cache["calibration"], cache["processing_size"],
        camera_height=cfg.CENTERLINE_CAMERA_HEIGHT, camera_x=cfg.CENTERLINE_CAMERA_X,
        camera_y=cfg.CENTERLINE_CAMERA_Y, camera_yaw_deg=cfg.CENTERLINE_CAMERA_YAW_DEG,
        fallback_pitch_down_deg=cfg.CENTERLINE_CAMERA_PITCH_DOWN_DEG, max_range=cfg.CENTERLINE_GROUND_MAX_RANGE,
    )
    memory = CenterlinePathMemory(
        max_memory_seconds=cfg.CENTERLINE_PATH_MEMORY_MAX_SECONDS,
        max_memory_distance=cfg.CENTERLINE_PATH_MEMORY_MAX_DISTANCE,
        min_remaining_length=cfg.CENTERLINE_PATH_MEMORY_MIN_REMAINING,
    )
    follower = PurePursuitFollower(
        wheelbase=cfg.FOLLOWER_WHEELBASE, steering_ratio=args.steering_ratio, steering_sign=cfg.FOLLOWER_STEERING_SIGN,
        lookahead_distance=args.lookahead, max_steering_wheel_deg=cfg.FOLLOWER_MAX_STEERING_WHEEL_DEG,
        max_steering_rate_deg_s=args.max_steering_rate, target_speed=cfg.FOLLOWER_TARGET_SPEED,
        max_path_age_seconds=cfg.FOLLOWER_MAX_PATH_AGE_SECONDS,
    )
    events = build_path_events(cache, odometry, projector, memory, args.latency)
    if not any(e[3] is not None for e in events):
        raise RuntimeError("No odometry overlaps the SVO timestamps; is this the matching bag?")
    rows = run_follower(events, odometry, bag, follower, cfg.FOLLOWER_RATE_HZ)

    latencies = np.array([r["latency"] for r in cache["records"]])
    summary = [
        f"run: {name}",
        f"follower: lookahead {args.lookahead} m, steering ratio {args.steering_ratio}, "
        f"max rate {args.max_steering_rate} deg/s",
        f"perception: {len(latencies)} frames, delay median {np.median(latencies):.2f} s, "
        f"p90 {np.percentile(latencies, 90):.2f} s" + (f" (using fixed {args.latency} s)" if args.latency else ""),
    ] + summarize(rows, args.steering_offset_deg, cfg.FOLLOWER_RATE_HZ)
    paths = write_outputs(rows, summary, out_dir, name, args.steering_offset_deg)
    print("\n".join(summary))
    print("\nwrote:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
