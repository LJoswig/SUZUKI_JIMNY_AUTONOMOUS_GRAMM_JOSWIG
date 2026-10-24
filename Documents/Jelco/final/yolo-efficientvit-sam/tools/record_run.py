#!/usr/bin/env python3
"""
record_run.py

Records raw ZED SVO/SVO2 data for offline vision development.

It saves:
- run.svo2 / run.svo
- rosbag/ (ros2 bag of the vehicle and centerline topics, same run folder)
- metadata.json
- camera_calibration.json
- zed_pose.csv

Important:
- Do NOT run main.py at the same time.
- This script records raw camera data. Vision can be run later at home.
- Source ROS first (source /opt/ros/humble/setup.bash) so the rosbag can be recorded.
"""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pyzed.sl as sl


RUNNING = True

# Recorded next to the SVO so every camera recording has a matching bag.
ROSBAG_TOPICS = [
    # Motion and odometry
    "/imu_attitude",
    "/feedback_data",
    "/steering_feedback",
    "/gps",
    "/gps_raw",
    # Commands, modes and safety state
    "/ackermann_cmd",
    "/rc_inputs",
    "/rc_steer_debug",
    "/mode",
    "/heartbeat",
    "/signal_error",
    # Centerline perception and road follower
    "/centerline/path",
    "/centerline/status",
    "/centerline_follower/cmd_shadow",
    "/centerline_follower/debug",
]


def start_rosbag(bag_dir: Path, topics, log_path: Path):
    """Start `ros2 bag record` in its own process group; returns the process or None."""
    if shutil.which("ros2") is None:
        print("[WARNING] ros2 not found (ROS not sourced?); recording the SVO without a rosbag.")
        return None
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(bag_dir), *topics],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # CTRL+C stops the SVO first; the bag is stopped explicitly after it
    )
    time.sleep(2.0)
    if process.poll() is not None:
        print(f"[WARNING] rosbag recorder exited immediately; see {log_path}")
        return None
    print(f"[INFO] Rosbag recording started: {bag_dir}")
    return process


def stop_rosbag(process) -> bool:
    if process is None:
        return False
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[WARNING] rosbag recorder did not stop in time; terminating it.")
        process.terminate()
        process.wait(timeout=5)
    return process.returncode == 0


def handle_signal(sig, frame):
    global RUNNING
    print("\n[INFO] Stop requested. Closing recording safely...")
    RUNNING = False


def enum_from_string(enum_class, name: str, default):
    name = name.upper()
    return getattr(enum_class, name, default)


def make_run_dir(root: Path, name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = name if name else f"run_{timestamp}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _zed_obj_to_list(obj):
    """
    Convert ZED SDK vector/matrix objects, numpy arrays, or lists to plain Python lists.
    Different ZED SDK versions expose calibration T/R differently.
    """
    try:
        if hasattr(obj, "get"):
            obj = obj.get()
    except Exception:
        pass

    try:
        return [float(x) for x in obj]
    except Exception:
        try:
            return [float(obj)]
        except Exception:
            return str(obj)


def _plain_list(obj):
    """
    Convert ZED SDK objects, numpy arrays, lists, tuples, or scalars
    into JSON-safe Python data.
    """
    if obj is None:
        return []

    try:
        if hasattr(obj, "get"):
            obj = obj.get()
    except Exception:
        pass

    try:
        if hasattr(obj, "tolist"):
            obj = obj.tolist()
    except Exception:
        pass

    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            if isinstance(x, (list, tuple)):
                out.append([float(v) for v in x])
            else:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(str(x))
        return out

    try:
        return [float(x) for x in obj]
    except Exception:
        try:
            return [float(obj)]
        except Exception:
            return str(obj)


def _matrix_from_zed(obj):
    """
    Convert ZED matrix-like objects to nested lists when possible.
    """
    if obj is None:
        return []

    try:
        if hasattr(obj, "m"):
            m = obj.m
            values = [float(x) for x in m]
            if len(values) == 16:
                return [values[i:i+4] for i in range(0, 16, 4)]
            if len(values) == 9:
                return [values[i:i+3] for i in range(0, 9, 3)]
            return values
    except Exception:
        pass

    return _plain_list(obj)


def get_calibration_dict(zed: sl.Camera) -> dict:
    info = zed.get_camera_information()
    cam_cfg = info.camera_configuration
    calib = cam_cfg.calibration_parameters

    stereo_translation = []
    stereo_rotation = []
    stereo_transform_matrix = []

    # ZED SDK 5 style
    stereo_transform = getattr(calib, "stereo_transform", None)
    if stereo_transform is not None:
        try:
            stereo_translation = _plain_list(stereo_transform.get_translation())
        except Exception:
            pass

        try:
            stereo_rotation = _matrix_from_zed(stereo_transform.get_rotation_matrix())
        except Exception:
            pass

        try:
            stereo_transform_matrix = _matrix_from_zed(stereo_transform)
        except Exception:
            pass

    # Older SDK compatibility
    if not stereo_translation and hasattr(calib, "T"):
        stereo_translation = _plain_list(calib.T)

    if not stereo_rotation and hasattr(calib, "R"):
        stereo_rotation = _plain_list(calib.R)

    baseline_m = ""
    try:
        baseline_m = float(calib.get_camera_baseline())
    except Exception:
        try:
            baseline_m = float(calib.getCameraBaseline())
        except Exception:
            pass

    return {
        "serial_number": int(info.serial_number),
        "resolution": {
            "width": int(cam_cfg.resolution.width),
            "height": int(cam_cfg.resolution.height),
        },
        "fps": float(cam_cfg.fps),
        "left_camera": {
            "fx": float(calib.left_cam.fx),
            "fy": float(calib.left_cam.fy),
            "cx": float(calib.left_cam.cx),
            "cy": float(calib.left_cam.cy),
            "disto": _plain_list(calib.left_cam.disto),
        },
        "right_camera": {
            "fx": float(calib.right_cam.fx),
            "fy": float(calib.right_cam.fy),
            "cx": float(calib.right_cam.cx),
            "cy": float(calib.right_cam.cy),
            "disto": _plain_list(calib.right_cam.disto),
        },
        "stereo_transform": {
            "translation": stereo_translation,
            "rotation": stereo_rotation,
            "matrix": stereo_transform_matrix,
            "baseline_m": baseline_m,
        },
    }


def safe_pose_dict(zed: sl.Camera) -> dict:
    pose = sl.Pose()
    state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

    out = {
        "tracking_state": str(state),
        "tx": "",
        "ty": "",
        "tz": "",
        "qx": "",
        "qy": "",
        "qz": "",
        "qw": "",
        "rx": "",
        "ry": "",
        "rz": "",
    }

    if state == sl.POSITIONAL_TRACKING_STATE.OK:
        try:
            t = pose.get_translation().get()
            out["tx"] = float(t[0])
            out["ty"] = float(t[1])
            out["tz"] = float(t[2])
        except Exception:
            pass

        try:
            q = pose.get_orientation().get()
            out["qx"] = float(q[0])
            out["qy"] = float(q[1])
            out["qz"] = float(q[2])
            out["qw"] = float(q[3])
        except Exception:
            pass

        try:
            r = pose.get_rotation_vector()
            out["rx"] = float(r[0])
            out["ry"] = float(r[1])
            out["rz"] = float(r[2])
        except Exception:
            pass

    return out


def main():
    parser = argparse.ArgumentParser(description="Record ZED SVO/SVO2 dataset.")
    parser.add_argument("--out", type=str, default="/home/user/recordings",
                        help="Root output folder.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run folder name. Default: timestamped run name.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Duration in seconds. 0 means record until CTRL+C.")
    parser.add_argument("--resolution", type=str, default="SVGA",
                        choices=["HD2K", "HD1080", "HD720", "SVGA", "VGA", "AUTO"],
                        help="ZED resolution.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Camera FPS.")
    parser.add_argument("--depth-mode", type=str, default="NEURAL",
                        choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"],
                        help="Depth mode used during recording session.")
    parser.add_argument("--compression", type=str, default="H265",
                        help="SVO compression: H264, H265, H264_LOSSLESS, H265_LOSSLESS, LOSSLESS.")
    parser.add_argument("--notes", type=str, default="",
                        help="Short notes about this run.")
    parser.add_argument("--min-free-gb", type=float, default=10.0,
                        help="Refuse to start if output drive has less free space.")
    parser.add_argument("--no-rosbag", action="store_true",
                        help="record only the SVO, without the matching rosbag")
    parser.add_argument("--pose-every-n", type=int, default=1,
                        help="Log ZED pose every N frames.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    free_gb = disk_free_gb(root)
    if free_gb < args.min_free_gb:
        print(f"[ERROR] Not enough disk space in {root}: {free_gb:.1f} GB free.")
        print(f"[ERROR] Required minimum: {args.min_free_gb:.1f} GB.")
        sys.exit(1)

    run_dir = make_run_dir(root, args.name)

    svo_path = run_dir / "run.svo2"
    bag_dir = run_dir / "rosbag"
    metadata_path = run_dir / "metadata.json"
    calibration_path = run_dir / "camera_calibration.json"
    pose_csv_path = run_dir / "zed_pose.csv"
    notes_path = run_dir / "notes.txt"

    print("=" * 80)
    print("ZED RAW RECORDING")
    print("=" * 80)
    print(f"Run folder:      {run_dir}")
    print(f"SVO output:      {svo_path}")
    print(f"Free disk:       {free_gb:.1f} GB")
    print(f"Duration:        {'until CTRL+C' if args.duration <= 0 else str(args.duration) + ' s'}")
    print(f"Resolution/FPS:  {args.resolution} @ {args.fps}")
    print(f"Depth mode:      {args.depth_mode}")
    print(f"Compression:     {args.compression}")
    print("=" * 80)

    # -------------------------
    # Open ZED
    # -------------------------
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = enum_from_string(
        sl.RESOLUTION, args.resolution, sl.RESOLUTION.SVGA
    )
    init_params.camera_fps = args.fps
    init_params.depth_mode = enum_from_string(
        sl.DEPTH_MODE, args.depth_mode, sl.DEPTH_MODE.NEURAL
    )
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open ZED camera: {err}")
        sys.exit(1)

    # -------------------------
    # Enable positional tracking
    # -------------------------
    tracking_enabled = False
    try:
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = True
        err = zed.enable_positional_tracking(tracking_params)
        if err == sl.ERROR_CODE.SUCCESS:
            tracking_enabled = True
            print("[INFO] Positional tracking enabled.")
        else:
            print(f"[WARNING] Positional tracking not enabled: {err}")
    except Exception as e:
        print(f"[WARNING] Positional tracking setup failed: {e}")

    # -------------------------
    # Save calibration and metadata
    # -------------------------
    calibration = get_calibration_dict(zed)
    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=2)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "svo_path": str(svo_path),
        "resolution": args.resolution,
        "fps": args.fps,
        "depth_mode": args.depth_mode,
        "compression": args.compression,
        "duration_requested_s": args.duration,
        "pose_logging_every_n_frames": args.pose_every_n,
        "positional_tracking_enabled": tracking_enabled,
        "rosbag_path": None if args.no_rosbag else str(bag_dir),
        "rosbag_topics": [] if args.no_rosbag else ROSBAG_TOPICS,
        "notes": args.notes,
        "important": {
            "vision_algorithm_was_not_run": True,
            "purpose": "raw ZED recording for offline replay and later vision development",
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(notes_path, "w") as f:
        f.write(args.notes + "\n")

    # -------------------------
    # Enable SVO recording
    # -------------------------
    compression = enum_from_string(
        sl.SVO_COMPRESSION_MODE,
        args.compression,
        sl.SVO_COMPRESSION_MODE.H265
    )

    rec_params = sl.RecordingParameters()
    rec_params.video_filename = str(svo_path)
    rec_params.compression_mode = compression

    err = zed.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print(f"[ERROR] Failed to enable SVO recording: {err}")
        sys.exit(1)

    runtime_params = sl.RuntimeParameters()

    rosbag_process = None
    if not args.no_rosbag:
        rosbag_process = start_rosbag(bag_dir, ROSBAG_TOPICS, run_dir / "rosbag_record.log")
    metadata["rosbag_started"] = rosbag_process is not None

    frame_id = 0
    start_time = time.time()
    last_print = time.time()

    print("[INFO] Recording started.")
    print("[INFO] Press CTRL+C to stop safely.")

    try:
        with open(pose_csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_id",
                "system_time_ns",
                "elapsed_s",
                "zed_timestamp_ns",
                "tracking_state",
                "tx", "ty", "tz",
                "qx", "qy", "qz", "qw",
                "rx", "ry", "rz",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while RUNNING:
                now = time.time()
                elapsed = now - start_time
                if args.duration > 0 and elapsed >= args.duration:
                    print("[INFO] Requested duration reached.")
                    break

                err = zed.grab(runtime_params)
                if err == sl.ERROR_CODE.SUCCESS:
                    # In most ZED SDK Python versions, enabled recording records grabbed frames.
                    # Some versions expose zed.record(); call it if available.
                    if hasattr(zed, "record"):
                        try:
                            zed.record()
                        except Exception:
                            pass

                    if frame_id % args.pose_every_n == 0:
                        timestamp_ns = ""
                        try:
                            timestamp_ns = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                        except Exception:
                            pass

                        pose_data = safe_pose_dict(zed)
                        row = {
                            "frame_id": frame_id,
                            "system_time_ns": time.time_ns(),
                            "elapsed_s": elapsed,
                            "zed_timestamp_ns": timestamp_ns,
                            **pose_data,
                        }
                        writer.writerow(row)

                    frame_id += 1

                    if time.time() - last_print > 2.0:
                        current_free_gb = disk_free_gb(root)
                        print(
                            f"[INFO] frames={frame_id} "
                            f"elapsed={elapsed:.1f}s "
                            f"free={current_free_gb:.1f}GB"
                        )
                        last_print = time.time()

                else:
                    print(f"[WARNING] grab failed: {err}")
                    time.sleep(0.01)

    finally:
        print("[INFO] Finalizing recording...")
        try:
            zed.disable_recording()
        except Exception:
            pass
        # Stop the bag only after the SVO is closed so it covers the whole recording.
        metadata["rosbag_closed_cleanly"] = stop_rosbag(rosbag_process)

    try:
        zed.disable_positional_tracking()
    except Exception:
        pass

    zed.close()

    final_size_gb = svo_path.stat().st_size / (1024 ** 3) if svo_path.exists() else 0.0

    metadata["frames_recorded"] = frame_id
    metadata["duration_actual_s"] = time.time() - start_time
    metadata["svo_size_gb"] = final_size_gb
    metadata["closed_cleanly"] = True

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80)
    print("[DONE] Recording finished.")
    print(f"Run folder: {run_dir}")
    print(f"Frames:     {frame_id}")
    print(f"SVO size:   {final_size_gb:.2f} GB")
    if rosbag_process is not None:
        print(f"Rosbag:     {bag_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
