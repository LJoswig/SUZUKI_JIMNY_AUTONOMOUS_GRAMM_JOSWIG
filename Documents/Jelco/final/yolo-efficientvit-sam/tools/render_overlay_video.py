#!/usr/bin/env python3
"""
render_overlay_video.py

Replay a recorded drive through main.py (with its rosbag for path memory and the follower
overlay) and write the preview as a real-time H.264 video that plays anywhere.

Usage (ROS must be sourced for reading the bag):
  source /opt/ros/humble/setup.bash
  .venv_yolo_efficientvit_sam/bin/python tools/render_overlay_video.py /home/user/recordings/run_<date>_<time>

  # SVO and bag in other places:
  ... tools/render_overlay_video.py --svo <file>.svo2 --bag <bag_dir> --name <name>

Output: road_captures/review/<name>_overlay.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Render the preview overlay of a recorded drive.")
    parser.add_argument("run_dir", nargs="?", help="run folder with run.svo2 and rosbag/ (from RECORD_RUN_ENABLED)")
    parser.add_argument("--svo", default=None)
    parser.add_argument("--bag", default=None, help="rosbag of the same drive; without it no path memory/odometry")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default=str(REPO / "road_captures" / "review"))
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        svo = Path(args.svo) if args.svo else run_dir / "run.svo2"
        bag = Path(args.bag) if args.bag else run_dir / "rosbag"
        name = args.name or run_dir.name
    elif args.svo:
        svo, bag = Path(args.svo), Path(args.bag) if args.bag else None
        name = args.name or svo.stem
    else:
        parser.error("give a run folder or --svo")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to write the H.264 video")

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import utils.config as cfg
    work = Path(tempfile.mkdtemp(prefix="overlay_"))
    raw_video = work / "raw.mp4"
    cfg.INPUT_MODE = "svo"
    cfg.SVO_FILE_PATH = str(svo)
    cfg.SVO_REAL_TIME_MODE = False  # Every recorded frame once, as fast as inference allows
    cfg.CENTERLINE_MOTION_SOURCE = "rosbag" if bag is not None and bag.exists() else "none"
    if bag is not None and bag.exists():
        cfg.CENTERLINE_MOTION_ROSBAG_URI = str(bag)
    cfg.RECORD_RUN_ENABLED = False
    cfg.ENABLE_REMOTE_PREVIEW = False
    cfg.CENTERLINE_ROS_PUBLISH_ENABLED = False
    cfg.SAVE_ANNOTATED_VIDEO = True
    cfg.ANNOTATED_VIDEO_PATH = str(raw_video)

    import main as pipeline
    app = pipeline.RealtimeRoadSegmentation()
    stamps = []
    camera_timestamp = app.camera.get_timestamp_seconds

    def recorded_timestamp():
        timestamp = camera_timestamp()
        stamps.append(timestamp)
        return timestamp

    app.camera.get_timestamp_seconds = recorded_timestamp
    app.run()

    # The writer stores frames at the camera frame rate; stretch them to the real drive time.
    if len(stamps) < 2:
        raise RuntimeError("No frames were processed")
    recorded_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    time_scale = app.camera.get_fps() / recorded_fps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{name}_overlay.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-itsscale", f"{time_scale:.5f}", "-i", str(raw_video),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", "-r", "30",
        str(output),
    ], check=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {output} ({len(stamps)} frames, {stamps[-1] - stamps[0]:.0f} s)")


if __name__ == "__main__":
    main()
