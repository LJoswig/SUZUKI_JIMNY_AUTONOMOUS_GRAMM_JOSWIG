#!/usr/bin/env python3
"""
thermal_view.py

Live view and recording for the FLIR Boson thermal camera (USB/UVC).

Goal: check how much thermal contrast roads have while driving, before deciding
whether to fuse thermal data into the road segmentation pipeline.

Features:
- Finds the Boson by its USB identity, not by /dev/videoN, because the node
  number changes after every drop of the bad connector.
- Capture thread with watchdog: on a drop it releases the device and keeps
  polling until the camera re-enumerates, then resumes automatically.
- Y16 (raw 16-bit, before the camera's AGC) by default, YUYV as fallback.
- Recording: lossless 16-bit PNG per frame + frames.csv timestamps,
  an 8-bit preview MP4, events.csv (connects/drops) and metadata.json.
  Saved to recordings/<timestamp>/ next to this script.

Keys: q quit, r toggle recording, s snapshot of the current raw frame.
"""

import argparse
import csv
import glob
import json
import os
import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
BY_ID_PATTERNS = ("/dev/v4l/by-id/*FLIR*-video-index0", "/dev/v4l/by-id/*Boson*-video-index0")
SYSFS_NAMES = ("boson", "flir")
WINDOW = "Boson thermal"

RUNNING = True


def handle_signal(sig, frame):
    global RUNNING
    RUNNING = False


def find_boson():
    """Return the current capture node of the Boson, or None if it is not connected."""
    for pattern in BY_ID_PATTERNS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return os.path.realpath(matches[0])

    # Fallback: scan V4L2 device names; index 0 is the capture node (1 = metadata).
    for node in sorted(glob.glob("/sys/class/video4linux/video*"),
                       key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            name = Path(node, "name").read_text().strip().lower()
            index = Path(node, "index").read_text().strip()
        except OSError:
            continue
        if index == "0" and any(key in name for key in SYSFS_NAMES):
            return "/dev/" + os.path.basename(node)
    return None


class BosonCapture:
    """Background capture with automatic reconnect."""

    def __init__(self, fmt="y16", timeout=2.0, device=None, on_frame=None, on_event=None):
        self.fmt = fmt
        self.timeout = timeout
        self.fixed_device = device
        self.on_frame = on_frame
        self.on_event = on_event or (lambda *args: None)

        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self.frame_count = 0
        self.connected = False
        self.device = None
        self.reconnects = 0
        self.disconnected_since = time.monotonic()
        self.actual_format = None
        self.frame_times = deque(maxlen=60)

        self._stop = False
        self._thread = threading.Thread(target=self._run, name="boson-capture", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=self.timeout + 2.0)

    def latest(self):
        with self._lock:
            return self._frame, self._frame_ts, self.frame_count

    def fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        return (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

    def _open(self, path):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        # Make read() return False instead of blocking for OpenCV's default 10 s.
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
        if self.fmt == "y16":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        else:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.actual_format = {
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "driver_fps": cap.get(cv2.CAP_PROP_FPS),
        }
        return cap

    def _run(self):
        cap = None
        last_frame_time = 0.0
        while not self._stop:
            if cap is None:
                path = self.fixed_device or find_boson()
                if path is None or not os.path.exists(path):
                    time.sleep(0.5)
                    continue
                cap = self._open(path)
                if cap is None:
                    time.sleep(0.5)
                    continue
                self.device = path
                last_frame_time = time.monotonic()

            ok, frame = cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                if not self.connected:
                    downtime = now - self.disconnected_since
                    self.connected = True
                    self.on_event("connected", self.device, downtime, self.actual_format)
                    print(f"[thermal] connected {self.device} {self.actual_format} "
                          f"(down {downtime:.1f} s)")
                last_frame_time = now
                self.frame_times.append(now)
                with self._lock:
                    self._frame = frame
                    self._frame_ts = now
                    self.frame_count += 1
                    index = self.frame_count
                if self.on_frame is not None:
                    self.on_frame(frame, now, index)
                continue

            # read() failed: the node is gone or the stream stalled (e.g. bad contact).
            # An FFC shutter freeze lasts ~0.5 s, well below the timeout.
            if os.path.exists(self.device) and now - last_frame_time < self.timeout:
                time.sleep(0.01)
                continue
            cap.release()
            cap = None
            if self.connected:
                self.connected = False
                self.reconnects += 1
                self.disconnected_since = now
                self.on_event("disconnected", self.device, None, None)
                print(f"[thermal] lost {self.device}, waiting for the camera to come back...")
            time.sleep(0.5)

        if cap is not None:
            cap.release()


class Recorder:
    """Writes raw frames on a separate thread so PNG encoding never stalls capture."""

    def __init__(self, out_dir, every=1):
        self.out_dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self.start_wall = datetime.now()
        self.saved = 0
        self.dropped = 0
        self.video = None

        self._csv_file = open(out_dir / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame_index", "t_monotonic", "t_wall", "filename"])
        self._queue = queue.Queue(maxsize=240)
        self._thread = threading.Thread(target=self._run, name="thermal-recorder", daemon=True)
        self._thread.start()

    def add_frame(self, frame, t_mono, index):
        if index % self.every:
            return
        try:
            self._queue.put_nowait((frame, t_mono, time.time(), index))
        except queue.Full:
            self.dropped += 1

    def add_preview(self, image_bgr, fps):
        if self.video is None:
            h, w = image_bgr.shape[:2]
            self.video = cv2.VideoWriter(str(self.out_dir / "preview.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h))
        self.video.write(image_bgr)

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, t_mono, t_wall, index = item
            name = f"{index:07d}.png"
            cv2.imwrite(str(self.frames_dir / name), frame)
            self._csv.writerow([index, f"{t_mono:.6f}", f"{t_wall:.6f}", name])
            self.saved += 1

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._csv_file.close()
        if self.video is not None:
            self.video.release()


def to_display(frame, cmap):
    """Raw Y16 -> 8-bit BGR with a percentile stretch; 8-bit frames pass through."""
    if frame.dtype == np.uint16:
        lo, hi = np.percentile(frame[::4, ::4], (1, 99))
        scale = 255.0 / max(hi - lo, 1.0)
        gray = np.clip((frame.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
    elif frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    if cmap is not None:
        return cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_overlay(image, cam, recording):
    if cam.connected:
        status = f"CONNECTED {cam.device}"
        color = (0, 255, 0)
    else:
        status = f"RECONNECTING ({time.monotonic() - cam.disconnected_since:.1f} s)"
        color = (0, 0, 255)
    lines = [status, f"FPS {cam.fps():.1f}  drops {cam.reconnects}  frames {cam.frame_count}"]
    for i, text in enumerate(lines):
        y = 20 + 20 * i
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if recording:
        cv2.circle(image, (image.shape[1] - 16, 16), 7, (0, 0, 255), -1)
    return image


def main():
    parser = argparse.ArgumentParser(description="FLIR Boson live view with auto-reconnect and recording")
    parser.add_argument("--format", choices=("y16", "yuyv"), default="y16",
                        help="y16 = raw 16-bit (default), yuyv = camera's own 8-bit AGC image")
    parser.add_argument("--device", help="Fixed /dev/videoN (disables auto-discovery)")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="Seconds without frames before treating the camera as disconnected")
    parser.add_argument("--record", action="store_true", help="Start recording immediately")
    parser.add_argument("--record-every", type=int, default=1, help="Save every Nth raw frame")
    parser.add_argument("--cmap", choices=("gray", "inferno", "jet"), default="gray")
    parser.add_argument("--scale", type=float, default=1.5, help="Display window scale")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    cmap = {"gray": None, "inferno": cv2.COLORMAP_INFERNO, "jet": cv2.COLORMAP_JET}[args.cmap]

    session = {"recorder": None, "events": [], "out_dir": None}

    def on_frame(frame, t_mono, index):
        recorder = session["recorder"]
        if recorder is not None:
            recorder.add_frame(frame, t_mono, index)

    def on_event(kind, device, downtime, fmt):
        session["events"].append({
            "t_monotonic": time.monotonic(),
            "t_wall": time.time(),
            "event": kind,
            "device": device,
            "downtime_s": downtime,
            "format": fmt,
        })

    def start_recording():
        out_dir = RECORDINGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        session["out_dir"] = out_dir
        session["events_start"] = len(session["events"])
        session["recorder"] = Recorder(out_dir, args.record_every)
        print(f"[thermal] recording to {out_dir}")

    def stop_recording():
        recorder = session["recorder"]
        if recorder is None:
            return
        session["recorder"] = None
        recorder.close()
        out_dir = session["out_dir"]
        events = session["events"][session["events_start"]:]
        with open(out_dir / "events.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_monotonic", "t_wall", "event", "device", "downtime_s"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(events)
        duration = (datetime.now() - recorder.start_wall).total_seconds()
        metadata = {
            "camera": "FLIR Boson",
            "requested_format": args.format,
            "actual_format": cam.actual_format,
            "args": vars(args),
            "start": recorder.start_wall.isoformat(),
            "stop": datetime.now().isoformat(),
            "duration_s": duration,
            "frames_saved": recorder.saved,
            "frames_dropped_by_recorder": recorder.dropped,
            "mean_saved_fps": recorder.saved * args.record_every / duration if duration > 0 else 0.0,
            "disconnects": sum(1 for e in events if e["event"] == "disconnected"),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[thermal] saved {recorder.saved} frames to {out_dir}")

    cam = BosonCapture(args.format, args.timeout, args.device, on_frame, on_event)
    cam.start()
    if args.record:
        start_recording()

    print("[thermal] waiting for the Boson... (q quit, r record, s snapshot)")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    last_index = -1
    display = None
    warned_format = False
    while RUNNING:
        frame, _, index = cam.latest()
        if cam.connected and frame is not None and index != last_index:
            last_index = index
            if args.format == "y16" and frame.dtype != np.uint16 and not warned_format:
                print(f"[thermal] WARNING: asked for Y16 but got {frame.dtype} {frame.shape}; "
                      f"driver format {cam.actual_format}")
                warned_format = True
            display = to_display(frame, cmap)
            if session["recorder"] is not None:
                session["recorder"].add_preview(display, cam.fps())
            shown = draw_overlay(display.copy(), cam, session["recorder"] is not None)
        else:
            if display is None:
                display = np.zeros((512, 640, 3), np.uint8)
            shown = draw_overlay((display // 3).copy(), cam, session["recorder"] is not None)
        if args.scale != 1.0:
            shown = cv2.resize(shown, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, shown)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            if session["recorder"] is None:
                start_recording()
            else:
                stop_recording()
        if key == ord("s") and frame is not None:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            path = RECORDINGS_DIR / datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), frame)
            print(f"[thermal] snapshot {path}")

    stop_recording()
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
