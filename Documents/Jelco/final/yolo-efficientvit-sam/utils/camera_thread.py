#!/usr/bin/env python3
"""Grab the camera in its own thread so the auto-stop runs on every frame (AUTO_STOP_THREADED).

The perception loop only takes the newest frame; frames it is too slow for are skipped for
perception but still go through the auto-stop (and into the SVO recording).
"""

import threading
import time
from typing import Optional


class ThreadedCamera:
    """Wraps ZEDCamera / SVOReplayCamera with the same interface; all ZED calls stay in one thread."""

    def __init__(self, camera, auto_stop):
        self.camera = camera
        self.auto_stop = auto_stop
        self.zed = camera.zed

        self._condition = threading.Condition()
        self._frame = None            # (image_rgb, depth, timestamp)
        self._frame_id = 0
        self._handed_out_id = 0
        self._timestamp = None
        self._auto_stop_result = (None, False, None)
        self._auto_stop_count = 0
        self._ended = False
        self._running = False
        self._thread = None
        self.auto_stop_time = 0.0     # seconds, last auto-stop update

    def start(self):
        """Start grabbing; call after enable_recording so every grabbed frame is recorded."""
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while self._running:
            success, image_rgb, depth = self.camera.grab_frame()
            if not success:
                break
            timestamp = self.camera.get_timestamp_seconds()
            if depth is not None:
                depth = depth.copy()  # The sl.Mat buffer is reused by the next grab

            start = time.perf_counter()
            try:
                result = self.auto_stop.update(self.zed)
            except Exception as exc:  # Keep grabbing; a dead camera thread would freeze perception
                print(f"Warning: auto-stop update failed: {exc}")
                result = (None, False, None)
            auto_stop_time = time.perf_counter() - start

            with self._condition:
                self._frame = (image_rgb, depth, timestamp)
                self._frame_id += 1
                self._auto_stop_result = result
                self._auto_stop_count += 1
                self.auto_stop_time = auto_stop_time
                self._condition.notify_all()

        with self._condition:
            self._ended = True
            self._condition.notify_all()

    # --- Same interface as ZEDCamera / SVOReplayCamera ---

    def grab_frame(self):
        """Newest frame not handed out yet; waits for it. (False, None, None) once grabbing ended."""
        with self._condition:
            while self._frame_id == self._handed_out_id and not self._ended:
                self._condition.wait()
            if self._frame_id == self._handed_out_id:
                return False, None, None
            self._handed_out_id = self._frame_id
            image_rgb, depth, self._timestamp = self._frame
            return True, image_rgb, depth

    def get_timestamp_seconds(self) -> Optional[float]:
        return self._timestamp

    def get_auto_stop_result(self):
        """(roi_points, obstacle_detected, roi_info) of the newest grabbed frame, and how many updates ran."""
        with self._condition:
            return self._auto_stop_result, self._auto_stop_count

    def get_resolution(self):
        return self.camera.get_resolution()

    def get_fps(self):
        return self.camera.get_fps()

    def get_calibration(self):
        return self.camera.get_calibration()

    def get_pose(self):
        return self.camera.get_pose()

    def enable_recording(self, svo_path):
        return self.camera.enable_recording(svo_path)

    def stop(self):
        """Stop grabbing (the camera stays open)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self):
        self.stop()
        self.camera.close()
