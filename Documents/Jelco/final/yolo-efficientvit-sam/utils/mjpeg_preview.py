#!/usr/bin/env python3

import cv2
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MJPEGPreviewServer:
    """Serve the latest annotated frame to a browser over an HTTP MJPEG stream."""

    def __init__(self, host="127.0.0.1", port=8765, jpeg_quality=75):
        self.jpeg_quality = int(jpeg_quality)
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._stopping = False

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    page = (
                        "<!doctype html><html><head><title>Road segmentation</title>"
                        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                        "</head><body style='margin:0;background:#111'>"
                        "<img src='/stream.mjpg' style='display:block;width:100%;height:auto'>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()

                sequence = -1
                try:
                    while True:
                        with owner._condition:
                            owner._condition.wait_for(
                                lambda: owner._sequence != sequence or owner._stopping,
                                timeout=2.0,
                            )
                            if owner._stopping:
                                break
                            if owner._sequence == sequence:
                                continue
                            frame = owner._frame
                            sequence = owner._sequence

                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mjpeg-preview-server",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def publish(self, image_bgr):
        success, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            return
        with self._condition:
            self._frame = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)