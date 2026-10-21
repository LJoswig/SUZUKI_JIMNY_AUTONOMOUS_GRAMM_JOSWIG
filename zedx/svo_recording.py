########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

#Changes:
#Records ZED X camera data to an SVO file for offline testing.
#Saves recordings in svo_recording_data next to this script,
#using a timestamped filename.
#Goal: Record test drive with ZEDX as an SVO file for repeatable offline
#testing and tuning of centerline extraction and spatial/temporal smoothing.

import sys
import pyzed.sl as sl
import argparse
import os
import subprocess
from signal import signal, SIGINT, SIGTERM

from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "svo_recording_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    datetime.now().strftime("testfahrt_%Y%m%d_%H%M%S.svo2")
)

cam = sl.Camera()
rosbag_process = None
stop_requested = False

# Defer shutdown to the main loop so both recordings can be finalized.
def handler(signal_received, frame):
    global stop_requested
    stop_requested = True

signal(SIGINT, handler)
signal(SIGTERM, handler)


def stop_rosbag(process):
    if process is None or process.poll() is not None:
        return

    print("Stopping ROS bag recording...")
    process.send_signal(SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("ROS bag did not stop cleanly; terminating it.")
        process.terminate()
        process.wait(timeout=5)

def main(opt):
    global rosbag_process

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE # Set configuration parameters for the ZED
    
    status = cam.open(init) 
    if status > sl.ERROR_CODE.SUCCESS: 
        print("Camera Open", status, "Exit program.")
        return 1

    recording_enabled = False
    bag_output = os.path.splitext(os.path.abspath(opt.output_svo_file))[0] + "_rosbag"
    try:
        recording_param = sl.RecordingParameters(
            opt.output_svo_file,
            sl.SVO_COMPRESSION_MODE.H265,
        )
        err = cam.enable_recording(recording_param)
        if err > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable SVO recording: {err}")
        recording_enabled = True

        print(f"Starting ROS 2 bag recording to: {bag_output}")
        rosbag_process = subprocess.Popen([
            "ros2", "bag", "record", "--all", "--output", bag_output
        ])

        print("SVO and ROS 2 bag are recording. Use Ctrl-C to stop.")
        frames_recorded = 0
        while not stop_requested:
            if rosbag_process.poll() is not None:
                raise RuntimeError(
                    f"ROS bag recorder exited with code {rosbag_process.returncode}"
                )

            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                frames_recorded += 1
                print(f"Frame count: {frames_recorded}", end="\r", flush=True)
            else:
                print("\nCamera frame grab failed; stopping both recordings.")
                break
    finally:
        try:
            if recording_enabled:
                cam.disable_recording()
        finally:
            try:
                cam.close()
            finally:
                stop_rosbag(rosbag_process)
                rosbag_process = None

    print(f"\nSaved SVO: {opt.output_svo_file}")
    print(f"Saved ROS 2 bag: {bag_output}")
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--output_svo_file',
    type=str,
    default=DEFAULT_OUTPUT,
    help='Path to the output SVO file'
    )
    opt = parser.parse_args()
    if not opt.output_svo_file.endswith((".svo", ".svo2")):
        print("--output_svo_file parameter should be a .svo file but is not : ",opt.output_svo_file,"Exit program.")
        sys.exit(2)
    sys.exit(main(opt))


