#!/usr/bin/env python3
"""
analyze_follower_bag.py

Evaluate the ROS shadow follower (centerline_follower node) from recorded bags.

Reads /centerline_follower/debug and, when present, /centerline/path, /ackermann_cmd,
/steering_feedback and /feedback_data from one or more bags (e.g. a live run recorded with
record_run.py, or the node output plus the replayed desk-test bag). Produces the same
comparison with the human steering as shadow_follower.py, plus timing statistics of the
path messages for choosing timeouts.

Desk test (node on a replayed bag vs. the offline tool):
  analyze_follower_bag.py --bag node_out --bag desk_bag \
      --reference-csv road_captures/shadow/<run>_shadow.csv --steering-offset-deg -7

Live shadow drive:
  analyze_follower_bag.py --bag <run>/rosbag --steering-offset-deg -7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import utils.config as cfg  # noqa: E402
from shadow_follower import summarize, value_at, write_outputs  # noqa: E402

DEBUG_TOPIC = "/centerline_follower/debug"


def read_bags(bag_uris):
    import rosbag2_py
    from nav_msgs.msg import Path as PathMessage
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    debug, human_cmd, human_meas, speed, path_times = [], [], [], [], []
    for uri in bag_uris:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("", ""))
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        classes = {name: get_message(type_name) for name, type_name in types.items()
                   if name in (cfg.CENTERLINE_TOPIC_ACKERMANN_CMD, cfg.CENTERLINE_TOPIC_STEERING_FEEDBACK,
                               cfg.CENTERLINE_TOPIC_FEEDBACK_DATA)}
        while reader.has_next():
            topic, payload, stamp_ns = reader.read_next()
            t = stamp_ns * 1e-9
            if topic == DEBUG_TOPIC:
                debug.append(json.loads(deserialize_message(payload, String).data))
            elif topic == cfg.CENTERLINE_TOPIC_PATH:
                path = deserialize_message(payload, PathMessage)
                stamp = path.header.stamp.sec + path.header.stamp.nanosec * 1e-9
                path_times.append((t, stamp, len(path.poses)))
            elif topic == cfg.CENTERLINE_TOPIC_ACKERMANN_CMD:
                human_cmd.append((t, deserialize_message(payload, classes[topic]).steering_angle))
            elif topic == cfg.CENTERLINE_TOPIC_STEERING_FEEDBACK:
                data = deserialize_message(payload, classes[topic]).data
                if len(data) > 0:
                    human_meas.append((t, data[0]))
            elif topic == cfg.CENTERLINE_TOPIC_FEEDBACK_DATA:
                data = deserialize_message(payload, classes[topic]).data
                if len(data) > 1:
                    speed.append((t, data[1]))
    as_array = lambda values: np.array(sorted(values), dtype=np.float64).reshape(-1, 2)  # noqa: E731
    debug.sort(key=lambda row: row["time"])
    return debug, as_array(human_cmd), as_array(human_meas), as_array(speed), sorted(path_times)


def timing_summary(debug, path_times):
    lines = []
    if path_times:
        received = np.array([p[0] for p in path_times])
        stamps = np.array([p[1] for p in path_times])
        gaps = np.diff(received)
        delays = received - stamps
        lines.append(f"path messages: {len(path_times)}, {1 / np.median(gaps):.1f} Hz; gap between paths median "
                     f"{np.median(gaps):.2f} s, p95 {np.percentile(gaps, 95):.2f} s, p99 {np.percentile(gaps, 99):.2f} s, "
                     f"max {gaps.max():.2f} s")
        lines.append(f"perception delay (receive - image stamp): median {np.median(delays):.2f} s, "
                     f"p95 {np.percentile(delays, 95):.2f} s, max {delays.max():.2f} s")
    ages = np.array([row["path_age"] for row in debug if row["path_age"] is not None])
    if len(ages):
        lines.append(f"path age at each control cycle: median {np.median(ages):.2f} s, p95 {np.percentile(ages, 95):.2f} s, "
                     f"p99 {np.percentile(ages, 99):.2f} s, max {ages.max():.2f} s "
                     f"(current limit {cfg.FOLLOWER_MAX_PATH_AGE_SECONDS} s)")
    cycle = np.diff([row["time"] for row in debug])
    if len(cycle):
        lines.append(f"control cycle: median {np.median(cycle) * 1000:.0f} ms, p99 {np.percentile(cycle, 99) * 1000:.0f} ms, "
                     f"max {cycle.max() * 1000:.0f} ms")
    odometry = np.mean([row["odometry_ok"] for row in debug]) if debug else np.nan
    lines.append(f"odometry available in {odometry:.1%} of control cycles")
    return lines


def reference_summary(debug, reference_csv):
    import csv
    with open(reference_csv) as f:
        rows = list(csv.DictReader(f))
    reference_t = np.array([float(r["time"]) for r in rows])
    reference = np.array([float(r["follower_steering_deg"]) for r in rows])
    reference_tracking = np.array([r["tracking"] == "True" for r in rows])
    t = np.array([row["time"] for row in debug])
    node = np.array([row["steering_wheel_deg"] for row in debug])
    node_tracking = np.array([row["tracking"] for row in debug])
    inside = (t >= reference_t[0]) & (t <= reference_t[-1])
    expected = np.interp(t[inside], reference_t, reference)
    expected_tracking = reference_tracking[np.clip(np.searchsorted(reference_t, t[inside]), 0, len(reference_t) - 1)]
    difference = np.abs(node[inside] - expected)
    return [
        f"node vs offline shadow tool ({inside.sum()} control cycles):",
        f"  |steering difference| median {np.median(difference):.2f} deg, p95 {np.percentile(difference, 95):.1f} deg, "
        f"max {difference.max():.1f} deg",
        f"  same tracking state in {np.mean(node_tracking[inside] == expected_tracking):.1%} of cycles",
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ROS shadow follower from bags.")
    parser.add_argument("--bag", action="append", required=True, help="bag to read; repeat for several")
    parser.add_argument("--steering-offset-deg", type=float, default=0.0,
                        help="measured steering-wheel angle when driving straight (about -7 with the current "
                             "encoder); subtracted from the human steering for the comparison only")
    parser.add_argument("--reference-csv", default=None, help="offline shadow_follower.py CSV to compare against")
    parser.add_argument("--out-dir", default=str(REPO / "road_captures" / "shadow"))
    parser.add_argument("--name", default=None, help="output name (default: first bag folder name)")
    args = parser.parse_args()

    debug, human_cmd, human_meas, speed, path_times = read_bags(args.bag)
    if not debug:
        raise RuntimeError(f"No {DEBUG_TOPIC} messages in {args.bag}")
    name = args.name or Path(args.bag[0]).name
    summary = [f"bags: {', '.join(args.bag)}"] + timing_summary(debug, path_times)
    if args.reference_csv:
        summary += [""] + reference_summary(debug, args.reference_csv)

    if len(human_cmd) and len(human_meas) and len(speed):
        rows = [(
            row["time"], row["steering_wheel_deg"], row["speed"], row["tracking"], row["reason"],
            row["path_status"] or "", np.nan if row["path_age"] is None else row["path_age"],
            value_at(human_cmd, row["time"]), value_at(human_meas, row["time"]), value_at(speed, row["time"]),
        ) for row in debug]
        summary += [""] + summarize(rows, args.steering_offset_deg, cfg.FOLLOWER_RATE_HZ)
        paths = write_outputs(rows, summary, Path(args.out_dir), f"{name}_node", args.steering_offset_deg)
        print("\n".join(summary))
        print("\nwrote:", *paths, sep="\n  ")
    else:
        print("\n".join(summary))
        print("\n(no human steering/speed topics in the bags; skipped the comparison with the human)")


if __name__ == "__main__":
    main()
