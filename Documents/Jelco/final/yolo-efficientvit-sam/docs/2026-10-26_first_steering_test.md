# First steering test: checklist (26 Oct 2026, dusk)

Follower mode = **ACRO**: the centerline follower steers (clamped to ±300°), the **throttle
stick sets the speed as a dead-man** (centred / released = stop, full forward = 1.39 m/s).
Any problem latches a stop: full brake via the safety flag, steering held. **Only disarming
clears it** (same as the Herelink e-stop). MANUAL works exactly as before.

Go only if every step passes. Stop and go back to MANUAL at the first surprise.

## 0. Setup (once)
- `cd ~/workspaces/jimny/jimny_ws && colcon build && source install/setup.bash`
- `config.py` (yolo-efficientvit-sam): `INPUT_MODE = "live"`, `CENTERLINE_MOTION_SOURCE = "live"`,
  `RECORD_RUN_ENABLED = True`, `ZED_DEPTH_MODE = "NONE"`, `CENTERLINE_DETECTION_SELECTION = "best"`.
- Same `export ROS_DOMAIN_ID=<n>` in every terminal.

## 1. Start (car standing, not armed, MANUAL)
1. `ros2 launch to_vehicle do_jimny.launch.py` (MPC is off; mode_switch logs `ACRO source: follower`).
2. Steering node: `python3 ackermann_to_steering_serial_final.py /dev/ttyCH341USB0`.
3. `source /opt/ros/humble/setup.bash && .venv_yolo_efficientvit_sam/bin/python main.py`
   (preview on port 8765; prints the run folder at the end).
4. `ros2 launch centerline_follower shadow_follower.launch.py`
   (logs `steering limit 300.0 deg`).
5. `ros2 topic hz /centerline/path` ≈ 4–8 Hz, `ros2 topic hz /centerline_follower/cmd` = 20 Hz.

## 2. Shadow drive at dusk (MANUAL)
- 2–3 runs on the planned test road, ~5 km/h, in the middle of the road. Stop `main.py`
  after each run (Ctrl+C) so the run is closed; start it again for the next one.
- Evaluate in the car:
  ```bash
  .venv_yolo_efficientvit_sam/bin/python tools/analyze_follower_bag.py \
      --bag /home/user/recordings/<run>/rosbag --name <run> --steering-offset-deg -7
  ```
  and look at `road_captures/shadow/<run>_node_shadow.png`.
- **Go criteria** (on the stretch you want to drive):
  - follower tracking while moving ≥ 95 %, longest stretch without tracking < 1 s
  - same turn direction in curves ≥ 95 % (if there are curves)
  - no excursions in the plot that you would not have steered yourself
  - in the preview, the green line stays on the road
- Not met → no steering test today; the recordings are still valuable.

## 3. Standstill tests (armed, car stopped on flat ground, nobody near the wheels)
| Test | Do | Expected |
|---|---|---|
| Waiting | ACRO with the camera covered | no movement; `/signal_error` key 2 "Follower mode waiting" |
| Engage | ACRO, road visible, stick centred | key 1; front wheels turn towards the follower command (watch the preview text) |
| Dead-man | stick stays centred | car does not move |
| Herelink e-stop | press the button | latched, brake; release + disarm clears it |
| Follower crash | in ACRO, stop the follower node (Ctrl+C) | latched within 0.3 s "Follower command timeout"; disarm, restart node |
| Road lost | in ACRO, cover the camera | latched within ~1–3 s "Follower is not tracking the road" |
| Back to MANUAL | ACRO → MANUAL | note how long until MANUAL takes effect (heartbeat, up to ~1 s) |

Watch `ros2 topic echo /signal_error` during the tests.

## 4. First metres (only if 2 and 3 passed)
- Straight, empty road; a spotter next to the road; RC in hand, thumb ready on the e-stop.
- Car stopped, steering roughly straight, ACRO, stick centred → engaged.
- Push the stick gently (walking speed first); full forward = 5 km/h maximum.
- **Release the stick at the first unexpected steering movement.** E-stop if it does not stop.
- 20–30 m, stop, disarm, look at the recording; only then the next run.
- Abort the session if: a latched stop you cannot explain, steering towards the road edge,
  oscillating steering, or the follower losing the road repeatedly.

## Known limits (from the replays)
- In bright afternoon light the detector misses the road 30–70 % of the time and sometimes
  detects grass as road; the follower then mostly refuses (stops). Test at dusk only.
- Path checks reject paths more than 1.5 m to the side or jumping > 1 m at 6 m ahead, so
  curves tighter than ~12 m radius will stop the car.
- The steering limit is ±300° (`max_steering_wheel_deg` in `config/shadow_follower.yaml` and
  `follower_max_steering_deg` in mode_switch); the follower aims for the road centre.
