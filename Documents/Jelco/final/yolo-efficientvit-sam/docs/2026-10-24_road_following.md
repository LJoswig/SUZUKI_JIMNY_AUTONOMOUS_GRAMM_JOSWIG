# Road following: changes of 24 Oct 2026 and the test plan for the car

Branch `centerline-anchor-kalman` (home repository), on top of `main` at `eee8d0c8`.
Goal: the Jimny follows the camera centerline at ~5 km/h. So far everything runs in
**shadow mode**: the follower computes commands on its own topic and nothing on the car
listens to it.

---

## 1. What changed

### 1.1 Centerline Kalman filter (`utils/centerline.py`), commit `1efb57b3`
- State = centerline x at three fixed image rows (47.5 / 60 / 72.5 % of the height), not
  quadratic coefficients. The coefficient state was badly conditioned: at the live rate of
  ~5 Hz half of all measurements were rejected and the filter reset 55 times on 181107.
- Measurement noise from each fit's geometry, random-walk process model, re-lock after
  3 agreeing gate rejections.
- Motion model reduced to one yaw term; **the Kalman filter now runs `visual_only`**,
  because yaw rate barely predicts image motion (the road stays ahead in a curve).
- Result on 181107: gate rejections 49 % → 3 %, resets 55 → 3, time without output 9 % → 2 %.

### 1.2 Path memory and path fusion (`utils/path_memory.py`), commits `bc025da1`, `6dba28fa`
- The centerline is projected onto the ground (flat ground, camera 1.78 m high, pitched
  8.5° down, 1.27 m ahead of the rear axle) and kept in an **odometry frame**.
- Odometry = measured speed (`/feedback_data` data[1], m/s) + IMU yaw rate
  (`/imu_attitude` angular_velocity.z, **sign flipped**: the cube reports + = right turn).
  Validated against GPS: distance 489 vs 491 m, heading change 1° median over 5 s.
- **Path memory**: when the filter loses the road, the stored path is replayed for up to
  3 s / 8 m (cyan in the preview). Offline: 8–12 cm median error at 4.5 m.
- **Path fusion**: each new observation is blended 50/50 into the stored path; a path that
  differs by > 0.8 m is a glitch unless 4 consecutive observations agree (then it replaces
  the stored one). On 181107: p99 frame-to-frame path jump 0.69 → 0.20 m, correlation with
  the human steering 0.45 → 0.59, about 0.1 s more lag.
- ZED visual odometry was tried and dropped: at the pipeline's ~4 Hz grab rate its
  position does not move.

### 1.3 Road follower (`utils/follower.py`), commit `d8b772e8`
- Pure pursuit: lookahead point 6 m ahead of the rear axle, curvature κ = 2y/L²,
  road-wheel angle atan(2.27 m · κ), steering wheel = −20 · angle (positive = right,
  like `/ackermann_cmd`). Clamp ±620°, rate limit 120°/s, 1.39 m/s.
- Delay compensation: the path is stored at the odometry pose of its image time and
  evaluated at the current pose.
- Cannot track (no path, stale path > 1 s, odometry restart, path too short): speed 0,
  steering held. The emergency decision is left to `mode_switch` (not implemented yet).
- Fitted from the bags: positive `steering_angle` = right turn, effective steering ratio
  ~20 (nominal 14), straight ahead at about −7° (encoder offset, **not** compensated in code).

### 1.4 Visualisation (`main.py`), commits `d8b772e8`, `6dba28fa`
- Preview overlay (`SHOW_FOLLOWER_OVERLAY`): yellow dot = lookahead point, magenta arc =
  path the steering would drive (up to the dot), text = steering command.
- Centerline colour: green observed, orange coasting, cyan remembered, red lost.

### 1.5 ROS interface, commits `6b463354`, `64f0f6f0`
- `main.py` publishes `/centerline/path` (`nav_msgs/Path`, `base_link` at image time,
  `header.stamp` = image time, empty when lost) and `/centerline/status`
  (`observed` / `coasting` / `remembered` / `lost`). Switch: `CENTERLINE_ROS_PUBLISH_ENABLED`.
- New package `jimny_ws/src/centerline_follower` (node `follower`, launch
  `shadow_follower.launch.py`, parameters `config/shadow_follower.yaml`): subscribes to
  path, status, IMU and feedback, publishes `/centerline_follower/cmd_shadow`
  (`AckermannDrive`) and `/centerline_follower/debug` (JSON) at 20 Hz.
- `pure_pursuit.py` / `odometry.py` in the package are **copies** of the perception code;
  keep them in sync (the desk test shows drift).

### 1.6 Tools, commits `d8b772e8`, `6dba28fa`, `efbfaa8f`, and this commit
- `tools/shadow_follower.py`: offline shadow test on an SVO + bag (perception cached),
  compares with the human steering, `--steering-offset-deg` for the comparison only,
  `--fusion-weight`, `--export-bag` for the desk test. Output in `road_captures/shadow/`.
- `tools/analyze_follower_bag.py`: evaluates the ROS node from bags (desk test or live
  drive), including timing statistics for choosing timeouts.
- `tools/record_run.py`: records a matching rosbag next to the SVO.
- `main.py` (`RECORD_RUN_ENABLED`): during a live run, records the SVO (only the processed
  frames, ~3–5 fps) and the matching rosbag into `RECORD_RUN_DIR/run_<date>_<time>/`.
- `do_jimny.launch.py`: the GPS waypoint MPC controller starts only with `use_mpc:=true`.

### 1.7 Current results (offline, testfahrt 20261021)

| | 181107 (233 s) | 181710 (38 s) |
|---|---|---|
| Follower tracking while moving | 99.7–100 % | 100 % |
| Correlation with human command | 0.59 | 0.46 |
| Same turn direction in curves | 94 % (ROS node) | 100 % (8 samples) |
| Mean bias | +32° right | +23° right |

The bias comes from the driven line: the road centre was ~25 cm right of the car on
straights (the follower aims for the centre). Remaining large deviations come from
segmentation errors that persist for many frames (e.g. 205–220 s in 181107, dark frames).

---

## 2. Test plan on the car

### 2.0 Before every session
- Build: `cd ~/workspaces/jimny/jimny_ws && colcon build && source install/setup.bash`.
- Unit tests: `colcon test --packages-select centerline_follower --event-handlers console_direct+`
  (expect 8 passed).
- Set the same unique `export ROS_DOMAIN_ID=<n>` in every terminal (otherwise anyone on the
  network with domain 0 sees, and can publish, the car's topics).
- `config.py` for live runs: `INPUT_MODE = "live"`, `CENTERLINE_MOTION_SOURCE = "live"`,
  `RECORD_RUN_ENABLED = True`.

### 2.1 Desk test (after every change to the follower code)
Checks that the ROS node computes the same as the offline tool.
```bash
cd ~/Documents/Jelco/final/yolo-efficientvit-sam && source /opt/ros/humble/setup.bash
.venv_yolo_efficientvit_sam/bin/python tools/shadow_follower.py --svo <run>.svo2 --bag <run>_rosbag \
    --steering-offset-deg -7 --export-bag /tmp/desk_bag
ros2 launch centerline_follower shadow_follower.launch.py use_sim_time:=true          # terminal 2
ros2 bag record --use-sim-time -o /tmp/node_out /centerline_follower/cmd_shadow /centerline_follower/debug  # terminal 3
ros2 bag play /tmp/desk_bag --clock
.venv_yolo_efficientvit_sam/bin/python tools/analyze_follower_bag.py --bag /tmp/node_out --bag /tmp/desk_bag \
    --reference-csv road_captures/shadow/<run>_shadow.csv --steering-offset-deg -7
```
Pass: node vs offline steering difference median well below 1°, same tracking state > 99 %.

### 2.2 Standstill check on the car (not armed, MANUAL)
Start order:
1. `ros2 launch to_vehicle do_jimny.launch.py` (MPC is off by default).
2. Steering node as usual: `python3 ackermann_to_steering_serial_final.py /dev/ttyCH341USB0`.
3. `main.py` (live, records the run).
4. `ros2 launch centerline_follower shadow_follower.launch.py`.

Checks:

| Check | Command | Expected |
|---|---|---|
| Path rate | `ros2 topic hz /centerline/path` | ~3–5 Hz |
| Command rate | `ros2 topic hz /centerline_follower/cmd_shadow` | 20 Hz |
| Nobody listens to the command | `ros2 topic info /centerline_follower/cmd_shadow` | only the bag recorder subscribed |
| Odometry live (first live test of `LiveRosMotionProvider`) | `ros2 topic echo /centerline_follower/debug` while rolling | `"odometry_ok": true` |
| Perception stops | stop `main.py` | within ~1 s `stale_path`, speed 0 |
| Road not visible | cover the camera | after path memory runs out `no_path`, speed 0 |
| IMU lost | stop `cuberos_node` | `no_odometry` |
| Steering sign | stand in front of a visible curve | preview arc and `debug` steer into the curve (+ = right) |
| Recording | stop `main.py` | `run_<date>_<time>/run.svo2` and `rosbag/` exist |

### 2.3 Live shadow drive
- MANUAL only, about 5 km/h, known road with curves; **never switch to ACRO**.
- Several runs, including the difficult light conditions where segmentation drops out.
- Drive in the middle of the road where possible (the follower aims for the centre).

Evaluate:
```bash
.venv_yolo_efficientvit_sam/bin/python tools/analyze_follower_bag.py --bag <run_dir>/rosbag --steering-offset-deg -7
.venv_yolo_efficientvit_sam/bin/python tools/shadow_follower.py --svo <run_dir>/run.svo2 --bag <run_dir>/rosbag \
    --steering-offset-deg -7      # re-run offline, e.g. with other --lookahead / --fusion-weight
```
Look at:
- **Timing**: path rate, gaps (p99, max), perception delay, path age, odometry availability.
  These set the timeouts for emergency mode (instead of guessing values like 0.3 s).
- **Agreement**: same turn direction in curves (target > 95 %), correlation, deviations
  > 90°, longest stretch without tracking.
- **Plot** `road_captures/shadow/<name>_node_shadow.png` and the preview recording for the
  places where the follower disagrees.

### 2.4 Only after the shadow drives look good (not built yet)
- Safety work in `mode_switch` (see section 3), then a hybrid "dead-man" mode: software
  steers, the RC throttle stick sets the speed and releasing it stops the car.
- Standstill tests of every emergency trigger with the wheels free or the car stopped.
- First autonomous metres: straight road, spotter, RC in hand.

---

## 3. Open points and reminders

**Safety gaps (to tackle when asked):**
- `mode_switch`: `signal_error` is not latched; any new MPC/follower command clears it.
  Must latch until MANUAL.
- Emergency triggers to add: follower lost, follower command timeout, stale perception
  path, stale speed/IMU, steering tracking error. Timeouts from measured live data.
- `to_vehicle` sends the safety flag only every 0.2 s → brake reaction up to ~0.5 s.
- sterfBoard computes `noAckReceived` (500 ms without command) but never acts on it.
- Emergency (full brake via axelBrake, no PID) does not stop the steering board.
- Unexplained steering tracking loss at the end of both 20261021 bags (measured and
  commanded steering > 100° apart).

**Reminders:**
- Two smoothers in series (image Kalman filter + odometry-frame fusion) add lag. Consider
  replacing the image Kalman filter with the fusion once live shadow results allow it.
- Follower/odometry code exists twice (perception repo and `centerline_follower`);
  consolidate later.
- Steering encoder offset (~−7°): fix on the steering side; only the analysis scripts
  take an offset parameter.
- Steering ratio 20 and pitch 8.5° are fitted values; re-check with live data.
- The camera can only be opened by one process: use `main.py` with `RECORD_RUN_ENABLED`,
  not `record_run.py`, while perception runs.
