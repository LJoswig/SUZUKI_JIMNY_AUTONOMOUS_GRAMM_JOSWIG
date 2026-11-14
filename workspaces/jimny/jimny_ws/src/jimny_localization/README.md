# jimny_localization

GPS/IMU localization for the Jimny following ROS REP-105, built on `robot_localization`.
It is the first step towards global navigation: knowing where the car is on a street map, so that
a route layer can later decide which way to take at junctions. Lateral control stays with the
camera-based `centerline_follower`; this package does **not** steer the car.

The map side (OSM road graph, RViz view) lives in [`jimny_map`](../jimny_map).

**Status (2026-11-14):** runs on the car; the first live test showed the pipeline works end to
end, and showed that the Cube's compass heading is not usable yet (see [Known issues](#known-issues)).
No full test drive has been evaluated yet.

## Architecture

```
            map ──(ekf_global)──> odom ──(ekf_local)──> base_link ──(static)──> gps_link, imu_link
```

| Frame | Meaning | Continuous? |
|---|---|---|
| `map` | Earth-fixed, x = east, y = north. Origin = where navsat_transform initialised (car's start position, so it differs every run). | No, jumps when GPS corrects |
| `odom` | Dead-reckoned. Starts at 0 with yaw 0 = the car's initial direction. | Yes, never jumps, drifts slowly |
| `base_link` | Centre of the rear axle (same reference as `centerline_follower`). x forward, y left, z up. | |

Two EKFs because one filter cannot be smooth and globally correct at the same time. Local
consumers (the follower, a future local costmap) use `odom`. Global consumers (map matching,
routing) use `map`.

### Nodes

| Node | Inputs | Outputs |
|---|---|---|
| `speed_twist` (this package) | `/gps_raw` | `/speed/twist` |
| `ekf_local` (robot_localization) | `/speed/twist` (vx, vy = 0), `/imu/data` (yaw rate) | `/odometry/local`, TF `odom → base_link` |
| `ekf_global` (robot_localization) | as above + `/imu/data` absolute yaw + `/odometry/gps` | `/odometry/global`, TF `map → odom` |
| `navsat_transform` (robot_localization) | `/gps/fix`, `/imu/data`, `/odometry/global` | `/odometry/gps`, `/gps/filtered`, `fromLL` service |
| static TFs | launch arguments | `base_link → gps_link`, `base_link → imu_link` |

Configuration: [`config/ekf.yaml`](config/ekf.yaml), [`config/navsat.yaml`](config/navsat.yaml),
[`launch/localization.launch.py`](launch/localization.launch.py).

### Inputs from cuberos

`cuberos_node.py` publishes two topics in ROS conventions (REP-103) for this package. The old
topics (`/imu_attitude` in NED/FRD, `/gps_raw` as `GPSFix`) are unchanged, because
`centerline_follower` and `state_pub_node` rely on them.

| Topic | Type | Content |
|---|---|---|
| `/gps/fix` | `sensor_msgs/NavSatFix` | From MAVLink `GPS_RAW_INT`. Covariance = receiver `h_acc²`/`v_acc²`, fallback DOP × 3 m. Only new fixes are published. |
| `/imu/data` | `sensor_msgs/Imu` | ArduPilot `ATTITUDE` + `SCALED_IMU3`, converted from NED/FRD to ENU/FLU: yaw 0 = east, counter-clockwise positive; turning left = positive yaw rate. Only new samples are published. |

cuberos requests `ATTITUDE`/`SCALED_IMU3` at 20 Hz and `GPS_RAW_INT` at 10 Hz from the Cube
(`cuberos_py/mavlink.py`). Without that request the Cube sends at its default stream rates (~4–6 Hz),
and the EKF output drops to that rate.

### Where the speed comes from

There is no wheel speed sensor in ROS. `speed_twist` uses the GPS **Doppler** ground speed from
`/gps_raw` (accuracy ~0.1 m/s, independent of the GPS position error). `/feedback_data[1]`,
which the follower uses, is the same value after a round trip
`to_vehicle → sterfBoard → CAN 0x121 → axelBrake → CAN 0x12 → sterfBoard → serial`. Reading
`/gps_raw` directly removes that delay and the dependency on `/ackermann_cmd` being published.
Speeds below 0.1 m/s are published as 0 so the EKF does not creep while parked.

## Setup

```bash
sudo apt install ros-humble-robot-localization ros-humble-geographic-msgs ros-humble-diagnostic-updater
colcon build --packages-select cuberos jimny_localization jimny_map to_vehicle
```
`diagnostic_updater` was missing on the Jetson even with robot_localization installed; without it
the EKF and navsat nodes die at startup with exit code 127. Check with
`ldd /opt/ros/humble/lib/robot_localization/ekf_node | grep "not found"`.

### Sensor mounting

Launch arguments, relative to `base_link` (rear axle centre), in metres:

| Argument | Value | Note |
|---|---|---|
| `gps_x`, `gps_y`, `gps_z` | 0.93, 0.0, 1.35 | measured 2026-11-14 (default) |
| `imu_x`, `imu_y`, `imu_z`, `imu_yaw` | 0 | position barely matters (only the yaw rate is used). `imu_yaw` stays 0 as long as ArduPilot's `AHRS_ORIENTATION` matches the Cube mounting. |

## Running

On the Jetson (one terminal each, `source install/setup.bash` first):
```bash
ros2 run cuberos cuberos_node.py
ros2 launch jimny_localization localization.launch.py
ros2 launch jimny_map map_view.launch.py rviz:=false     # map + trails for RViz
```
Or together with the vehicle stack: `ros2 launch to_vehicle do_jimny.launch.py use_localization:=true`.

The map appears once `map_visualizer` logs `published map`, which needs a GPS fix and the first
output of `ekf_global`.

### RViz on a laptop

The laptop needs no workspace, only RViz of the same ROS distro as the Jetson (Humble).
With Docker, from the repo root:
```bash
xhost +local:docker
docker run -it --rm --net=host --device /dev/dri -e DISPLAY=$DISPLAY -e ROS_DOMAIN_ID=0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $PWD/workspaces/jimny/jimny_ws/src/jimny_map/rviz:/rviz \
  osrf/ros:humble-desktop rviz2 -d /rviz/map.rviz
```
- Set `ROS_DOMAIN_ID` explicitly to the Jetson's value (empty = 0). Passing an unset variable
  (`-e ROS_DOMAIN_ID=$ROS_DOMAIN_ID`) breaks the `ros2` CLI.
- Both machines must be on the same network. Phone hotspots often block the multicast DDS discovery
  needs; check with `docker run ... ros2 topic list`.
- `libGL`/`MESA` errors without `--device /dev/dri` only mean software rendering.

### Quick sign check (car standing, then rolling)

| Topic | Expected |
|---|---|
| `/gps/fix` | `status ≥ 0`, position inside the map area |
| `/imu/data` orientation | facing east: z ≈ 0, w ≈ 1 · north: z ≈ w ≈ 0.71 (yaw = 2·atan2(z, w)) |
| `/imu/data` `angular_velocity.z` | ≈ 0 standing, positive when turning left |
| `/speed/twist` | 0.0 standing, positive while rolling |

## Recording a test drive

Record only the raw inputs. Localization, map view and all their parameters can then be re-run
on the bag (`use_sim_time:=true`, `ros2 bag play --clock`) without driving again.
```bash
ros2 bag record -o testfahrt_$(date +%Y%m%d_%H%M%S) \
  /gps_raw /gps/fix /imu/data /imu_attitude /feedback_data \
  /centerline/path /centerline/status /rc_inputs /heartbeat
```
Do not record `/tf` or use `-a`: transforms from a live localization run would clash with the
replayed ones. Optionally add the compressed camera topic (useful later for junction detection).

A useful drive contains: 1 min standing at start and end, a closed loop, one road driven in both
directions (separates GPS error from map offset), straight stretches above ~2 m/s (heading check),
several junctions, stop-and-go, and the junction ids passed (from RViz) written down.

## Known issues

1. **Cube heading (compass) is not usable yet.** First test: facing east the Cube reported 78°
   instead of 0°, facing north 98° instead of 90°, so a 90° turn showed up as only 20°. That looks
   like hard-iron distortion from the car body. (Whether the car was really turned between both
   readings was not double-checked.) The yaw *rate* is not affected. Consequences: navsat_transform
   aligns GPS with the map using this heading, so the map appears rotated against the GPS track,
   and `ekf_global` is pulled towards a wrong heading.
   Options: (a) ArduPilot yaw from GPS instead of the compass (e.g. `EK3_SRC1_YAW = 8`, GSF);
   (b) compass calibration in the car (ArduPilot "Large Vehicle MagCal"); (c) initialise the
   heading from GPS course over ground in a small node. Until then, set the yaw entry of `imu0` in
   `ekf_global` to `false`.
2. **GPS covariance probably too optimistic.** The receiver reports σ ≈ 0.58 m, while ~2.5 m was
   expected. If the real error is larger, the EKF trusts GPS too much (visible as `map → odom`
   jumps). Compare the GPS trail against the roads on a recorded drive; if needed, add a
   covariance scale factor in cuberos.
3. **Speed has no sign.** GPS speed is a magnitude, so reversing is integrated as driving forward.
   A real wheel/driveshaft speed (ABS sensors via the car CAN, OBD-II, or a Hall sensor) would fix
   this and also help where GPS is shadowed.
4. **Yaw rate is integrated while standing.** `PlanarOdometry` skips yaw at standstill; the EKF
   does not. With a gyro bias this slowly rotates `odom` while parked. Check the standstill yaw
   rate; if it has a constant offset, gate the IMU input at standstill.
5. **map frame origin changes every run** (navsat initialises at the start position). Fine for
   viewing; a route layer must work in lat/lon or convert via `fromLL` each run.
6. **OSM accuracy.** The test area is mostly `track`; OSM geometry there is often off by 1–3 m,
   about the size of the GPS error. Four junction pairs are less than 15 m apart (edges 33, 40, 41,
   46 in `test_area.geojson`) and cannot be told apart by GPS alone.

## Possible next steps

In the order they build on each other:

1. Evaluate a recorded drive: EKF vs. `PlanarOdometry` drift, GPS cross-track error against OSM
   (per direction of travel), `map → odom` jump size, Cube heading vs. GPS course over ground.
2. Fix the heading (known issue 1).
3. Map matching on the road graph: GPS + odometry + "junction seen" events → current edge and
   distance to the next junction. This is where the 2.5 m GPS becomes good enough for junction
   decisions.
4. Junction handling in perception: the camera pipeline must offer more than one branch at a
   junction; the route layer picks one by heading. Expected to be the hardest part.
5. If accuracy is still insufficient: RTK with correction data (NTRIP, e.g. SAPOS, or a base
   station) gives ~2 cm absolute. Moving-baseline RTK (second antenna) only gives heading.

A full Nav2 stack (global costmap + planner + controller) is not recommended with this GPS: a
metric path off by half a road would fight the vision follower. Nav2's Route Server (graph-based,
from Kilted on) fits the topological approach better than the costmap stack. A local costmap
from the road mask in `odom` is possible later for obstacles or junction manoeuvres.
