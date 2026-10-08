# MPC Driving Controller

### Overview

The MPC Driving Controller package provides a ROS 2 Humble implementation of Model Predictive Control (MPC) system for autonomous vehicle path planning and control. 
The MPC functions with provided GPS waypoint data that include latitude, longitude, heading angle, and optionally velocity. 


### Table of Contents

[Installation and Setup](#installation-and-setup)

[Package Structure](#package-structure)

[Nodes](#nodes)

[Custom Messages and Services](#custom-messages-and-services)

[Usage](#usage)

[Configuration](#configuration)

[Troubleshooting](#troubleshooting)


### Installation and Setup

##### Cloning

Navigate to a workspace (or make one) and clone the package:

```
cd colcon_ws/src
git clone git@gitlab.com:tut-robotics/spots/agv/mpc_driving_controller.git
```

Alternatively, import as submodule:

```
cd colcon_ws/src
git submodule add git@gitlab.com:tut-robotics/spots/agv/mpc_driving_controller.git
git submodule update --init --recursive
```

##### Dependencies

Get dependencies from rosdep:

```
cd colcon_ws
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
```

Get python dependencies from:

```
pip install -r /colcon_ws/src/mpc_driving_controller/requirements.txt
```

See [troubleshooting](#troubleshooting) for potential solutions to common errors

##### Setup

Update all shebang lines in scripts/ using a provided bash script. Make sure you are using your desired python interpreter (virtual environment or not)
```
cd colcon_ws
chmod +x src/mpc_driving_controller/utils/update_py_env.sh
./src/mpc_driving_controller/utils/update_py_env.sh
```

Source build and source as usual:
```
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### Package Structure

- `data/waypoints/`
waypoint data in CSV files, see data.txt for structure info
- `launch`
launch files
- `scripts`
Python code (not nodes). Includes the controller and trajectory generator
- `msg`
Custom message definitions, see [custom messages](#custom-messages-and-services)
- `scripts`
Python files defining ROS2 nodes
- `srv`
Custom service definitions, see [custom services](#custom-messages-and-services)
- `utils`
Useful scripts, including shebang line updater and basic waypoint file generator


### Nodes

This package contains the following five nodes:

1. controller_node

Node for the MPC controller that generates control commands.
Implements an asynchronous service client to get waypoints for the vehicle to follow.

Subscribes to:

- `/state_est`: Current vehicle state

Publishes:

- `/ackermann_cmds`: Control commands for the vehicle

- `/mpc_path`: MPC path for plotting

Service clients:

- `/get_waypoints`: Waypoint generation service


2. state_pub_node

Node publishing the vehicle state from GPS, IMU and steering angle data.

Subscribes to:

- `/gps/fix`: GPS fix data

- `/gps/vel`: GPS velocity data

- `/imu/data`: IMU data

- `/vehicle/steering`: Steering angle data

Publishes:

- `/state_est`: Vehicle state data


3. traj_plotter_node

Node plotting the global GPS trajectory and the vehicle's path tracking behavior.

Subscribes to:

- `/state_est`: Vehicle state data

- `/mpc_path`: MPC planned path

- `/traj_ref`: Reference trajectory data


4. vehicle_simulator_node

Node simulating the vehicle dynamics. The simulator uses a linear tire model, more precise than the MPC's. 

Subscribes to:

- `/ackermann_cmds`: Ackermann drive commands

Publishes:

- `/state_est`: Vehicle state data


5. waypoint_generator_node

Node providing a service to generate waypoints for the vehicle to follow, from a GPS trajectory. 
The trajectory already takes into account the vehicle's target velocity, path cruvature and MPC horizon.

Publishes:

- `/traj_ref`: Reference trajectory data

Services provided:

- `/get_waypoints`: Service to generate waypoints for the vehicle to follow
    


### Custom Messages and Services

This package includes custom messages and one custom service. See the corresponding files in mpc_driving_controller/msg and mpc_driving_controller/srv for more information. 

##### Messages:
- `mpc_driving_controller/msg/MpcPath.msg`
Predicted path from the MPC for plotting (and debugging)
- `mpc_driving_controller/msg/RefTraj.msg`
Reference trajectory for plotting (and debugging)
- `mpc_driving_controller/msg/StateEst.msg`
State information (position, heading, velocity, etc.)

##### Services:
- `mpc_driving_controller/srv/GenerateWaypoints.src`
Waypoint service for passing reference trajectory from current state of the vehicle


### Usage

Launch the Package. Make sure you update the desired waypoint CSV file and starting coordinates in the chosen launch file. Read more about this in [Configuration](#configuration)
For simulation:
```
ros2 launch mpc_driving_controller simulate_controller.launch.py
```
For real usage:
```
ros2 launch mpc_driving_controller controller.launch.py
```

Running Individual Nodes
Each node can also be run individually. For example, to run waypoint_generator_node:

```
ros2 run mpc_driving_controller waypoint_generator_node.py
```

### Configuration

All parameters can be changed from the used launch files (see [Usage](#usage)). These launch files additionally provide a description for each individual parameter. Below are the most important:

##### Parameter Definitions

- `waypoint_csv_path`
complete path to waypoint CSV file

- `TRACK_VELOCITY`
Boolean parameter to select whether velocity should be tracked. Note that this requires having velocity data in your waypoint CSV file

- `TRACK_SCALE`
if `TRACK_VELOCITY` is set to `True`, `TRACK_SCALE` will define the scale to which the reference velocity will be accelerated. E.g. `TRACK_SCALE=0.5` will lead to a velocity 2x slower than the recorded one, and `TRACK_SCALE=2`, 2x faster

- init_params 
latitude and longitude of the starting coordinates of the trajectory

mpc_params:
Config parameters for the MPC. These include the following:
- `f`
Frequency of the controller
- `N`
Prediction horizon
- `DT`
Time step
- `v_ref`
Reference velocity, if `TRACK_VELOCITY` is set to `False`


### Troubleshooting

##### Setup Issues
If you encounter numpy/scipy issues, try and update your scipy version as follows:
```
pip install scipy --update
```
If possible, try and use the versions specified in requirements.txt

##### Solver issues
If you find that the solver is behaving strangely, it could be that it is running out of time.
For this, you can lower `f` or lower `N`, which should ease the computational load.
Alternatively, you can try and change the `'max_cpu_time'` parameter at the end of the `__init__` in `mpc_driving_controller/mpc_driving_controller_py/kinematic_cartesian_mpc.py` to a higher number (should be 0.4s by  default)
