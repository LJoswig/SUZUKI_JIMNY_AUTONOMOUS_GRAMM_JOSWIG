# ROS 2 workspace for Jimny! <img src="../docs/logos/spots_logo.png" alt="Spots Logo">

This folder contains all ROS nodes and files to run on the Jimny. 
Nodes are contained in Git submodule repositories to ensure easy code sharing between the Jimny and Rover projects.

## Overview
- [Installation and Setup](#installation-and-setup)
  - [Pulling changes](#pulling-changes-from-submodules)
  - [Pushing changes](#pushing-changes-to-submodules)
  - [Pre-requisites](#pre-requisites)
  - [Dependencies](#dependencies)
  - [setup](#dependencies)
- [Package Structure](#package-structure)
  - [Nodes](#nodes)
  - [Custom Messages and Services](#custom-messages-and-services)
- [Usage](#usage)
  - [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Installation and Setup
Make sure to clone the repository including all submodules.
```
git clone --recurse-submodules git@gitlab.com:tut-robotics/spots/agv/mpc_driving_controller.git
```

After cloning the repository, run the following command to initialize and fetch all submodules:
```
git submodule update --init --recursive
```
This command will download all the required submodules and ensure they are correctly integrated into the project.

### Pulling Changes from Submodules
To pull changes for the main repository and all submodules, use:
```
git pull --recurse-submodules
```

If you only want to update a specific submodule, navigate to the submodule's directory and run:
```
git pull origin <branch-name>
```

### Pushing Changes to Submodules
If you modify a submodule and want to push those changes, you must commit the changes inside the submodule first. Navigate to the submodule directory and commit:
```
cd src/<submodule-name>
git add .
git commit -m "Description of changes"
git push origin <branch-name>
```

Then, return to the main repository and commit the submodule reference update:

```
cd ../..
git add src/<submodule-name>
git commit -m "Updated submodule reference for <submodule-name>"
git push origin master
```

### Pre-requisites
Jetson running Ubuntu 22.04 with Jetpack 6.0. ROS 2 Humble and the ZED SDK/drivers.

### Dependencies
Get dependencies from rosdep:
```
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
```

Get python dependencies from:
```
pip install -r jimny_ws/requirements.txt
```

See [troubleshooting](#troubleshooting) for potential solutions to common errors

### Setup
Source build and source ROS 2 environment as usual:
```
cd jimny_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Package Structure

### nodes
- `cuberos`
Bridge between the CubePilot and ROS 2 using Mavlink messages from the pymavlink interface.
- `data_capture`
Record datapoints while driving to create a big dataset.
- `mode_switching`
Switch between different operational modes
- `mpc_driving_controller`
Package implementing MPC controller for autonomous driving
- `to_vehicle`
Sends the Ackermann_msg to the vehicle and receives data back from the vehicle.
Communication is either via serial/UART (for the Rover) or direct CAN bus (for the Jimny).

### Custom Messages and Services
- `custom_msg` contains all required custom messages to communicate between ROS nodes.
  - `mode` Message to convey the heartbeat data of the CubePilot
  - `RC_in` Message to send all RC channels from the herelink controller via the CubePilot to other nodes.


## Usage

TODO

### Configuration

TODO

## Troubleshooting

Good luck