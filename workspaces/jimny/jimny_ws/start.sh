WORKSPACE_PATH=$(pwd)
export ROS_WORKSPACE_PATH="$WORKSPACE_PATH"

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
