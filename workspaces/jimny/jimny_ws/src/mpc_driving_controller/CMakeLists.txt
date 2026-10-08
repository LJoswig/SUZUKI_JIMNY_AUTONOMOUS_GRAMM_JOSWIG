cmake_minimum_required(VERSION 3.8)
project(mpc_driving_controller)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)  # For C++
find_package(rclpy REQUIRED)    # For Python
find_package(rosidl_default_generators REQUIRED)
find_package(gps_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StateEst.msg"
  "msg/MpcPath.msg"
  "msg/RefTraj.msg"
  "srv/GenerateWaypoints.srv"
  DEPENDENCIES std_msgs  # Add other dependencies if needed
)

# Install Python scripts
install(PROGRAMS
  scripts/waypoint_generator_node.py
  scripts/controller_node.py
  scripts/vehicle_simulator_node.py
  scripts/traj_plotter_node.py
  scripts/state_pub_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install package (this should be the last installation call)
ament_python_install_package(${PROJECT_NAME}_py)
ament_package()
