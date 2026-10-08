cmake_minimum_required(VERSION 3.8)
project(cuberos)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)  # For C++
find_package(rclpy REQUIRED)    # For Python
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/RCIn.msg"
    "msg/Heartbeat.msg"
    DEPENDENCIES std_msgs
)

# Install Python scripts
install(PROGRAMS
  scripts/cuberos_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install package (this should be the last installation call)
ament_python_install_package(${PROJECT_NAME}_py)

ament_package()
