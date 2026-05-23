cmake_minimum_required(VERSION 3.0.2)
project(lab2_team1)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/astar_node.py
  scripts/turtlebot_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
