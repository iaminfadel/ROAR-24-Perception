cmake_minimum_required(VERSION 3.1.3)
project(pick_and_place_simulation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_grasp_plugin
)

catkin_package()

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
