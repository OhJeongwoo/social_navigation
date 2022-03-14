cmake_minimum_required(VERSION 3.0.2)
project(social_navigation)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  ackermann_msgs
  message_generation
  message_filters
  gazebo_ros
  geometry_msgs
  nav_msgs
)
find_package(gazebo 9 REQUIRED)
find_package(ignition-msgs1 REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  Status.msg
  Command.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime
  gazebo_msgs
  std_msgs
  geometry_msgs
  nav_msgs
)



include_directories(include ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS} ${SDFormat_INCLUDE_DIRS})

add_library(HumanPlugin SHARED plugins/HumanPlugin.cc)
target_link_libraries(HumanPlugin ${GAZEBO_LIBRARIES} ${IGNITION-MSGS_LIBRARIES})
add_dependencies(HumanPlugin ${catkin_EXPORTED_TARGETS})

install(TARGETS HumanPlugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/actor_scheduler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)