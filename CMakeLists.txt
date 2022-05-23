cmake_minimum_required(VERSION 3.0.2)
project(social_navigation)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  rosbag
  std_msgs
  ackermann_msgs
  message_generation
  message_filters
  gazebo_ros
  geometry_msgs
  nav_msgs
  sensor_msgs
  cv_bridge
  tf
  tf2
)
find_package(gazebo 11 REQUIRED)
#find_package(ignition-msgs1 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3)
find_package(PCL 1.8 REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  Status.msg
  Command.msg
  InstanceImage.msg
  InstanceArray.msg
  PedestrianArray.msg
  StateInfo.msg
  RRT.msg
  RRTresponse.msg
  Trajectory.msg
  TrajectoryArray.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  Step.srv
  Reset.srv
  State.srv
  Jackal.srv
  TrajectoryPredict.srv
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime
  gazebo_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  DEPENDS Eigen
	opencv_highgui opencv_imgproc
)



include_directories(include ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS} ${SDFormat_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

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

add_executable(depth2pc src/depth2pc.cpp ${HEADERS} ${SOURCES})
target_link_libraries(depth2pc ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(depth2pc ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(rrt src/rrt.cpp ${HEADERS} ${SOURCES})
target_link_libraries(rrt ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(rrt ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  scripts/actor_scheduler.py
  scripts/ynet/ynet.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

