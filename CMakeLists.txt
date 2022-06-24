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
# find_package(gazebo 11 REQUIRED)
find_package(gazebo 9 REQUIRED)
#find_package(ignition-msgs1 REQUIRED)
# find_package(OpenCV 4 REQUIRED)
find_package(OpenCV 3 REQUIRED)
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
  Cluster.msg
  ClusterArray.msg
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
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
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


# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
# )

# add_library(${PROJECT_NAME}
#   lib/point.cpp
#   lib/RRT.cpp
# )

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/point.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/pixel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/RRT.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/utils.cpp
)

set(HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/point.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pixel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/RRT.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Transform.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel.h
)

add_executable(depth2pc src/depth2pc.cpp ${HEADERS} ${SOURCES})
target_link_libraries(depth2pc ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(depth2pc ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(rrt src/rrt.cpp ${HEADERS} ${SOURCES})
target_link_libraries(rrt ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(rrt ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(rrt_2 src/rrt_2.cpp ${HEADERS} ${SOURCES})
target_link_libraries(rrt_2 ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(rrt_2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(global_planner src/global_planner.cpp ${HEADERS} ${SOURCES})
target_link_libraries(global_planner ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
add_dependencies(global_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(clustering_module src/clustering_module.cpp ${HEADERS} ${SOURCES})
# target_link_libraries(clustering_module ${catkin_LIBRARIES} ${CV_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${EIGEN_LIBRARIES})
# add_dependencies(clustering_module ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  scripts/actor_scheduler.py
  scripts/ynet/ynet.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

