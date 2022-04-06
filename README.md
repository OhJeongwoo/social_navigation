# Social-Friendly Safe Delivery Robot Project

## Installation

### 1. ROS installation

please refer http://wiki.ros.org/melodic/Installation/Ubuntu

### 2. Clone repository and build

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/OhJeongwoo/social_navigation.git
cd ..
catkin_make
```

```
echo "source ~/catkin_ws/src/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```


## Pedestrian Detection

### 1. Install realsense2-ros package

please refer https://github.com/IntelRealSense/realsense-ros

```
sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
cd ~/catkin_ws
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
cd ..
catkin_init_workspace
cd ..
catkin_make clean
catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
catkin_make install
```

### 2. Install Detectron2

please refer https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### 3. Launch Pedestrian Detection Algorithm (jeongwoo version)

```
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud align_depth:=true
```

open another terminal
```
roscd social_navigation
cd scripts
python3 pedestrian_detector.py
```

and open another terminal

```
rosrun social_navigation depth2pc
```

## Reinforcement Learning in GAZEBO

Terminal 1
```
roslaunch social_navigation default.launch
```

Terminal 2
```
roscd social_navigation
cd scripts
python2 gazebo_master.py
```

Terminal 3
```
roscd social_navigation
cd scripts
python {YOUR_CODE}.py
```

