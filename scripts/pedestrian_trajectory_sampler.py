import numpy as np
import rospy
import time
import random
import json

from social_navigation.msg import RRT
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from gazebo_master import PedSim
from utils import *


class Demo:
    def __init__(self):
        self.pub_ = rospy.Publisher("/rrt", RRT, queue_size=10)
        self.sub_ = rospy.Subscriber("/local_goal", Float64MultiArray, self.callback)

        self.valid = False
        self.waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                           [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
        self.n_waypoints_ = len(self.waypoints_)
        self.n_samples_ = 100
        self.lookahead_ = 8
        self.path = []
        time.sleep(1.0)
        self.loop()

    def callback(self, msg):
        new_path = []
        rt_data = msg.data
        n_point = len(rt_data) // 2
        for i in range(n_point):
            new_path.append([rt_data[2*i], rt_data[2*i+1]])
        self.path = new_path
        time.sleep(0.001)
        self.valid = True

    def loop(self):
        path_storage = []
        for _ in range(self.n_samples_):
            self.valid = False
            rt = RRT()
            while True:
                root_index = random.randint(0, self.n_waypoints_-1)
                goal_index = random.randint(0, self.n_waypoints_-1)
                if root_index != goal_index:
                    break
            rt.root = Point(self.waypoints_[root_index][0], self.waypoints_[root_index][1], 0.0)
            rt.goal = Point(self.waypoints_[goal_index][0], self.waypoints_[goal_index][1], 0.0)
            rt.option = True
            print(rt)
            self.pub_.publish(rt)
            t = time.time()
            while not self.valid:
                if time.time() - t > 1.0:
                    break
                time.sleep(0.001)
            path_storage.append(self.path)
            time.sleep(1.0)
        with open("ped_traj_sample.json", "w") as jf:
            json.dump(path_storage, jf, indent=4)

if __name__=='__main__':
    rospy.init_node("demo")
    ped_sim = Demo()
    rospy.spin()
