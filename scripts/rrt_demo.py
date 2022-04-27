import numpy as np
import rospy
import time

from social_navigation.msg import RRT
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from gazebo_master import PedSim
from utils import *


class Demo:
    def __init__(self):
        self.ped_sim_ = PedSim()

        self.pub_ = rospy.Publisher("/rrt", RRT, queue_size=10)
        self.sub_ = rospy.Subscriber("/local_goal", Float64MultiArray, self.callback)

        self.valid = False
        self.ped_sim_.reset()
        self.ped_sim_.step([0,1])
        self.global_goal_ = Point(self.ped_sim_.jackal_goal_[0], self.ped_sim_.jackal_goal_[1], 0.0)
        self.local_goal_ = self.global_goal_
        self.loop()

    def callback(self, msg):
        self.local_goal_ = msg
        self.valid = True

    def loop(self):
        step = 0
        while True:
            rt = RRT()
            rt.root = self.ped_sim_.jackal_pose_.position
            rt.goal = self.global_goal_
            if step % 50 == 0:
                rt.option = True
            else:
                rt.option = False
            print(rt)
            self.pub_.publish(rt)
            t = time.time()
            while not self.valid:
                if time.time() - t > 1.0:
                    break
                time.sleep(0.001)
            a = purepursuit(self.ped_sim_.jackal_pose_, self.local_goal_, 1.0)
            a = get_similar_action(a)
            self.ped_sim_.step(a)
            self.valid = False
            step += 1

if __name__=='__main__':
    rospy.init_node("demo")
    ped_sim = Demo()
    rospy.spin()
