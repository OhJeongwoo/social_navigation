#!/usr/bin/python2
import time
import argparse
import os
import sys
import yaml
import json
import rospkg
import rospy
import random

from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Pose, Point, PoseStamped, Twist
from social_navigation.msg import Status, Command

from utils import *

L = 30
V = 3.0
T = 8 * L / V
_init_pos = [Point(L,L,0), Point(-L,L,0), Point(-L,-L,0), Point(L,-L,0)]



class ActorScheduler:
    def __init__(self, N):
        self.time_ = 0.0
        self.last_pub_time_ = 0.0
        self.pub_interval_ = 0.1
        self.n_actor_ = N
        self.actor_name_ = []
        self.status_ = {}
        self.status_time_ = {}
        self.offset_ = {}
        self.reset_pose_ = {}
        self.pose_ = {}
        self.goal_ = {}
        self.pub_ = {}
        self.sub_status_ = {}
        self.sub_pose_ = {}
        self.threshold_ = 1.0


        for seq in range(self.n_actor_):
            name = 'actor_'+str(seq).zfill(3)
            self.actor_name_.append(name)
            self.status_[name] = WAIT
            self.status_time_[name] = self.time_
            self.offset_[name] = seq * T / 4
            self.reset_pose_[name] = _init_pos[seq]
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=10)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)
            self.sub_pose_[name] = rospy.Subscriber('/' + name + '/pose', PoseStamped, self.callback_pose)

        self.pub_jackal_ = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.sub_clock_ = rospy.Subscriber('/clock', Clock, self.callback_clock)
        
        time.sleep(1.0)
        for seq in range(self.n_actor_):
            name = self.actor_name_[seq]
            rt = Command()
            rt.name = name
            rt.status = INIT
            rt.goal = Pose(position=_init_pos[seq])
            self.pub_[name].publish(rt)
        
        self.loop()

    def callback_clock(self, msg):
        self.time_ = msg.clock.secs + msg.clock.nsecs * 1e-9

    def callback_status(self, msg):
        name = msg.name
        if self.status_[name] != msg.status:
            self.status_[name] = msg.status
            self.status_time_[name] = self.time_

    def callback_pose(self, msg):
        name = msg.header.frame_id
        self.pose_[name] = msg.pose.position

    def loop(self):
        while True:
            if self.time_ - self.last_pub_time_ < self.pub_interval_:
                continue

            for name in self.actor_name_:
                rt = Command()
                if self.status_[name] == WAIT:
                    rt.name = name
                    rt.status = INIT
                    rt.goal = Pose(position=self.reset_pose_[name])
                    self.pub_[name].publish(rt)
                    x = random.random() * 10.0 - 0.5 * self.reset_pose_[name].x
                    y = random.random() * 10.0 - 0.5 * self.reset_pose_[name].y
                    self.goal_[name] = Point(x,y,0)
                elif self.status_[name] == MOVE:
                    if L2dist(self.goal_[name], self.pose_[name]) < self.threshold_:
                        rt.status = WAIT
                        self.pub_[name].publish(rt)
                    else:
                        rt.status = MOVE
                        rt.goal = Pose(position=self.goal_[name])
                        rt.velocity = V
                        self.pub_[name].publish(rt)
                
            rt = Twist()
            rt.linear.x = 1.0
            rt.angular.z = 0.1
            self.pub_jackal_.publish(rt)

            self.last_pub_time_ = self.time_



if __name__=="__main__":
    rospy.init_node("actor_scheduler")
    N = 4

    actor_scheduler = ActorScheduler(N)
    rospy.spin()