#!/usr/bin/python
import time
import argparse
import os
import sys
import yaml
import json
import rospkg
import rospy

from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Pose, Point
from social_navigation.msg import Status, Command

from utils import *

L = 15
V = 2.0
T = 8 * L / V
_init_pos = [Point(L,L,0), Point(-L,L,0), Point(-L,-L,0), Point(L,-L,0)]
def f(t):
    t = t - T * (t // T)
    if t < T / 4:
        return Point(L - V * t, L, 0)
    t -= T / 4
    if t < T / 4:
        return Point(-L, L - V * t, 0)
    t -= T / 4
    if t < T / 4:
        return Point(-L + V * t, -L, 0)
    t -= T /4
    return Point(L, -L + V * t, 0)

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
        self.pub_ = {}
        self.sub_status_ = {}

        for seq in range(self.n_actor_):
            name = 'actor_'+str(seq).zfill(3)
            self.actor_name_.append(name)
            self.status_[name] = WAIT
            self.status_time_[name] = self.time_
            self.offset_[name] = seq * T / 4
            self.reset_pose_[name] = _init_pos[seq]
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=10)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)

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

    def loop(self):
        while True:
            if self.time_ - self.last_pub_time_ < self.pub_interval_:
                continue
            
            for name in self.actor_name_:
                if self.status_[name] != MOVE:
                    continue
                rt = Command()
                rt.status = MOVE
                rt.goal = Pose(position=f(self.time_ - self.status_time_[name] + self.offset_[name] + 2.0))
                rt.velocity = V
                self.pub_[name].publish(rt)
                


            self.last_pub_time_ = self.time_



if __name__=="__main__":
    rospy.init_node("actor_scheduler")
    # parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
    # parser.add_argument('--yaml', default='test', type=str)
    # args = parser.parse_args()

    # rospack = rospkg.RosPack()
    # PKG_PATH = rospack.get_path("social_navigation")
    # CFG_PATH = PKG_PATH + '/config/'
    # YML_PATH = PKG_PATH + '/yaml/'    
    # YML_FILE = YML_PATH + args.yaml + '.yaml'
    # # set yaml path
    # with open(YML_FILE) as file:
    #     args = yaml.load(file, Loader=yaml.FullLoader)
    #     print(args)
    # ACT_FILE = CFG_PATH + args['act_file_name'] + '.json'
    
    # with open(ACT_FILE) as file:
    #     act_dict = json.load(file)
    # actor_info_list = transform_actor_info(act_dict)
    
    N = 4

    actor_scheduler = ActorScheduler(N)
    rospy.spin()