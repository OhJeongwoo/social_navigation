import rospy
import rospkg
import numpy as np
import json
import random
import time

from social_navigation.msg import Status, Command
from social_navigation.srv import Step, State, Jackal, Reset, StepResponse, StateResponse, JackalResponse, ResetResponse
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan

from utils import *

class GazeboMaster:
    def __init__(self, gazebo_ns='/gazebo'):
        # parater for file
        self.traj_file_ = "traj.json"
        self.spawn_file_ = "spawn.json"

        # parameter for time
        self.time_ = 0.0
        self.target_time_ = -1.0
        self.last_published_time_ = 0.0
        self.pub_interval_ = 0.1

        # parameter for gazebo
        self.is_pause_ = False
        self.reset_ = True
        self.dt_ = 0.1
        self.spawn_threshold_ = 3.0
        self.lookahead_time_ = 1.0
        self.actor_prob_ = 1.0

        # parameter for jackal
        self.accel_ = 0.0
        self.omega_ = 0.0
        with open(self.spawn_file_, 'r') as jf:
            self.spawn_ = json.load(jf)

        # parameter for lidar
        self.scan_size_ = 720
        self.scan_dim_ = 10

        # parameter for actor
        self.n_actor_ = 4
        self.actor_name_ = []
        self.status_ = {}
        self.status_time_ = {}
        self.reset_pose_ = {}
        self.pose_ = {}
        self.goal_ = {}
        self.traj_idx_ = {}
        self.pub_ = {}
        self.sub_status_ = {}
        self.sub_pose_ = {}

        # paramter for trajectory
        with open(self.traj_file_, 'r') as jf:
            self.traj_ = json.load(jf)
        self.n_traj_ = len(self.traj_)

        for seq in range(self.n_actor_):
            name = 'actor_'+str(seq).zfill(3)
            self.actor_name_.append(name)
            self.status_[name] = WAIT
            self.status_time_[name] = self.time_
            self.traj_idx_[name] = -1
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=10)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)
            self.sub_pose_[name] = rospy.Subscriber('/' + name + '/pose', PoseStamped, self.callback_pose)

        self.server_step_ = rospy.Service('step', Step, self.service_step)
        self.server_jackal_ = rospy.Service('jackal', Jackal, self.service_jackal)
        self.server_state_ = rospy.Service('state', State, self.service_state)
        self.server_reset_ = rospy.Service('reset', Reset, self.service_reset)
        self.client_pause_ = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.client_unpause_ = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_model_ = rospy.ServiceProxy(gazebo_ns + '/set_model_state', SetModelState)
        self.sub_clock_ = rospy.Subscriber('/clock', Clock, self.callback_clock)
        # self.sub_scan_ = rospy.Subscriber('/front/scan', LaserScan, self.callback_scan)
        
        self.pub_jackal_ = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)

        time.sleep(1.0)
        self.loop()
    
    
    def callback_status(self, msg):
        name = msg.name
        if self.status_[name] != msg.status:
            print(msg)
            self.status_[name] = msg.status
            self.status_time_[name] = self.time_


    def callback_pose(self, msg):
        name = msg.header.frame_id
        self.pose_[name] = msg.pose.position


    def callback_clock(self, msg):
        self.time_ = msg.clock.secs + msg.clock.nsecs * 1e-9
        if not self.reset_ and self.time_ > self.target_time_ :
            self.client_pause_(Empty())

    
    def callback_scan(self, msg):
        ranges = np.nan_to_num(np.array(msg.ranges), copy=True, posinf=30.0, neginf=-30.0)
        lidar_state = []
        for i in range(self.scan_dim_):
            lidar_state.append(np.mean(ranges[int(i*self.scan_size_/self.scan_dim_):int((i+1)*self.scan_size_/self.scan_dim_)]))
        self.lidar_state_ = lidar_state
        


    def service_step(self, req):
        rt = StepResponse()
        self.target_time_ += self.time_ + self.dt_
        self.is_pause_ = False
        self.client_unpause_(Empty())
        while not self.is_pause_:
            continue
        rt.success = True
        return rt


    def service_jackal(self, req):
        rt = JackalResponse()
        self.accel_ = req.accel
        self.omega_ = req.omega

        rt.success = True
        return rt


    def service_state(self, req):
        rt = StateResponse()
        state = [1]
        rt.state = state
        rt.success = True
        return rt

    
    def service_reset(self, req):
        self.reset_ = True
        # check valid starting point
        candidates = []
        for pos in self.spawn_:
            for name in self.actor_name_:
                if self.status_[name] != MOVE:
                    continue
                if get_length(pos['spawn'], [self.pose_[name].x, self.pose_[name].y]) < self.spawn_threshold_:
                    check = False
                    break
            if check:
                candidates.append(pos)

        # randomly choice
        candidate = random.choice(candidates)
        self.jackal_goal_ = candidate['goal']

        # unpause gazebo
        self.is_pause_ = False
        self.client_unpause_(Empty())

        # replace jackal
        self.replace_jackal(candidate['spawn'])

        # pause gazebo
        self.client_pause_(Empty())
        self.reset_ = False
        self.target_time_ = self.time_

        rt = ResetResponse()
        rt.success = True
        return rt
    

    def replace_jackal(self, pose):
        req = SetModelStateRequest()
        req.model_state.model_name = 'jackal'
        req.model_state.pose = Pose(position=Point(pose[0],pose[1],1.0), orientation=y2q(random.uniform(0.0,2*np.pi)))
        try:
            res = self.set_model_(req)
            if not res.success:
                print("error")
                rospy.logwarn(res.status_message)
        except:
            pass

    def get_goal(self, name):
        traj = self.traj_[self.traj_idx_[name]]
        time = min(self.time_ - self.status_time_[name] + self.lookahead_time_, traj['time'] - EPS)
        interval = traj['interval']
        k = int(time // interval)
        A = traj['waypoints'][k]
        B = traj['waypoints'][k+1]
        alpha = (time - interval * k) / interval
        goal = interpolate(A,B,alpha)
        return Point(goal[0], goal[1], 2.0)

    def loop(self):
        while True:
            if self.time_ - self.last_published_time_ < self.pub_interval_:
                continue
            print("start loop")
            # control pedestrian
            for name in self.actor_name_:
                if self.status_[name] == MOVE:
                    traj_num = self.traj_idx_[name]
                    print(self.time_ - self.status_time_[name])
                    print(self.traj_[traj_num]['interval'])
                    print(self.traj_[traj_num]['time'])
                    if self.time_ - self.status_time_[name] > self.traj_[traj_num]['time']:
                        self.traj_idx_[name] = -1
                        self.status_[name] = WAIT
                        rt = Command()
                        rt.name = name
                        rt.status = WAIT
                        self.pub_[name].publish(rt)
                        continue
                    self.goal_[name] = self.get_goal(name)
                    rt = Command()
                    rt.name = name
                    rt.status = MOVE
                    rt.goal = Pose(position=self.goal_[name])
                    rt.velocity = L2dist(self.pose_[name], self.goal_[name]) / self.lookahead_time_
                    print(rt)
                    self.pub_[name].publish(rt)
                
                elif self.status_[name] == WAIT:
                    p = random.uniform(0.0, 1.0)
                    if p > self.actor_prob_:
                        continue
                    
                    # select possible trajectory
                    while True:
                        traj_num = random.randint(0, self.n_traj_-1)
                        loop_out = True
                        for name2 in self.actor_name_:
                            if self.traj_idx_[name2] == traj_num:
                                loop_out = False
                                break
                        if loop_out:
                            break
                    self.traj_idx_[name] = traj_num
                    print(self.traj_[traj_num])

                    # INIT actor
                    rt = Command()
                    rt.name = name
                    rt.status = INIT
                    rt.goal = Pose(position=Point(self.traj_[traj_num]['waypoints'][0][0], self.traj_[traj_num]['waypoints'][0][1], 0.0))
                    print(rt)
                    self.pub_[name].publish(rt)
            # control jackal
            cmd = Twist()
            cmd.linear.x = self.accel_
            cmd.angular.z = self.omega_
            self.pub_jackal_.publish(cmd)

            self.last_published_time_ = self.time_



if __name__ == "__main__":
    rospy.init_node("Gazebo Master")
    gazebo_master = GazeboMaster()
    rospy.spin()