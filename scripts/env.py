import rospy
import rospkg
import numpy as np

from social_navigation.msg import Status, Command, StateInfo
from social_navigation.srv import Step, State, Jackal, Reset, StepResponse, StateResponse, JackalResponse, StepRequest, StateRequest, JackalRequest, ResetRequest, ResetResponse

from utils import *

class PedSimEnv:
    def __init__(self):
        self.client_step_ = rospy.ServiceProxy('step', Step)
        self.client_state_ = rospy.ServiceProxy('state', State)
        self.client_jackal_ = rospy.ServiceProxy('jackal', Jackal)
        self.client_reset_ = rospy.ServiceProxy('reset', Reset)
        self.dt_ = 0.1
        self.history_rollout_ = 5
        self.goal_reward_coeff_ = 1.0
        self.control_cost_coeff_ = 1.0
        self.map_cost_coeff_ = 1.0
        self.ped_cost_coeff_ = 1.0
        self.ped_collision_threshold_ = 0.1
        self.map_collision_threshold_ = 0.1
        self.goal_threshold_ = 0.1
        


    def step(self, a):
        s = self.get_obs()
        self.jackal_cmd(a)
        req = StepRequest()
        req.request = True
        res = self.client_step_(req)
        ns = self.get_obs()

        reward = 0.0
        done = False

        # goal reward
        g = s[0].goal
        ng = ns[0].goal
        dg = (g.x ** 2 + g.y ** 2) ** 0.5
        dng = (ng.x ** 2 + ng.y ** 2) ** 0.5
        goal_reward = self.goal_reward_coeff_ * (dng - dg) / self.dt_
        if dng < self.goal_threshold_:
            done = True

        # jackal cost
        control_cost = self.control_cost_coeff_ * (abs(ns[0].accel) + abs(ns[0].ang_vel - s[0].ang_vel))

        # map cost
        map_cost = self.map_cost_coeff_ * collision_cost(min(ns[0].lidar))
        if min(ns[0].lidar) < self.map_collision_threshold_:
            done = True
        map_cost =0

        # peds cost
        ped_cost = 0.0
        P = len(ns[0].pedestrians)
        for i in range(P):
            p = ns[0].pedestrians[i]
            d = (p.x ** 2 + p.y ** 2) ** 0.5
            if d < self.ped_collision_threshold_:
                done = True
            ped_cost += collision_cost(d)
        ped_cost = self.ped_cost_coeff_ * ped_cost

        reward = goal_reward
        cost = control_cost + map_cost + ped_cost

        info = {'reward':{'total': reward, 'goal': goal_reward}, 'cost': {'total': cost, 'control':control_cost, 'map': map_cost, 'ped': ped_cost}, 'done': done}

        return ns, reward, done, info

    def reset_model(self):
        req = ResetRequest()
        req.request = True
        print(req)
        res = self.client_reset_(req)
        return self.get_obs()

    def get_obs(self):
        req = StateRequest()
        req.request = True
        res = self.client_state_(req)
        return res.state

    def jackal_cmd(self, a):
        req = JackalRequest()
        req.accel = a[0]
        req.omega = a[1]
        req.request = True
        res = self.client_jackal_(req)
        return res.success