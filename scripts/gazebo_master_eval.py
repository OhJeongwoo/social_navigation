import rospy
import rospkg
import numpy as np
import json
import random
import time
from PIL import Image, ImageOps

from social_navigation.msg import Status, Command, StateInfo, Request, GlobalPathRequest, GlobalPathResponse, GlobalPlannerRequest, GlobalPlannerResponse
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from zed_interfaces.msg import ObjectsStamped
from zed_interfaces.msg import Object

import matplotlib.pyplot as plt

from social_navigation.msg import RRT, RRTresponse


from utils import *

class PedSim:
    def __init__(self, mode='safeRL', gazebo_ns='/gazebo', args=None):
        # mode description
        self.ped_mode_ = args.ped_mode # on: True, off: False
        self.terminal_cond_ = args.terminal_condition # 'goal' or 'time'
        self.high_level_controller_ = args.high_level_controller # exist: True
        self.low_level_controller_ = args.low_level_controller
        self.social_models = ['cadrl', 'sarl']

        # parater for file
        self.package_path_ = rospkg.RosPack().get_path("social_navigation")
        self.config_path_ = self.package_path_ + "/config/"

        self.traj_file_ = self.config_path_ + "ped_traj_candidate.json" # pedestrian trajectory database
        self.spawn_file_ = self.config_path_ + "general_scenario_easy.json" # jackal root-goal database
        self.density = False
        self.replan = True
        self.rrt = False

        print(self.spawn_file_)
        print(self.density)
        # parameter for time
        self.time_ = 0.0 # current time
        self.target_time_ = -1.0 # for simulation step, if time reach to target time, pause simulation
        self.last_published_time_ = 0.0 # frequently manipulate simulation component
        self.pub_interval_ = 0.02 # publishing time period
        self.episode_id_ = 0

        # parameter for gazebo
        self.is_pause_ = False # whether simulation is paused now
        self.reset_ = False # resete signal
        self.pause_time_ = time.time()
        self.dt_ = 0.1
        self.spawn_threshold_ = 3.0
        self.lookahead_time_ = 1.0
        self.lookahead_count_ = 5
        self.actor_prob_ = 1.0
        self.history_rollout_ = 1
        self.history_queue_ = []
        self.timestep = 0

        # parameter for env
        self.goal_reward_coeff_ = 1.0
        self.control_cost_coeff_ = 1.0
        self.map_cost_coeff_ = 1.0
        self.ped_cost_coeff_ = 1.0
        self.ped_collision_threshold_ = 0.3
        self.map_collision_threshold_ = 0.3
        self.goal_threshold_ = 1.0
        self.action_limit_ = 1.0
        self.mode_ = mode
        
        self.action_weight_ = 0.5

        # parameter for jackal
        self.jackal_pose_ = Pose()
        self.jackal_twist_ = Twist()
        self.accel_ = 0.0
        self.omega_ = 0.0

        with open(self.spawn_file_, 'r') as jf:
            self.spawn_ = json.load(jf)

        self.waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                           [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
        self.n_waypoints_ = len(self.waypoints_)

        self.ped_waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                               [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
        self.n_ped_waypoints_ = len(self.ped_waypoints_)
        

        # parameter for lidar
        self.scan_size_ = 1081
        self.scan_dim_ = 30

        #Lidar sin, cos, angles
        self.lidar_angles = np.linspace(-np.pi/4, 5 * np.pi/4, 1081)
        self.lidar_sin = np.sin(self.lidar_angles)
        self.lidar_cos = np.cos(self.lidar_angles)


        # parameter for actor
        if self.density:
            self.n_actor_ = 30
        else:
            self.n_actor_ = 15
        self.actor_name_ = []
        self.group_id_ = {}
        self.r_pos_ = {}
        self.actor_status_ = {}
        self.reset_pose_ = {}
        self.pose_ = {}
        self.goal_ = {}
        self.pub_ = {}
        self.sub_status_ = {}
        self.sub_pose_ = {}

        # parameter for env
        self.max_goal_dist = 5.0
        

        # paramter for trajectory
        with open(self.traj_file_, 'r') as jf:
            self.traj_ = json.load(jf)


        self.n_traj_ = len(self.traj_)
        self.traj_list_ = [[] for _ in range(self.n_ped_waypoints_)]
        for i in range(self.n_traj_):
            self.traj_list_[self.traj_[i]['start']].append(i)
        
        
        groups = make_group(self.n_actor_)
        self.n_groups_ = len(groups)
        g_ids = []
        for g_id, g in enumerate(groups):
            for i in range(g):
                g_ids.append(g_id)

        r_pos = []
        for g in groups:
            p = set_relative_pose(g)
            r_pos += p
        
        self.status_ = []
        self.status_time_ = []
        self.traj_idx_ = []
        self.waypoint_idx_ = []

        for g in  range(self.n_groups_):
            self.status_.append(WAIT)
            self.status_time_.append(0.0)
            self.traj_idx_.append(-1)
            self.waypoint_idx_.append(-1)

        # set actor parameters
        for seq in range(self.n_actor_):
            name = 'actor_'+str(seq).zfill(3)
            self.actor_name_.append(name)
            self.pose_[name] = Point()
            self.group_id_[name] = g_ids[seq]
            self.r_pos_[name] = r_pos[seq]
            self.actor_status_[name] = WAIT
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=30)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)

        self.last_zed_published_time_ = 0.0
        self.zed_publish_interval_ = 0.1
        self.clock_ = Clock()
        self.flag_ = True
        self.global_path_flag_ = True
        self.scenario_num = 0
        self.estop_ = False
        self.vision_threshold_ = 20.0
        self.lookahead_distance_ = 5.0

        # define ROS communicator
        self.sub_pose_ = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_pose)
        self.set_model_ = rospy.ServiceProxy(gazebo_ns + '/set_model_state', SetModelState)
        self.sub_scan_ = rospy.Subscriber('/front/scan', LaserScan, self.callback_scan)
        self.sub_local_goal_ = rospy.Subscriber('/global_planner/response', GlobalPlannerResponse, self.callback_goal)
        self.sub_global_path_ = rospy.Subscriber('/global_path/response', GlobalPathResponse, self.callback_global_path)
        self.pub_jackal_ = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.pub_zed_ = rospy.Publisher('/objects', ObjectsStamped, queue_size=10)
        self.pub_global_goal_ = rospy.Publisher('/global_goal', Point, queue_size=10)
        self.pub_global_path_ = rospy.Publisher('/global_path/request', GlobalPathRequest, queue_size=10)
        self.pub_request_ = rospy.Publisher('/global_planner/request', GlobalPlannerRequest, queue_size=10)
        self.client_pause_ = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.client_unpause_ = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        time.sleep(1.0)
        self.sub_clock_ = rospy.Subscriber('/clock', Clock, self.callback_clock)
        time.sleep(1.0)


    def callback_status(self, msg):
        name = msg.name
        self.actor_status_[name] = msg.status


    def callback_pose(self, msg):
        rt = ObjectsStamped()
        if self.time_ - self.last_zed_published_time_ > self.zed_publish_interval_:
            self.last_zed_published_time_ = self.time_
            rt.header.stamp = self.clock_
            objects = []
            candidates = []
            for name in self.actor_name_:
                try:
                    idx = msg.name.index(name)
                    pos = msg.pose[idx].position
                    d = L2dist(pos, self.jackal_pose_.position)
                    if d > self.vision_threshold_:
                        continue
                    candidates.append({'name':name, 'd':d})
                except:
                    print("no name")
            candidates = sorted(candidates, key=lambda x: x['d'])
            candidates = candidates[:min(len(candidates), 10)]
            for cand in candidates:
                name = cand['name']
                try:
                    idx = msg.name.index(name)
                    pos = msg.pose[idx].position
                    obj = Object()
                    obj.position = [pos.x, pos.y, pos.z]
                    obj.label_id = 0
                    obj.tracking_state = 1
                    objects.append(obj)
                except:
                    print("no name")
            rt.objects = objects
            self.pub_zed_.publish(rt)
            # for name in  self.actor_name_:
            #     try:
            #         idx = msg.name.index(name)
            #         pos = msg.pose[idx].position
            #         if L2dist(pos, self.jackal_pose_.position) < self.vision_threshold:
            #             candidates.append([])
            #         obj = Object()
            #         obj.position = [pos.x, pos.y, pos.z]
            #         obj.label_id = 0
            #         obj.tracking_state = 1
            #         objects.append(obj)
            #         rt.objects = objects
            #     except:
            #         print("no name")
            # self.pub_zed_.publish(rt)
        name_list = msg.name
        for name in self.actor_name_:
            try:
                idx = name_list.index(name)
                self.pose_[name] = msg.pose[idx].position
            except:
                print("no name")
        try:
            idx = name_list.index('jackal')
            self.jackal_pose_ = msg.pose[idx]
            self.jackal_twist_ = msg.twist[idx]
        except:
            print("no jackal")


    def callback_clock(self, msg):
        self.clock_ = msg.clock
        self.time_ = msg.clock.secs + msg.clock.nsecs * 1e-9
        self.pause_time_ = time.time()
        self.loop()
        if not self.reset_ and self.time_ > self.target_time_ :
            self.is_pause_ = True
            self.client_pause_()
        
    
    def callback_scan(self, msg):
        ranges = np.nan_to_num(np.array(msg.ranges), copy=True, posinf=self.max_goal_dist, neginf=-0.0)
        ranges = np.clip(ranges, 0.0, self.max_goal_dist)
        self.lidar_state_raw = ranges
        
        lidar_state = []
        for i in range(self.scan_dim_):
            lidar_state.append(np.mean(ranges[int(i*self.scan_size_/self.scan_dim_):int((i+1)*self.scan_size_/self.scan_dim_)]))
        self.lidar_state_ = lidar_state


    def callback_goal(self, msg):
        self.estop_ = msg.estop
        self.local_goal_ = [msg.local_goal.x, msg.local_goal.y]
        if get_length(self.jackal_goal_, self.local_goal_) < 3.0:
            self.local_goal_ = self.jackal_goal_
        if self.scenario_num == msg.id:
            rt = GlobalPlannerRequest()
            rt.id = msg.id
            rt.seq = msg.seq + 1
            self.pub_request_.publish(rt)


    def callback_global_path(self, msg):
        if msg.type == 1:
            return
        self.global_path_ = msg.points
        self.global_path_flag_ = True
        self.global_path_length_ = msg.distance


    def get_obs(self):
        state = []
        L = len(self.history_queue_)
        for i in range(L):
            state.append(self.history_queue_[L-1-i])
        for _ in range(self.history_rollout_ - len(self.history_queue_)):
            state.append(self.history_queue_[0])

        return state


    def get_dim(self):
        o = self.reset()
        return o.shape[0], 2


    def get_random_action(self):
        return 2 * self.action_limit_ * (np.random.rand(2) - 0.5)


    def get_goal(self, g):
        x = 0.0
        y = 0.0
        ng = 0
        for name in self.actor_name_:
            if self.group_id_[name] == g:
                pos = self.pose_[name]
                rpos = self.r_pos_[name]
                x += pos.x - rpos.x
                y += pos.y - rpos.y
                ng += 1
        x = x / ng
        y = y / ng
        traj = self.traj_[self.traj_idx_[g]]
        N = len(traj['waypoints'])
        d = 1000.0
        idx = -1
        for i in range(N):
            nd = get_length([x,y], traj['waypoints'][i])
            if nd < d:
                d = nd
                idx = i
        if idx == -1:
            idx = N-1
        idx = min(idx+self.lookahead_count_, N-1)
        goal = traj['waypoints'][idx]
        return Point(goal[0], goal[1], 2.0)
        # traj = self.traj_[self.traj_idx_[g]]
        # time = min(self.time_ - self.status_time_[g] + self.lookahead_time_, traj['time'] - EPS)
        # interval = traj['interval']
        # k = int(time // interval)
        # A = traj['waypoints'][k]
        # B = traj['waypoints'][k+1]
        # alpha = (time - interval * k) / interval
        # goal = interpolate(A,B,alpha)
        # return Point(goal[0], goal[1], 2.0)


    def check_reach(self, g):
        traj = self.traj_[self.traj_idx_[g]]
        goal = traj['waypoints'][-1]
        for name in self.actor_name_:
            if self.group_id_[name] == g:
                pos = self.pose_[name]
                rpos = self.r_pos_[name]
                goal_pose = Point(rpos.x + goal[0], rpos.y + goal[1], 0.0)
                d = L2dist(goal_pose, pos)
                if d > 1.0:
                    return False
        return True

    def obs2list(self, obs):
        state = []
        for i in range(len(obs)):
            state.append(obs[i].lin_vel)
            state.append(obs[i].ang_vel)
            state.append(obs[i].accel)
            state.append(obs[i].goal.x)
            state.append(obs[i].goal.y)
            state.append(obs[i].goal_distance)
            state += obs[i].lidar
            
        return np.array(state)


    def jackal_cmd(self, a):
        self.accel_ = a[0]
        self.omega_ = a[1]
        return


    def replace_jackal(self, pose):
        req = SetModelStateRequest()
        req.model_state.model_name = 'jackal'
        yaw = random.uniform(0.0, 2 *np.pi)
        req.model_state.pose = Pose(position=Point(pose[0],pose[1],1.0), orientation=y2q(yaw))
        try:
            res = self.set_model_(req)
            if not res.success:
                print("error")
                rospy.logwarn(res.status_message)
            else:         
                self.jackal_pose_.position.x = pose[0]
                self.jackal_pose_.position.y = pose[1]
                self.jackal_pose.orientation = y2q(yaw)
        except:
            pass


    def set_local_goal(self):
        print("set local goal")
        try:
            jx = self.jackal_pose_.position.x
            jy = self.jackal_pose_.position.y
            N = len(self.global_path_)
            idx = -1
            for i in range(N):
                if get_length([jx,jy], self.global_path_[i]) < self.lookahead_distance_:
                    idx = i
                    break
            if idx == -1:
                local_goal = self.jackal_goal_
            else:
                local_goal = self.global_path_[idx][0:2]
        except:
            local_goal = self.jackal_goal_    
        self.local_goal_ = local_goal


    def update_state(self):
        state = StateInfo()
        jx = self.jackal_pose_.position.x
        jy = self.jackal_pose_.position.y
        q = self.jackal_pose_.orientation
        qx = 1 - 2 * (q.y ** 2 + q.z ** 2)
        qy = 2 * (q.w * q.z + q.x * q.y)
        ct = qx / (qx ** 2 + qy ** 2) ** 0.5
        st = qy / (qx ** 2 + qy ** 2) ** 0.5
        
        # goal state
        # gx, gy = transform_coordinate(self.jackal_goal_[0] - jx, self.jackal_goal_[1] - jy, ct, st)
        gx, gy = transform_coordinate(self.local_goal_[0] - jx, self.local_goal_[1] - jy, ct, st)
        state.goal_distance = (gx**2 + gy**2)**0.5
        if state.goal_distance < 0.01:
            dir_gx = 0.0
            dir_gy = 0.0
        else:
            dir_gx = gx / state.goal_distance
            dir_gy = gy / state.goal_distance
        state.goal = Point(dir_gx, dir_gy, 0)

        # self.goal_distance = ((self.jackal_goal_[0] - jx)**2 + (self.jackal_goal_[1]-jy)**2)**0.5
        self.goal_distance = ((self.local_goal_[0] - jx)**2 + (self.local_goal_[1]-jy)**2)**0.5

        state.goal_distance = state.goal_distance if state.goal_distance < self.max_goal_dist else self.max_goal_dist

        # jackal state
        state.lin_vel = self.jackal_twist_.linear.x ** 2 + self.jackal_twist_.linear.y ** 2
        state.ang_vel = self.jackal_twist_.angular.z
        state.accel = 0.2 * (state.lin_vel - self.prev_vel) / self.dt_
        self.prev_vel = state.lin_vel

        # lidar state
        state.lidar = (1-np.array(self.lidar_state_) / self.max_goal_dist).tolist()
        
        self.history_queue_.append(state)
        if len(self.history_queue_) > self.history_rollout_:
            self.history_queue_.pop(0)

    
    def simulation(self):
        self.target_time_ = self.time_ + self.dt_
        self.is_pause_ = False
        self.client_unpause_()
        while True:
            if self.time_ < self.target_time_:
                if time.time() - self.pause_time_ > 0.1:
                    try:
                        self.client_unpause_()
                    except:
                        pass
            else:
                break
        self.update_state()
        return


    def reset(self):
        self.reset_ = True
        self.scenario_num += 1
        self.tot_moving_dist = 0
        self.travel_dist = 0.0
        self.moving_dist = [0.0]

        #Initialize prev values
        self.prev_vel = 0
        self.prev_ang = 0
        self.prev_acc = 0
        self.prev_acc_ang = 0
        self.timestep = 0
        self.recent_action = [0,0]

        # check valid starting point
        candidates = []
        while True:
            candidates = []
            for pos in self.spawn_:
                check = True
                for name in self.actor_name_:
                    if get_length(pos['spawn'], [self.pose_[name].x, self.pose_[name].y]) < self.spawn_threshold_:
                        check = False
                        break
                if check:
                    candidates.append(pos)
            if len(candidates) > 0:
                break
            else:
                print("empty candidate")
                self.is_pause_ = False
                self.client_unpause_()
                time.sleep(10.0)
                self.client_pause_()
        
        # randomly choice
        candidate = random.choice(candidates)
        # candidate['spawn'] = [0.0, -1.0]
        # candidate['goal'] = [-10.0, -5.0]
        self.jackal_goal_ = candidate['goal']
        self.local_goal_ = self.jackal_goal_
        if not self.replan and self.rrt:
            self.global_path_ = candidate['rrt_path']
        else:
            self.global_path_ = candidate['carrt_path']

        # publish global goal
        self.pub_global_goal_.publish(Point(self.jackal_goal_[0], self.jackal_goal_[1], 0.0))
        time.sleep(0.1)
        # rt = Request()
        # rt.jackal = self.jackal_pose_.position
        # rt.goal = Point(self.jackal_goal_[0], self.jackal_goal_[1], 0.0)
        # rt.reset = False
        # self.pub_request_.publish(rt)

        # unpause gazebo
        self.is_pause_ = False
        self.client_unpause_()
        time.sleep(0.1)

        # replace jackal
        self.replace_jackal(candidate['spawn'])
        self.prev_jackal = candidate['spawn']
        self.epi_path_len_ = get_length(candidate['spawn'], candidate['goal'])
        time.sleep(0.1)

        self.jackal_cmd([0,0])
        
        # For stable state initilization
        global_planner_request = GlobalPlannerRequest()
        global_planner_request.id = self.scenario_num
        global_planner_request.seq = 0
        self.pub_request_.publish(global_planner_request)
        self.client_pause_()
        self.reset_ = False
        self.target_time_ = self.time_
        self.history_queue_ = []
        self.update_state()
        self.prev_goal_distance = self.goal_distance

        return self.obs2list(self.get_obs())
    

    def step(self, a):
        '''
        driving score
        1) collision
        minimum distance = d
        score = min(1.0, max(0.0, (0.4 - d) * 5.0))

        total score = mean(score in episode) * 100

        2) speed
        v = travel_dist / travel_step / dt
        score = max(0.0, min(1.0, v - 0.5)) * 100
        
        3) jerk
        abs(d^2v/dt^2) > 10.0 -> 0.5
        abs(d^2w/dt^2) > 10.0 -> 0.5
        '''
        self.pub_global_goal_.publish(Point(self.jackal_goal_[0], self.jackal_goal_[1], 0.0))
        self.timestep += 1
        s = self.get_obs()
        # action smoothing

        if self.low_level_controller_ in self.social_models:
            a = [self.recent_action[0] * self.action_weight_ + np.clip(a[0], 0.0, 1.0) * (1-self.action_weight_), self.recent_action[1] * self.action_weight_ + (1-self.action_weight_) * np.clip(a[1], -1.0, 1.0)]
        else :
            a = [np.clip(self.recent_action[0] + np.clip(a[0], -1.0, 1.0) * 1.5 / 10, 0, 1.5), self.recent_action[1]  * self.action_weight_ + (1- self.action_weight_) * np.clip(a[1], -1.0, 1.0) *1.5]
        if self.estop_:
            a = [0.0, 0.3]
        # a = [0.0, 0.1]
        self.recent_action = a
        self.jackal_cmd(a)
        self.simulation()
        ns = self.get_obs()

        success = False
        is_dangerous = False
        done = False
        collision_cost = 0.0
        travel_time = self.timestep * self.dt_
        jerk_cost = 0.0
        step_type = 0
        # 0: keep going, 1: goal reach, 2: collision, 3: timeout

        jx = self.jackal_pose_.position.x
        jy = self.jackal_pose_.position.y
        self.tot_moving_dist += get_length([jx,jy],self.prev_jackal)
        self.moving_dist.append(self.tot_moving_dist)
        self.prev_jackal = [jx, jy]
        if ((self.jackal_goal_[0] - jx)**2 + (self.jackal_goal_[1]-jy)**2)**0.5 < self.goal_threshold_:
            done = True
            success = True
            step_type = 1
            print("goal reached!")

        d = min(self.lidar_state_)

        # Lidar cost  
        if d < self.map_collision_threshold_:
            done = True
            success = False
            step_type = 2
            print('lidar collision')

        if not done and self.timestep >= 1000:
            done = True
            success = False
            step_type = 3
            print('timeout!')

        if self.timestep > 100:
            if self.moving_dist[self.timestep] - self.moving_dist[self.timestep-100] < 1.0:
                done = True
                success = False
                step_type = 4
                print('stuck!')
        
        if d < 1.5:
            is_dangerous = True
        if d < 1.0:
            collision_cost = 1.0
        if d < 0.5:
            collision_cost = 2.0

        # cur_travel_length = 0.0
        # min_dist = 1000
        # for p in self.global_path_:
        #     if get_length([jx,jy], p) < min_dist:
        #         min_dist = get_length([jx,jy], p)
        #         cur_travel_length = p[2]
        # self.travel_dist = max(self.travel_dist, cur_travel_length)
        self.travel_dist = self.epi_path_len_ - get_length(self.jackal_goal_, [jx, jy])
        
        info = {'success': success,
                'is_dangerous': is_dangerous, 
                'jackal': [jx, jy],
                'd': d,
                'collision_cost': collision_cost, 
                'moving_dist': self.tot_moving_dist,
                'travel_dist': self.travel_dist, 
                'travel_time': travel_time,
                'step_type': step_type}
        
        return self.obs2list(ns), None, done, info


    def loop(self):
        # wait for publish interval
        if self.time_ - self.last_published_time_ < self.pub_interval_:
            return
        
        if not self.replan:
            self.set_local_goal()

        # control pedestrian
        peds = {}
        goals = []
        for name in self.actor_name_:
            ped = {'group': self.group_id_[name], 'pos': self.pose_[name], 'rpos': self.r_pos_[name]}
            peds[name] = ped        
        for g in range(self.n_groups_):
            if self.status_[g] == MOVE:
                traj_num = self.traj_idx_[g]
                goal = self.get_goal(g)
                if self.check_reach(g):
                # if self.time_ - self.status_time_[g] > self.traj_[traj_num]['time']:
                    goal = None
                    self.status_[g] = WAIT
                    for name in self.actor_name_:
                        if self.group_id_[name] == g:
                            self.actor_status_[name] = WAIT
                        continue
            elif self.status_[g] == WAIT:
                if self.waypoint_idx_[g] == -1:
                    traj_num = random.randint(0, self.n_traj_-1)
                else:
                    traj_num = random.choice(self.traj_list_[self.waypoint_idx_[g]])
                self.waypoint_idx_[g] = self.traj_[traj_num]['end']
                self.traj_idx_[g] = traj_num
                self.status_[g] = INIT
                goal = None
            elif self.status_[g] == INIT:
                check = True
                traj_num = self.traj_idx_[g]
                for name in self.actor_name_:
                    if self.group_id_[name] != g:
                        continue
                    if self.actor_status_[name] != MOVE:
                        check = False
                        break
                if check:
                    self.status_[g] = MOVE
                    self.status_time_[g] = self.time_
                goal = Point(self.traj_[traj_num]['waypoints'][0][0], self.traj_[traj_num]['waypoints'][0][1], 0.0)
            goals.append(goal)

        jackal = self.jackal_pose_.position
        sub_goals, sub_vel = pedestrian_controller(peds, goals, jackal)
        for name in self.actor_name_:
            g = self.group_id_[name]
            if self.status_[g] == MOVE:
                if not name in sub_goals.keys():
                    continue
                self.actor_status_[name] = MOVE
                rt = Command()
                rt.name = name
                rt.status = MOVE
                rt.goal = Pose(position = sub_goals[name])
                rt.velocity  = sub_vel[name]
                self.pub_[name].publish(rt)
            elif self.status_[g] == INIT:
                traj_num = self.traj_idx_[g]
                pq = Point(self.traj_[traj_num]['waypoints'][0][0] + self.r_pos_[name].x, self.traj_[traj_num]['waypoints'][0][1] + self.r_pos_[name].y, 0.0)
                if L2dist(pq, jackal) < 1.0:
                    continue
                goal = Point(self.traj_[traj_num]['waypoints'][0][0] + self.r_pos_[name].x, self.traj_[traj_num]['waypoints'][0][1] + self.r_pos_[name].y, 0.0)
                if L2dist(goal, self.pose_[name]) < 5.0:
                    rt = Command()
                    rt.name = name
                    rt.status = MOVE
                    rt.goal = Pose(position = goal)
                    rt.velocity  = 1.5
                    self.pub_[name].publish(rt)
                else:
                    rt = Command()
                    rt.name = name
                    rt.status = INIT
                    rt.goal = Pose(position=goal)
                    self.pub_[name].publish(rt)


        # control jackal
        cmd = Twist()
        cmd.linear.x = self.accel_
        cmd.angular.z = self.omega_
        self.pub_jackal_.publish(cmd)

        self.last_published_time_ = self.time_

