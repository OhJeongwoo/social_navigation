from black import T
import rospy
import rospkg
import numpy as np
import json
import random
import time
from PIL import Image, ImageOps

from social_navigation.msg import Status, Command, StateInfo
from social_navigation.srv import Step, State, Jackal, Reset, StepResponse, StateResponse, JackalResponse, StepRequest, StateRequest, JackalRequest, ResetRequest, ResetResponse
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from utils import *


class PedSim:
    def __init__(self, mode='safeRL', gazebo_ns='/gazebo'):
        # parater for file
        self.traj_file_ = "traj.json"
        self.spawn_file_ = "goal.json"
        self.collision_file_ = rospkg.RosPack().get_path("social_navigation") + "/config/collision_301_1f.png"

        # parameter for time
        self.time_ = 0.0
        self.target_time_ = -1.0
        self.last_published_time_ = 0.0
        self.pub_interval_ = 0.02

        # parameter for gazebo
        self.is_pause_ = False
        self.reset_ = False
        self.step_ = False
        self.pause_time_ = time.time()
        self.dt_ = 0.1
        self.spawn_threshold_ = 3.0
        self.lookahead_time_ = 1.0
        self.actor_prob_ = 1.0
        self.history_rollout_ = 5
        self.history_queue_ = []

        # parameter for env
        self.goal_reward_coeff_ = 1.0
        self.control_cost_coeff_ = 1.0
        self.map_cost_coeff_ = 1.0
        self.ped_cost_coeff_ = 1.0
        self.ped_collision_threshold_ = 0.5
        self.map_collision_threshold_ = 0.5
        self.goal_threshold_ = 0.25
        self.action_limit_ = 1.0
        self.mode_ = mode
        self.collision_map_ = Image.open(self.collision_file_)
        self.img_w_, self.img_h_ = self.collision_map_.size
        self.sy_ = -0.05
        self.sx_ = 0.05
        self.cy_ = 30.0
        self.cx_ = -59.4

        # parameter for jackal
        self.jackal_pose_ = Pose()
        self.jackal_twist_ = Twist()
        self.accel_ = 0.0
        self.omega_ = 0.0
        with open(self.spawn_file_, 'r') as jf:
            self.spawn_ = json.load(jf)

        # parameter for lidar
        self.scan_size_ = 720
        self.scan_dim_ = 20

        # parameter for actor
        self.n_actor_ = 0
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

        # parameter for env
        self.max_goal_dist = 10.0
        

        # paramter for trajectory
        with open(self.traj_file_, 'r') as jf:
            self.traj_ = json.load(jf)


        self.n_traj_ = len(self.traj_)
        self.adj_mat_ = []

        for i in range(self.n_traj_):
            adj_row = []
            for j in range(self.n_traj_):
                check = False
                n = len(self.traj_[i]['waypoints'])
                m = len(self.traj_[j]['waypoints'])
                for seq1 in range(n):
                    for seq2 in range(m):
                        if get_length(self.traj_[i]['waypoints'][seq1], self.traj_[j]['waypoints'][seq2]) < 3.0:
                            check = True
                            break
                    if check:
                        break
                adj_row.append(check)
            self.adj_mat_.append(adj_row)
        

        # set actor parameters
        for seq in range(self.n_actor_):
            name = 'actor_'+str(seq).zfill(3)
            self.actor_name_.append(name)
            self.status_[name] = WAIT
            self.status_time_[name] = self.time_
            self.traj_idx_[name] = -1
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=10)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)
            # self.sub_pose_[name] = rospy.Subscriber('/' + name + '/pose', PoseStamped, self.callback_pose)

        # define ROS communicator
        self.sub_pose_ = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_pose)
        self.pub_jackal_ = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.set_model_ = rospy.ServiceProxy(gazebo_ns + '/set_model_state', SetModelState)
        self.sub_scan_ = rospy.Subscriber('/front/scan', LaserScan, self.callback_scan)
        self.client_pause_ = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.client_unpause_ = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        
        time.sleep(1.0)
    
        self.sub_clock_ = rospy.Subscriber('/clock', Clock, self.callback_clock)

        time.sleep(1.0)


    def callback_status(self, msg):
        name = msg.name
        if self.status_[name] != msg.status:
            self.status_[name] = msg.status
            self.status_time_[name] = self.time_


    def callback_pose(self, msg):
        name_list = msg.name
        for name in self.actor_name_:
            try:
                idx = name_list.index(name)
                self.pose_[name] = msg.pose[idx].position
            except:
                print("no name")
        try:
            idx = name_list.index('jackal')
            #prin(msg.twist[idx])
            self.jackal_pose_ = msg.pose[idx]
            self.jackal_twist_ = msg.twist[idx]
        
        except:
            print("no jackal")


    def callback_clock(self, msg):
        self.time_ = msg.clock.secs + msg.clock.nsecs * 1e-9
        self.pause_time_ = time.time()
        self.loop()
        if not self.reset_ and self.time_ > self.target_time_ :
            self.is_pause_ = True
            self.client_pause_()
        
    
    def callback_scan(self, msg):
        ranges = np.nan_to_num(np.array(msg.ranges), copy=True, posinf=10.0, neginf=-0.0)
        ranges = np.clip(ranges, 0.0, 10.0)
        lidar_state = []

        for i in range(self.scan_dim_):
            lidar_state.append(np.mean(ranges[int(i*self.scan_size_/self.scan_dim_):int((i+1)*self.scan_size_/self.scan_dim_)]))
        self.lidar_state_ = lidar_state
        

    def is_collision(self):
        jx = self.jackal_pose_.position.x
        jy = self.jackal_pose_.position.y
        px = int((jx - self.cx_) / self.sx_)
        py = int((jy - self.cy_) / self.sy_)
        if px < 0 or px >= self.img_w_ or py < 0 or py >= self.img_h_:
            return True
        if self.collision_map_.getpixel((px, py)) == 0:
            return True
        return False


    def simulation(self):
        self.target_time_ = self.time_ + self.dt_
        self.is_pause_ = False
        self.step_ = True
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
        self.step_ = False
        return


    def jackal_cmd(self, a):
        self.accel_ = a[0]
        self.omega_ = a[1]
        return


    def reset(self):
        self.reset_ = True

        #Initialize prev values
        self.prev_vel = 0
        self.prev_ang = 0
        self.prev_acc = 0
        self.prev_acc_ang = 0

        self.recent_action = [0,0]

        # check valid starting point
        # candidates = []
        # for pos in self.spawn_:
        #     check = True
        #     for name in self.actor_name_:
        #         if self.status_[name] != MOVE:
        #             continue
        #         if get_length(pos['spawn'], [self.pose_[name].x, self.pose_[name].y]) < self.spawn_threshold_:
        #             check = False
        #             break
        #     if check:
        #         candidates.append(pos)

        # randomly choice
        # candidate = random.choice(candidates)
        # self.jackal_goal_ = candidate['goal']
        self.jackal_goal_ = [28.9,16.7]

        # unpause gazebo
        self.is_pause_ = False
        self.client_unpause_()
        time.sleep(0.1)

        # replace jackal
        # self.replace_jackal(candidate['spawn'])
        self.replace_jackal([-27.0, -6.0])
        self.jackal_cmd([0,0])
        
        # pause gazebo
        time.sleep(2) # For stable state initilization
        self.client_pause_()
        self.reset_ = False
        self.target_time_ = self.time_
        self.history_queue_ = []
        self.update_state()

        return self.obs2list(self.get_obs())
    

    def step(self, a):
        s = self.get_obs()

        weight = 0.6
        a = [self.recent_action[0] * weight + np.clip(a[0], 0.0, 1.0) * (1-weight), self.recent_action[1] * weight + (1-weight) * np.clip(a[1], -1.0, 1.0)]
        self.recent_action = a

        self.jackal_cmd(a)
        self.simulation()
        ns = self.get_obs()

        reward = 0.0
        done = False

        # goal reward
        g = s[0].goal
        ng = ns[0].goal
        dg = (g.x ** 2 + g.y ** 2) ** 0.5
        #print(dg, g.x, g.y)
        dng = (ng.x ** 2 + ng.y ** 2) ** 0.5
        #print(dng, ng.x, ng.y)
        #goal_reward = self.goal_reward_coeff_ * (dg - dng) / self.dt_
        goal_reward = s[0].goal_distance - ns[0].goal_distance
        #print('goal reward : ', goal_reward)
        if dng < self.goal_threshold_:
            done = True

        # jackal cost
        control_cost = self.control_cost_coeff_ * (abs(ns[0].accel) + abs(ns[0].ang_vel - s[0].ang_vel))

        # map cost
        map_cost = self.map_cost_coeff_ * collision_cost(min(ns[0].lidar))
        if min(self.lidar_state_) < self.map_collision_threshold_:
            done = True

        
        # peds cost
        ped_cost = 0.0
        P = len(ns[0].pedestrians)
        for i in range(P):
            p = ns[0].pedestrians[i]
            d = (p.x ** 2 + p.y ** 2) ** 0.5
            #if d < self.ped_collision_threshold_:
            #    done = True
            ped_cost += collision_cost(d)
        ped_cost = self.ped_cost_coeff_ * ped_cost

        reward = goal_reward
        cost = control_cost + map_cost + ped_cost

        cost = 0

        info = {'reward':{'total': reward, 'goal': goal_reward}, 'cost': {'total': cost, 'control':control_cost, 'map': map_cost, 'ped': ped_cost}, 'done': done, 'dist': dng}
        
        if self.mode_ == 'RL':
            reward = reward - cost
        
        return self.obs2list(ns), reward, done, info


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


    def obs2list(self, obs):
        rt = []
        for i in range(len(obs)):
            state = []
            state.append(obs[i].lin_vel)
            state.append(obs[i].ang_vel)
            state.append(obs[i].accel)
            state.append(obs[i].goal.x)
            state.append(obs[i].goal.y)
            state.append(obs[i].goal_distance)
            state += obs[i].lidar
            '''
            for ped in obs[i].pedestrians:
                state.append(ped.x)
                state.append(ped.y)
            '''
            rt += state
        return np.array(rt)


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


    def update_state(self):
        state = StateInfo()
        #import ipdb;ipdb.set_trace
        jx = self.jackal_pose_.position.x
        jy = self.jackal_pose_.position.y
        q = self.jackal_pose_.orientation
        qx = 1 - 2 * (q.y ** 2 + q.z ** 2)
        qy = 2 * (q.w * q.z + q.x * q.y)
        ct = qx / (qx ** 2 + qy ** 2) ** 0.5
        st = qy / (qx ** 2 + qy ** 2) ** 0.5

        # goal state
        #print("jx, jy, goal :", jx, jy, self.jackal_goal_)
        gx, gy = transform_coordinate(self.jackal_goal_[0] - jx, self.jackal_goal_[1] - jy, ct, st)
        state.goal_distance = (gx**2 + gy**2)**0.5
        dir_gx = gx / state.goal_distance
        dir_gy = gy / state.goal_distance
        state.goal = Point(dir_gx, dir_gy, 0)
        

        # jackal state
        state.lin_vel = self.jackal_twist_.linear.x ** 2 + self.jackal_twist_.linear.y ** 2
        state.ang_vel = self.jackal_twist_.angular.z
        state.accel = 0.1 * (state.lin_vel - self.prev_vel)/self.dt_ # 0.1 multiplied for normalization, the actual value is (state.lin_vel - self.prev_vel)/self.dt

        #import ipdb;ipdb.set_trace()
        

        # lidar state
        state.lidar = ((1-np.array(self.lidar_state_)) / 10).tolist()

        # pedestrian state
        peds = []
        for name in self.actor_name_:
            px, py = transform_coordinate(self.pose_[name].x - jx, self.pose_[name].y - jy, ct, st)
            # print("px: %.2f, py: %.2f, jx: %.2f, jy: %.2f, x: %.2f, y: %.2f, ct: %.2f, st: %.2f" %(px, py, jx, jy, self.pose_[name].x, self.pose_[name].y, ct, st))
            peds.append(Point(px, py, 0.0))
        state.pedestrians = sorted(peds, key=norm_2d)
        
        self.history_queue_.append(state)
        if len(self.history_queue_) > self.history_rollout_:
            self.history_queue_.pop(0)


    def loop(self):
        # wait for publish interval
        if self.time_ - self.last_published_time_ < self.pub_interval_:
            return
        
        # control pedestrian
        for name in self.actor_name_:
            if self.status_[name] == MOVE:
                traj_num = self.traj_idx_[name]
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
                self.pub_[name].publish(rt)
            
            elif self.status_[name] == WAIT:
                p = random.uniform(0.0, 1.0)
                if p > self.actor_prob_:
                    continue
                
                # select possible trajectory
                traj_cand = [True for i in range(self.n_traj_)]
                for name2 in self.actor_name_:
                    if self.traj_idx_[name2] == -1:
                        continue
                    for i in range(self.n_traj_):
                        if self.adj_mat_[self.traj_idx_[name2]][i]:
                            traj_cand[i] = False
                s=0
                for i in range(self.n_traj_):
                    if traj_cand[i]:
                        s += 1
                if s != 0:
                    while True:
                        traj_num = random.randint(0, self.n_traj_-1)
                        if traj_cand[traj_num]:
                                break
                    self.traj_idx_[name] = traj_num

                    # INIT actor
                    rt = Command()
                    rt.name = name
                    rt.status = INIT
                    rt.goal = Pose(position=Point(self.traj_[traj_num]['waypoints'][0][0], self.traj_[traj_num]['waypoints'][0][1], 0.0))
                    self.pub_[name].publish(rt)
        
        # control jackal
        cmd = Twist()
        cmd.linear.x = self.accel_
        cmd.angular.z = self.omega_
        self.pub_jackal_.publish(cmd)

        self.last_published_time_ = self.time_
