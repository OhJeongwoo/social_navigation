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
from std_msgs.msg import Bool
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from zed_interfaces.msg import ObjectsStamped, Object

from social_navigation.msg import RRT, Request


from utils import *
#from numba import njit

class PedSim:
    
    def __init__(self, mode='safeRL', gazebo_ns='/gazebo'):

        #Define RRT publisher and subscriber
        self.rrt_pub = rospy.Publisher("/request", Request, queue_size=1)
        self.rrt_sub = rospy.Subscriber("/local_goal", Point, self.callback_rrt)
        self.sig_pub = rospy.Publisher("/signal", Bool, queue_size = 1)
        self.rrt_valid = False


        # parater for file
        # self.traj_file_ = "traj.json"
        self.traj_file_ = rospkg.RosPack().get_path("social_navigation") + "/config/ped_traj_candidate.json"
        #self.spawn_file_ = "goal.json"
        self.spawn_file_ = "goal_mcts_example_v2.json"
        #self.spawn_file_ = "goal.json"
        #self.spawn_file_ = "goal_medium.json"
        self.collision_file_ = rospkg.RosPack().get_path("social_navigation") + "/config/free_space_301_1f.png"

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

        self.waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                           [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
        self.n_waypoints_ = len(self.waypoints_)

        self.ped_waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                               [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
        self.n_ped_waypoints_ = len(self.ped_waypoints_)
        

        # parameter for lidar
        self.scan_size_ = 1081
        self.scan_dim_ = 60


        # parameter for actor
        self.n_actor_ = 10 # 10
        self.actor_name_ = []
        self.group_id_ = {}
        self.r_pos_ = {}
        # self.status_ = {}
        self.actor_status_ = {}
        # self.status_time_ = {}
        self.reset_pose_ = {}
        self.pose_ = {}
        self.goal_ = {}
        # self.traj_idx_ = {}
        # self.waypoint_idx_ = {}
        self.pub_ = {}
        self.sub_status_ = {}
        self.sub_pose_ = {}

        # parameter for env
        self.max_goal_dist = 5.0

        # parameter for pseudo zed
        self.last_zed_published_time_ = 0.0
        self.zed_publish_interval_ = 0.1
        self.zed_pose_ = {}
        self.clock_ = Clock()

        

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
            self.group_id_[name] = g_ids[seq]
            self.r_pos_[name] = r_pos[seq]
            self.pose_[name] = Point()
            self.actor_status_[name] = WAIT
            # self.status_[name] = WAIT
            # self.status_time_[name] = self.time_
            # self.traj_idx_[name] = -1
            # self.waypoint_idx_[name] = -1
            self.pub_[name] = rospy.Publisher('/' + name + '/cmd', Command, queue_size=10)
            self.sub_status_[name] = rospy.Subscriber('/' + name + '/status', Status, self.callback_status)


        # define ROS communicator
        self.sub_pose_ = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_pose)
        self.pub_jackal_ = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.pub_zed_ = rospy.Publisher('/objects', ObjectsStamped, queue_size=10)
        self.pub_flag_ = rospy.Publisher('/flag', Bool, queue_size=10)
        self.set_model_ = rospy.ServiceProxy(gazebo_ns + '/set_model_state', SetModelState)
        self.sub_scan_ = rospy.Subscriber('/front/scan', LaserScan, self.callback_scan)
        self.client_pause_ = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.client_unpause_ = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        
        time.sleep(1.0)
    
        self.sub_clock_ = rospy.Subscriber('/clock', Clock, self.callback_clock)

        time.sleep(1.0)


        #self.received_flag = False

    def callback_rrt(self, msg):
        
        print('callback_rrt')
        print(time.time() - self.published_time_)

        self.published_time_ = time.time()
        self.local_goal_ = Point(msg.x, msg.y, 0)
        
        print('new request')
        mcts = Request()
        mcts.jackal = Point(self.jackal_pose_.position.x, self.jackal_pose_.position.y, 0)
        mcts.goal = Point(self.jackal_goal_[0], self.jackal_goal_[1], 0.0)
        mcts.reset = False
        self.rrt_pub.publish(mcts)
        self.mcts_initialized_ = True
        
        print('____________________________')





    def callback_status(self, msg):
        name = msg.name
        # if self.status_[name] != msg.status:
        #     self.status_[name] = msg.status
        #     self.status_time_[name] = self.time_
        if self.actor_status_[name] != msg.status:
            self.actor_status_[name] = msg.status


    def callback_pose(self, msg):
        rt = ObjectsStamped()
        if self.time_ - self.last_zed_published_time_ > self.zed_publish_interval_:
            self.last_zed_published_time_ = self.time_
            rt.header.stamp = self.clock_
            objects = []
            for name in  self.actor_name_:
                try:
                    idx = msg.name.index(name)
                    pos = msg.pose[idx].position
                    obj = Object()
                    obj.position = [pos.x, pos.y, pos.z]
                    obj.label_id = 0
                    obj.tracking_state = 1
                    # if name not in self.zed_pose_ or L2dist(pos, self.zed_pose_[name]) < 3.0:
                    #     obj.label_id = 0
                    #     obj.tracking_state = 1
                    #     self.zed_pose_[name] = pos
                    objects.append(obj)
                    rt.objects = objects
                except:
                    print("no name")
            self.pub_zed_.publish(rt)
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
        

    def send_flag(self):
        rt = Bool()
        rt.data = True
        self.pub_flag_.publish(rt) 


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
        
        
        self.jackal_cmd([0, 0])
        time.sleep(5)
        self.mcts_initialized_ = False

        #Initialize prev values
        self.prev_vel = 0
        self.prev_ang = 0
        self.prev_acc = 0
        self.prev_acc_ang = 0

        self.timestep = 0

        self.recent_action = [0,0]

        # check valid starting point
        candidates = []
        for pos in self.spawn_:
            check = True
            # for name in self.actor_name_:
            #     if self.actor_status_[name] != MOVE:
            #         continue
            #     if get_length(pos['spawn'], [self.pose_[name].x, self.pose_[name].y]) < self.spawn_threshold_:
            #         check = False
            #         break
            if check:
                candidates.append(pos)

        print('list of candidates')
        print(candidates)

        # randomly choice
        candidate = random.choice(candidates)
        self.jackal_goal_ = candidate['goal']



        

        # unpause gazebo
        self.is_pause_ = False
        self.client_unpause_()
        time.sleep(0.1)

        # replace jackal
        self.replace_jackal(candidate['spawn'])
        time.sleep(0.1)

        # print(candidate['goal'])
        self.jackal_cmd([0,0])

        self.sig_pub.publish(True)

        
        # pause gazebo
        time.sleep(6.0) # For stable state initilization
        self.client_pause_()

        #self.received_time_ = 0
        #while True:
        mcts = Request()
        mcts.jackal = Point(self.jackal_pose_.position.x, self.jackal_pose_.position.y, 0)
        mcts.goal = Point(self.jackal_goal_[0], self.jackal_goal_[1], 0.0)
        mcts.reset = True


        self.rrt_pub.publish(mcts)

        self.published_time_ = time.time()

        # while True:
        #     pass

        
            #print('sent : ', time.time())
        a = time.time()
        time.sleep(3)
        while True:
            if self.mcts_initialized_:
                self.prev_local_goal_ = self.local_goal_
                print("finished initial mcts")
            
                print('reset time : ', time.time() - a)
                break


        
        self.update_state()
        self.prev_goal_distance = self.goal_distance
        self.prev_local_goal_distance = self.local_goal_distance
        return self.obs2list(self.get_obs())
    

    def step(self, a):
        
        self.timestep += 1

        s = self.get_obs()

 


      


        weight = 0.6
        a = [self.recent_action[0] * weight + np.clip(a[0], 0.0, 1.0) * (1-weight), self.recent_action[1] * weight + (1-weight) * np.clip(a[1], -1.0, 1.0)]
        self.recent_action = a

        

        self.jackal_cmd(a)
        self.simulation()
        ns = self.get_obs()
        #print(self.local_goal_)
        #print(self.jackal_goal_)

        reward = 0.0

        success_reward = 0.0

        done = False

        # goal reward
        dist_reward = self.prev_local_goal_distance - self.dist_to_last_local_goal
        self.prev_local_goal_distance = self.local_goal_distance
        self.prev_local_goal_ = self.local_goal_
        self.prev_goal_distance = self.goal_distance
        if self.goal_distance < self.goal_threshold_:
            done = True
            success_reward = 10.0
            print("goal reached!")
 

        

        collision_cost_total = 0

        #Lidar cost  
        
        if min(self.lidar_state_) < self.map_collision_threshold_:
            done = True
            collision_cost_total = (1000 - self.timestep)/5
            print('lidar collision')



   
        


        if not done and self.timestep >= 1000:
            print('timeout!')

        reward = dist_reward + success_reward
        cost = 1/(1.0 + math.exp(10 * (min(self.lidar_state_)-0.35))) + collision_cost_total

        info = {'reward':{'total': reward, 'dist': dist_reward, 'success' : success_reward}, 'cost': {'total':cost}}
        
        
        
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
        self.send_flag()
        return o.shape[0], 2


    def get_random_action(self):
        return 2 * self.action_limit_ * (np.random.rand(2) - 0.5)



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



    def get_goal(self, g):
        traj = self.traj_[self.traj_idx_[g]]
        time = min(self.time_ - self.status_time_[g] + self.lookahead_time_, traj['time'] - EPS)
        interval = traj['interval']
        k = int(time // interval)
        A = traj['waypoints'][k]
        B = traj['waypoints'][k+1]
        alpha = (time - interval * k) / interval
        goal = interpolate(A,B,alpha)
        return Point(goal[0], goal[1], 2.0)


   
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
        #gx, gy = transform_coordinate(self.jackal_goal_[0] - jx, self.jackal_goal_[1] - jy, ct, st)
        gx, gy = transform_coordinate(self.local_goal_.x - jx, self.local_goal_.y - jy, ct, st)
        state.goal_distance = (gx**2 + gy**2)**0.5
        if state.goal_distance < 1e-6:
            dir_gx = 0
            dir_gy = 0
        else:
            dir_gx = gx / state.goal_distance
            dir_gy = gy / state.goal_distance
        state.goal = Point(dir_gx, dir_gy, 0)
        self.goal_distance = ((self.jackal_goal_[0] - jx)**2 + (self.jackal_goal_[1]-jy)**2)**0.5
        self.local_goal_distance = ((self.local_goal_.x - jx)**2 + (self.local_goal_.y-jy)**2)**0.5
        self.dist_to_last_local_goal = ((self.prev_local_goal_.x - jx)**2 + (self.prev_local_goal_.y-jy)**2)**0.5
        #self.dist_to_last_local_goal = ((self.prev_local_goal_[0] - jx)**2 + (self.prev_local_goal_[1]-jy)**2)**0.5


        state.goal_distance = state.goal_distance if self.local_goal_distance < self.max_goal_dist else self.max_goal_dist 
        


        #state.goal_distance = state.goal_distance if state.goal_distance < self.max_goal_dist else self.max_goal_dist #Clio goal distance going into

        
        
        

        # jackal state
        state.lin_vel = self.jackal_twist_.linear.x ** 2 + self.jackal_twist_.linear.y ** 2
        state.ang_vel = self.jackal_twist_.angular.z
        state.accel = 0.2 * (state.lin_vel - self.prev_vel) / self.dt_
        self.prev_vel = state.lin_vel



        # lidar state
        state.lidar = (1-np.array(self.lidar_state_) / self.max_goal_dist).tolist()

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
        peds = {}
        goals = []
        for name in self.actor_name_:
            ped = {'group': self.group_id_[name], 'pos': self.pose_[name], 'rpos': self.r_pos_[name]}
            peds[name] = ped        
        for g in range(self.n_groups_):
            if self.status_[g] == MOVE:
                traj_num = self.traj_idx_[g]
                if self.time_ - self.status_time_[g] > self.traj_[traj_num]['time']:
                    goal = None
                    self.status_[g] = WAIT
                    for name in self.actor_name_:
                        if self.group_id_[name] == g:
                            self.actor_status_[name] = WAIT
                        continue
                goal = self.get_goal(g)
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
                rt = Command()
                rt.name = name
                rt.status = INIT
                rt.goal = Pose(position=Point(self.traj_[traj_num]['waypoints'][0][0] + self.r_pos_[name].x, self.traj_[traj_num]['waypoints'][0][1] + self.r_pos_[name].y, 0.0))
                self.pub_[name].publish(rt)
        
        # control jackal
        cmd = Twist()
        cmd.linear.x = self.accel_
        cmd.angular.z = self.omega_
        self.pub_jackal_.publish(cmd)

        self.last_published_time_ = self.time_
