import numpy as np
import rospy
import time

from social_navigation.msg import RRT, RRTresponse
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from gazebo_master import PedSim
from utils import *
import pickle
from model_discrete import SACCoreDiscrete
import torch
import argparse
import yaml



PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"

class Demo:
    def __init__(self):
        self.ped_sim_ = PedSim()

        self.pub_ = rospy.Publisher("/rrt", RRT, queue_size=10)
        self.sub_ = rospy.Subscriber("/local_goal", RRTresponse, self.callback)
        parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
        parser.add_argument('--yaml', default='sac', type=str)
        args = parser.parse_args()


        YAML_FILE = YAML_PATH + args.yaml + ".yaml"

        # set yaml path
        with open(YAML_FILE) as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
            print(args)

        # set hyperparameters
        exp_name_ = args['exp_name'] # experiment name
        check_path(POLICY_PATH + exp_name_) # check if the path exists, if not make the directory




        

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        exp_obs_dim_, exp_act_dim_ = self.ped_sim_.get_dim()


    
        # set model
        hidden_layers_ = args['pi_hidden_layers']
        options_ = args['pi_options']
        learning_rate_ = args['learning_rate']
    


        self.ac_ = SACCoreDiscrete(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, self.device_, options_).to(self.device_)
        self.ac_weights = torch.load(POLICY_PATH + exp_name_ + "/ac_219.pt")
        self.ac_.load_state_dict(self.ac_weights.state_dict())

        self.valid = False
        self.lookahead_ = 5
        self.stop_step_ = 0
        self.n_episodes_ = 100
        time.sleep(1.0)
        self.loop()

    def callback(self, msg):
        self.path = msg.path
        self.stop = msg.stop
        if self.stop:
            self.stop_step_ += 1
        else:
            self.stop_step_ = 0
        if len(self.path) > self.lookahead_:
            self.local_goal_ = self.path[self.lookahead_-1]
        else:
            self.local_goal_ = self.path[len(self.path)-1]
        self.valid = True

    def get_action(self, o, remove_grad=True, train=True):
        #a = ac_(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_), train= train)[0]
        a = self.ac_({'grid' : torch.unsqueeze(torch.as_tensor(o['grid'], dtype=torch.float32), dim=0).to(device=self.device_),
        'flat' : torch.unsqueeze(torch.as_tensor(o['flat'], dtype=torch.float32), dim=0).to(device=self.device_)}, train=train)[0]
        if remove_grad:
            return a.detach().cpu().numpy()
        return a

    def loop(self):

        state_array = {'grid' : [], 'flat' : []}
        action_array = []
        reward_array = []
        done_array = []
        next_state_array = {'grid' : [], 'flat' : []}



        step = 0
        for i_epi in range(self.n_episodes_):
            ep_step = 0
            o = self.ped_sim_.reset()
            self.global_goal_ = Point(self.ped_sim_.jackal_goal_[0], self.ped_sim_.jackal_goal_[1], 0.0)
            self.local_goal_ = self.global_goal_
            print("%d-th episode, root: %.3f, %.3f, goal: %.3f, %.3f" %(i_epi, self.ped_sim_.jackal_pose_.position.x, self.ped_sim_.jackal_pose_.position.y, self.ped_sim_.jackal_goal_[0], self.ped_sim_.jackal_goal_[1]))
            time.sleep(1.0)
            while True:
                self.valid = False
                rt = RRT()
                rt.root = self.ped_sim_.jackal_pose_.position
                rt.goal = self.global_goal_
                
                rt.option = False
                if self.stop_step_ > 100 or ep_step == 0:
                    rt.option = True

                self.pub_.publish(rt)
                t = time.time()
                while not self.valid:
                    if time.time() - t > 1.0:
                        break
                    time.sleep(0.001)
                # self.ped_sim_.jackal_goal_ = [self.local_goal_.x, self.local_goal_.y]
                #a = purepursuit(self.ped_sim_.jackal_pose_, self.local_goal_, 1.0)
                #a = get_similar_action(a)

                a = self.get_action(o)

                

                if self.stop:
                    a = [STOP]
                o2, r, done, _ =self.ped_sim_.step(a)
                
                for key in state_array.keys():
                    state_array[key].append(o[key])
                    next_state_array[key].append(o2[key])
                action_array.append(a)
                reward_array.append(r)
                done_array.append(done)
                

                o = o2

                step += 1
                ep_step += 1
                if done or ep_step >= 200:
                    break
        expert_data = {}
        expert_data['state_array'] = state_array
        expert_data['next_state_array'] = next_state_array
        expert_data['action_array'] = action_array
        expert_data['reward_array'] = reward_array
        expert_data['done_array'] = done_array

        with open('expert.pkl', 'wb') as f:
            pickle.dump(expert_data, f)

if __name__=='__main__':
    rospy.init_node("demo")
    ped_sim = Demo()
    rospy.spin()
