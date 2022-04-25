from copy import deepcopy
import torch
import numpy as np
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse
import rospy

from model import MLP, SACCore
import torch.nn
from utils import *
from replay_buffer import ReplayBuffer
from gazebo_master import PedSim

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"

if __name__ == "__main__":
    rospy.init_node("sac")
    init_time_ = time.time()
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Demo Collector')
    parser.add_argument('--yaml', default='test', type=str)
    args = parser.parse_args()

    YAML_FILE = YAML_PATH + args.yaml + ".yaml"

    # set yaml path
    with open(YAML_FILE) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)

    # set hyperparameters
    exp_name_ = args['exp_name']
    exp_env_name_ = args['exp_env_name']
    exp_obs_dim_, exp_act_dim_ = get_env_dim(exp_env_name_)
    exp_epi_len_ = args['exp_epi_len']
    env_ = gym.make(exp_env_name_)
    exp_policy_file_name_ = args['exp_policy_file_name']
    exp_demo_file_name_ = args['exp_demo_file_name']
    n_episode_ = args['n_episode']
    policy_file_ = POLICY_PATH + exp_name_ + "/" + exp_policy_file_name_ + ".pt"

    # set seed
    seed_ = args['seed']
    np.random.seed(seed_)

    # set rendering
    render_ = args['render']

    # initialize environment
    env_ = PedSim(mode='RL')
    time.sleep(3.0)

    # create model and replay buffer
    model = torch.load(policy_file_).to(device=device_)

    def get_action(o, remove_grad=True):
        a = model.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_))[0]
        if remove_grad:
            return torch.squeeze(a.detach().cpu()).numpy()
        return a

    o, r, d, ep_ret, ep_len, n = env_.reset(), 0, False, 0, 0, 0
    i_episode = 0
    tot_ep_ret = 0
    while i_episode < n_episode_:
        a = get_action(o)
        no, r, d, _ = env_.step(a)
        ep_ret += r
        ep_len += 1
        o = no
        if d or (ep_len == exp_epi_len_):
            print("Episode %d \t EpRet %.3f \t EpLen %d" %(i_episode, ep_ret, ep_len))
            i_episode += 1
            tot_ep_ret += ep_ret
            o = env_.reset()
            ep_ret = 0
            ep_len = 0
    print("Success to save demo")
    print("# of episodes: %d, avg EpRet: %.3f" %(n_episode_, tot_ep_ret / n_episode_))