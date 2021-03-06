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
from trc_models import Policy

import torch.nn
from utils import *
from gazebo_master_eval import PedSim

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"


if __name__ == "__main__":
    rospy.init_node("evaluate_policy")
    init_time_ = time.time()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--eval_name', default='trc', type=str)
    parser.add_argument('--ped_mode', default=True, type=bool)
    parser.add_argument('--terminal_condition', default='goal', type=str)
    parser.add_argument('--high_level_controller', default=False, type=bool)
    parser.add_argument('--low_level_controller', default='trc', type=str)
    parser.add_argument('--eval_type', default='general', type=str)
    parser.add_argument('--epi_len', default=1000, type=int)
    parser.add_argument('--n_episode', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)
    
    args = parser.parse_args()

    # set hyperparameters
    policy_file_ = POLICY_PATH + args.eval_name + "/best.pt"

    if args.eval_name is not 'trc':
        policy_ = torch.load(policy_file_)
    else:
        pol_parser = argparse.ArgumentParser(description='TRC')
        pol_parser.add_argument('--obs_dim', default=36, type=int)
        pol_parser.add_argument('--action_dim', default=2, type=int)
        pol_parser.add_argument('--hidden_dim', default=512, type=int)
        pol_parser.add_argument('--log_std_init', default=-1.0, type=float)
        pol_parser.add_argument('--activation', default='ReLU', type=str)
        pol_args = pol_parser.parse_args()
        policy_ = Policy(pol_args).to(device=device_)
        policy_.load_state_dict(torch.load(policy_file_)['policy'])


    # set environment
    env_ = PedSim(mode='RL', args=args)

    # set seed
    np.random.seed(args.seed)


    def get_action(o, remove_grad=True, train=True):
        # a = [[0,0]]

        if args.eval_name is not 'trc':
            a, log_prob = policy_.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_))
            if remove_grad:
                return a.detach().cpu().numpy()[0]
        else:
            a, _, _ = policy_(torch.unsqueeze(torch.tensor(o, dtype=torch.float32), dim=0).to(device=device_))
            a = a * 2 - 1 # unnormalize
            return a[0].detach().cpu().numpy()

        return a[0]


    def run_episode():
        o, d, success, ep_len, ep_col_cost, ep_dan_case, ep_type, travel_dist, travel_time = env_.reset(), False, False, 0, 0, 0, 0, 0, 0
        while not(d or (ep_len == args.epi_len)):
            o, r, d, info = env_.step(get_action(o))
            ep_len += 1
            if info['is_dangerous']:
                ep_dan_case += 1
                ep_col_cost += info['collision_cost']
            travel_dist = info['travel_dist']
            travel_time = info['travel_time']
            ep_type = info['step_type']
            if d:
                if info['success']:
                    success = True
                else:
                    success = False
                break
        rt = {}
        rt['success'] = success
        rt['travel_dist'] = travel_dist
        rt['travel_time'] = travel_time
        rt['dangerous'] = ep_dan_case
        rt['collision_cost'] = ep_col_cost
        rt['type'] = ep_type
        print(rt)
        return rt

    start_time = time.time()
    
    # Main loop: collect experience in env and update/log each epoch
    success_rate = 0.0
    tot_safety_score = 0.0
    tot_travel_dist = 0.0
    tot_travel_time = 0.0
    tot_dangerous = 0
    tot_collision_cost = 0
    tot_collision = 0
    tot_episode = 0
    tot_success = 0
    speed_weight = 20.0
    safe_weight = 50.0
    collision_weight = 30.0
    init_time = time.time()
    for i_episode in range(args.n_episode):
        result = run_episode()
        tot_episode += 1
        if result['type'] == 2:
            tot_collision += 1
        tot_collision_cost += result['collision_cost']
        tot_dangerous += result['dangerous']
        tot_travel_dist = result['travel_dist']
        tot_travel_time = result['travel_time']
        
        speed_score = min(max(tot_travel_dist/tot_travel_time - 0.5, 0.0), 0.5) * 2.0 * speed_weight
        if tot_dangerous == 0:
            safe_score = 1.0
        else:
            safe_score = 1.0 - min(max(tot_collision_cost / tot_dangerous, 0.0), 1.0)
        safe_score = safe_weight * safe_score
        collision_score = collision_weight * (1 - 1.0 * tot_collision / tot_episode)
        safety_score = speed_score + safe_score + collision_score
        if result['success']:
            tot_success += 1
        success_rate = 1.0 * tot_success / tot_episode
        driving_score = success_rate * safety_score
        print("[%.3f] Episode: %d, Success Rate: %.3f, Safety Score: %.3f, Driving Score: %.3f (Speed: %.3f, Safe: %.3f, Collision: %.3f)" %(time.time() - init_time, i_episode+1, success_rate, safety_score, driving_score, speed_score, safe_score, collision_score))