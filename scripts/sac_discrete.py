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

from model_discrete import SACCoreDiscrete
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
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    # set environment
    env_ = PedSim(mode='RL')
    time.sleep(3.0)
    exp_obs_dim_, exp_act_dim_ = env_.get_dim()
    exp_epi_len_ = args['exp_epi_len']
    #act_limit_ = env_.action_limit_
    
    # set model
    hidden_layers_ = args['pi_hidden_layers']
    options_ = args['pi_options']
    learning_rate_ = args['learning_rate']
    
    # set hyperparameters of SAC
    replay_size_ = args['replay_size']
    gamma_ = args['gamma']
    alpha_ = args['alpha']
    polyak_ = args['polyak']

    # set training hyperparameters
    epochs_ = args['epochs']
    steps_per_epoch_ = args['steps_per_epoch']
    batch_size_ = args['batch_size']
    n_log_epi_ = args['n_log_epi']
    start_steps_ = args['start_steps']
    update_after_ = args['update_after']
    update_every_ = args['update_every']
    save_interval_ = args['save_interval']
    plot_rendering_ = args['plot_rendering']

    # set seed
    seed_ = args['seed']
    np.random.seed(seed_)

    # create model and replay buffer
    ac_ = SACCoreDiscrete(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, device_, options_).to(device_)
    #ac_weights = torch.load(POLICY_PATH + exp_name_ + "/ac_best.pt")
    #ac_.load_state_dict(ac_weights.state_dict())
    ac_tar_ = deepcopy(ac_)
    replay_buffer_ = ReplayBuffer(exp_obs_dim_, 1, replay_size_, device_)

    # target network doesn't have gradient
    for p in ac_tar_.parameters():
        p.requires_grad = False
    
    # define q loss function
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac_.q1(o)
        q2 = ac_.q2(o)
        q1_current = q1.gather(1, a.long()).squeeze(-1)
        q2_current = q2.gather(1, a.long()).squeeze(-1)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            


            a2_probs = ac_.pi(o2) + 1e-8
            a2_log_probs = torch.log(a2_probs)
            q1_tar = ac_tar_.q1(o2)
            q2_tar = ac_tar_.q2(o2)
            q_tar = torch.min(q1_tar, q2_tar)

            backup = r + gamma_ * (1-d) * torch.sum(a2_probs * (q_tar - alpha_ * a2_log_probs), dim=-1)



        # MSE loss against Bellman backup
        loss_q1 = ((q1_current - backup)**2).mean()
        loss_q2 = ((q2_current - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
        #              Q2Vals=q2.detach().cpu().numpy())

        return loss_q

    # define pi loss function
    def compute_loss_pi(data):
        o = data['obs']
        probs = ac_.pi(o) + 1e-8
        log_probs = torch.log(probs)
        q1_pi = ac_.q1(o)
        q2_pi = ac_.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        loss_pi = torch.sum(probs *(alpha_ * log_probs - q_pi), dim=-1).mean()

        # Useful info for logging
        #pi_info = dict(LogPi=log_probs.detach().cpu().numpy())

        return loss_pi

    # define update function
    def update(data):
        ac_.q1.optimizer.zero_grad()
        ac_.q2.optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        ac_.q1.optimizer.step()
        ac_.q2.optimizer.step()
        
        for p in ac_.q1.parameters():
            p.requires_grad = False
        for p in ac_.q2.parameters():
            p.requires_grad = False

        ac_.pi.optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        ac_.pi.optimizer.step()

        for p in ac_.q1.parameters():
            p.requires_grad = True
        for p in ac_.q2.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac_.parameters(), ac_tar_.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak_)
                p_targ.data.add_((1 - polyak_) * p.data)

    def get_action(o, remove_grad=True, train=True):
        a = ac_(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_), train= train)[0]
        if remove_grad:
            return a.detach().cpu().numpy()
        return a

    # for evaluation in each epoch
    def test_agent():
        tot_ep_ret = 0.0
        for _ in range(n_log_epi_):
            #print('eval step : ', _)
            o, d, ep_ret, ep_len = env_.reset(), False, 0, 0
            while not(d or (ep_len == exp_epi_len_)):
                # Take deterministic actions at test time 
                o, r, d, _ = env_.step(get_action(o, train=False))
                ep_ret += r
                ep_len += 1
            tot_ep_ret += ep_ret
        return tot_ep_ret / n_log_epi_

    total_steps = steps_per_epoch_ * epochs_
    start_time = time.time()
    o, ep_ret, ep_len = env_.reset(), 0, 0

    ts_axis = []
    rt_axis = []
    max_avg_rt = -1000.0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        #import ipdb;ipdb.set_trace()
        if t >= 0:
            a = get_action(o)
        else:
            a = [env_.get_random_action()]
        #import ipdb;ipdb.set_trace()
        # Step the env
        
        o2, r, d, info = env_.step(a)
        #print('reward : ', r, ' timestep : ', t)
        
        # print("[%.3f] timestep: %d, " %(time.time() - init_time_, t+1))
        # print(info)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == exp_epi_len_ else d

        # Store experience to replay buffer
        replay_buffer_.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == exp_epi_len_):
            #print(info['dist'])
            o, ep_ret, ep_len = env_.reset(), 0, 0

        
        # Update handling
        if t >= update_after_ and t % update_every_ == 0:
            for j in range(update_every_):
                batch = replay_buffer_.sample_batch(batch_size_)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch_ == 0:
            epoch = (t+1) // steps_per_epoch_
            avg_rt = test_agent()

            
            ts_axis.append(t+1)
            rt_axis.append(avg_rt)
            if epoch % save_interval_ == 0:
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_"+str(epoch).zfill(3)+".pt")
            if max_avg_rt < avg_rt:
                max_avg_rt = avg_rt
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_best.pt")    
            print("[%.3f] Epoch: %d, Timesteps: %d, AvgEpReward: %.3f" %(time.time() - init_time_, epoch, t+1, avg_rt))
            if plot_rendering_:
                plt.plot(ts_axis, rt_axis)
                plt.pause(0.001)
            o = env_.reset()
