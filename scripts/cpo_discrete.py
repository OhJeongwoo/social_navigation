from copy import deepcopy
from itertools import accumulate
from sre_constants import RANGE_UNI_IGNORE
import torch
import numpy as np
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse
import rospy
import wandb

from model_discrete import CPOCoreDiscrete
import torch.nn
from utils import *
from replay_buffer_safe import ReplayBuffer
from gazebo_master import PedSim

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"

wandb.init(project='starlab')
wandb.run.name = 'cpo'

EPS = 1e-8

if __name__ == "__main__":
    rospy.init_node("sac")
    init_time_ = time.time()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
    parser.add_argument('--yaml', default='cpo', type=str)
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
    lambda_ = args['lambda']
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
    damping_coeff = 0.01
    max_kl = 0.001
    num_conjugate = 10
    cost_d = 0.025
    cost_d = cost_d / (1.0 - gamma_)
    line_decay = 0.8
    value_epochs = 200

    # set seed
    seed_ = args['seed']
    np.random.seed(seed_)

    # create model and replay buffer
    ac_ = CPOCoreDiscrete(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, device_, options_).to(device_)
    #ac_weights = torch.load(POLICY_PATH + exp_name_ + "/ac_219.pt")
    #ac_.load_state_dict(ac_weights.state_dict())
    ac_tar_ = deepcopy(ac_)
    replay_buffer_ = ReplayBuffer(exp_obs_dim_, 1, replay_size_, device_)

    # target network doesn't have gradient
    for p in ac_tar_.parameters():
        p.requires_grad = False

    def flatGrad(y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g


    #Get GAE Targets

    def getGaesTargets(rewards, values, dones, time_outs, next_values):
        deltas = rewards + (1.0 - dones) * gamma_ * next_values- values

        gaes = deltas.clone()
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1 - time_outs[t]) * gamma_ * lambda_ * gaes[t+1]

        targets = values + gaes
        return gaes, targets


    def getObjective(flat, encoded, actions, gaes, old_probs):
        probs = ac_.pi(flat, encoded)
        action_probs = probs.gather(1, actions.long()).squeeze(-1)
        log_probs =  torch.log(action_probs)
        old_action_probs = old_probs.gather(1, actions.long()).squeeze(-1)
        old_log_probs =  torch.log(old_action_probs)

        objective = torch.mean(torch.exp(log_probs - old_log_probs) * gaes)
        return objective

    def getCostSurrogate(flat, encoded, actions, old_probs, cost_gae, cost_mean):
        probs = ac_.pi(flat, encoded)
        action_probs = probs.gather(1, actions.long()).squeeze(-1)
        log_probs =  torch.log(action_probs)
        old_action_probs = old_probs.gather(1, actions.long()).squeeze(-1)
        old_log_probs =  torch.log(old_action_probs)
        cost_surrogate = cost_mean +(1.0/(1.0- gamma_)) * (torch.mean(torch.exp(log_probs - old_log_probs)*cost_gae) - torch.mean(cost_gae))
        return cost_surrogate


    
    def getKL(flat, encoded, old_probs):
        probs = ac_.pi(flat, encoded)
        kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs) - torch.log(probs)), dim=-1))
        return kl

    def applyParams(params):
        n = 0
        for p in ac_.pi.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    
    def Hx(kl, x):
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, ac_.pi.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, ac_.pi.parameters(), retain_graph=True)
        return H_x + x*damping_coeff



    def conjugateGradient(kl, g):
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=device_)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(num_conjugate):
            Ap = Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/rs_old)*p
            rs_old = rs_new
        return x





  

    

    def update(data):

        state_flat_tensor = data['obs']['flat']
        state_grid_tensor = data['obs']['grid']
        actions_tensor = data['act']
        rewards_tensor = data['rew']
        costs_tensor = data['cost']
        dones_tensor = data['done']
        timeout_tensor = data['timeout']
        next_state_flat_tensor = data['obs']['flat']
        next_state_grid_tensor = data['obs']['grid']


        for p in ac_.value.parameters():
            p.requires_grad = False
        for p in ac_.cost_value.parameters():
            p.requires_grad = False
        for p in ac_.encoder.parameters():
            p.requires_grad = False

        
        #For reward
        encoded_o = ac_.encoder(state_grid_tensor)
        next_encoded_o = ac_.encoder(next_state_grid_tensor)
        values_tensor = ac_.value(state_flat_tensor, encoded_o)
        next_values_tensor = ac_.value(next_state_flat_tensor, next_encoded_o)

        gaes_tensor, targets_tensor = getGaesTargets(rewards_tensor, values_tensor, dones_tensor, timeout_tensor, next_values_tensor)
        
        #For cost
        cost_values_tensor = ac_.cost_value(state_flat_tensor, encoded_o)
        next_cost_values_tensor = ac_.cost_value(next_state_flat_tensor, next_encoded_o)
        cost_gaes_tensor, cost_targets_tensor = getGaesTargets(costs_tensor, cost_values_tensor, dones_tensor, timeout_tensor, next_cost_values_tensor)

        cost_mean = costs_tensor.mean().cpu().numpy() / (1 - gamma_)


        # ======================================= #
        # ========== for policy update ========== #
        #backup old policy
        old_probs = ac_.pi(state_flat_tensor, encoded_o).detach().clone() + EPS

        #get objective, KL, cost_surrogate
        objective = getObjective(state_flat_tensor, encoded_o, actions_tensor, gaes_tensor, old_probs)
        cost_surrogate = getCostSurrogate(state_flat_tensor, encoded_o, actions_tensor, old_probs, cost_gaes_tensor, cost_mean)
        kl = getKL(state_flat_tensor, encoded_o, old_probs)


        #get gradient
        grad_g = flatGrad(objective, ac_.pi.parameters(), retain_graph=True)
        grad_b = flatGrad(-cost_surrogate, ac_.pi.parameters(), retain_graph = True)
        x_value = conjugateGradient(kl, grad_g)
        approx_g = Hx(kl, x_value)
        
        c_value = cost_surrogate - cost_d

        #solve Lagrangian problem
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, x_value)
            optim_case = 4
        
        else:
            H_inv_b = conjugateGradient(kl, grad_b)
            approx_b = Hx(kl, H_inv_b)
            scalar_q = torch.dot(approx_g, x_value)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2 / scalar_s # should be always positive (Cauchy-Shwarz)
            B_value = 2*max_kl - c_value**2 / scalar_s # does safety boundary intersect trust region? (positive = yes)
            if c_value < 0 and B_value < 0:
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else:
                optim_case = 0
        
        print("optimizing case :", optim_case)
        if optim_case in [3,4]:
            lam = torch.sqrt(scalar_q/(2*max_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value/B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q/(2*max_kl)), LB)
            f_a = lambda lam : -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5 * (scalar_q / (lam + EPS) + 2*max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2*max_kl / (scalar_s+EPS))

        delta_theta = (1./(lam + EPS))*(x_value + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
        beta = 1.0
        init_theta = torch.cat([t.view(-1) for t in ac_.pi.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        init_cost_surrogate = cost_surrogate.clone().detach()
        print("KKK")
        while True:
            theta = beta*delta_theta + init_theta
            applyParams(theta)
            objective = getObjective(state_flat_tensor, encoded_o, actions_tensor, gaes_tensor, old_probs)
            cost_surrogate = getCostSurrogate(state_flat_tensor, encoded_o, actions_tensor, old_probs, cost_gaes_tensor, cost_mean)
            print(objective, cost_surrogate)
            kl = getKL(state_flat_tensor, encoded_o, old_probs)
            print(kl)
            if kl <= max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            beta *= line_decay
        print("LLLL")

        #Value, Encoder Update
        for p in ac_.value.parameters():
            p.requires_grad = True
        for p in ac_.cost_value.parameters():
            p.requires_grad = True
        for p in ac_.encoder.parameters():
            p.requires_grad = True



        for _ in range(value_epochs):
            #print(_)
            encoded_o = ac_.encoder(state_grid_tensor)
            values_tensor = ac_.value(state_flat_tensor, encoded_o)
            value_loss = torch.mean(0.5 *torch.square(values_tensor - targets_tensor))
            ac_.value.optimizer.zero_grad()
            ac_.encoder.optimizer.zero_grad()
            value_loss.backward()
            ac_.value.optimizer.step()
            ac_.encoder.optimizer.step()

            encoded_o = ac_.encoder(state_grid_tensor).detach()
            cost_values_tensor = ac_.cost_value(state_flat_tensor, encoded_o)
            cost_value_loss = torch.mean(0.5 *torch.square(cost_values_tensor - cost_targets_tensor))
            ac_.cost_value.optimizer.zero_grad()
            cost_value_loss.backward()
            ac_.cost_value.optimizer.step()
        print("MMMM")
            


        #Entropy
        encoded_o = ac_.encoder(state_grid_tensor)
        probs = ac_.pi(state_flat_tensor, encoded_o) + 1e-8
        log_probs = torch.log(probs)
        entropy = torch.sum(-probs *log_probs, dim=-1).mean()


        scalar = lambda x:x.detach().cpu().numpy()
        np_value_loss = scalar(value_loss)
        np_cost_value_loss = scalar(cost_value_loss)
        np_objective = scalar(objective)
        np_cost_surrogate = scalar(cost_surrogate)
        np_kl = scalar(kl)
        np_entropy = scalar(entropy)
        return {"value loss":np_value_loss, "cost value loss":np_cost_value_loss, 
        "objective":np_objective, "cost surrogate":np_cost_surrogate, "kl":np_kl, "entropy":np_entropy}





        


    def get_action(o, remove_grad=True, train=True):
        #a = ac_(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_), train= train)[0]
        a = ac_({'grid' : torch.unsqueeze(torch.as_tensor(o['grid'], dtype=torch.float32), dim=0).to(device=device_),
        'flat' : torch.unsqueeze(torch.as_tensor(o['flat'], dtype=torch.float32), dim=0).to(device=device_)}, train=train)[0]
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
    o, ep_ret, ep_len, ep_cost = env_.reset(), 0, 0, 0

    ts_axis = []
    rt_axis = []
    
    max_avg_rt = -1000.0
    prev_time = 0

    score_logger = []
    cost_logger = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        #import ipdb;ipdb.set_trace()
        #if t >= 0:
        
        
        if t >= args['start_steps']:
            #print(time.time() - prev_time)
            #prev_time = time.time()
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
        ep_cost += info['cost']['total']
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        #d = False if ep_len == exp_epi_len_ else d
        timeout = True if ep_len == exp_epi_len_ else False

        # Store experience to replay buffer
        replay_buffer_.store(o, a, r, o2, d, timeout, info['cost']['total'] * 0)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == exp_epi_len_):
            #print(info['dist'])
            score_logger.append(ep_ret)
            cost_logger.append(ep_cost)
            o, ep_ret, ep_len, ep_cost = env_.reset(), 0, 0, 0

        # Update handling
        if (t+1) >= update_after_ and (t+1) % update_every_ == 0:
            batch = replay_buffer_.sample_all()
            log_infos = update(data=batch)
            log_infos['time steps']  = t
            log_infos['reward'] = np.mean(score_logger[-3:])
            log_infos['cost'] = np.mean(cost_logger[-3:])
            wandb.log(log_infos)
         
       
        if (t+1) % steps_per_epoch_ == 0:
            epoch = (t+1) // steps_per_epoch_
            torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_" + str(epoch).zfill(3)+".pt")
