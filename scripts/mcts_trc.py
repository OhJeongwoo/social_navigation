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

from model import TRCCore
import torch.nn
from utils import *
from replay_buffer import ReplayBufferTRC
from gazebo_master_mcts import PedSim
#import wandb
from torch.distributions import Normal
from scipy.stats import norm



PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"

#wandb.init(project='starlab4')
#wandb.run.name = 'trc_medium_finetune'

EPS = 1e-8

if __name__ == "__main__":
    rospy.init_node("sac")
    init_time_ = time.time()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
    parser.add_argument('--yaml', default='trc_eval', type=str)
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
    act_limit_ = env_.action_limit_
    
    # set model
    hidden_layers_ = args['pi_hidden_layers']
    options_ = args['pi_options']
    learning_rate_ = args['learning_rate']
    
    # set hyperparameters of SAC
    replay_size_ = args['replay_size']
    gamma_ = args['gamma']
    lambda_ = args['lambda']
    polyak_ = args['polyak']
    cost_d_per_step = args['cost_d']
    sigma_unit = norm.pdf(norm.ppf(args['cost_alpha']))/args['cost_alpha']
    line_decay = args['line_decay']
    max_kl = args['max_kl']
    damping_coeff = args['damping_coeff']
    num_conjugate = args['num_conjugate']


    # set training hyperparameters
    epochs_ = args['epochs']
    steps_per_epoch_ = args['steps_per_epoch']
    batch_size_ = args['batch_size']
    update_after_ = args['update_after']
    update_every_ = args['update_every']
    save_interval_ = args['save_interval']
    plot_rendering_ = args['plot_rendering']

    # set seed
    seed_ = args['seed']
    np.random.seed(seed_)

    # create model and replay buffer
    ac_ = TRCCore(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, act_limit_, device_, options_).to(device_)
    ac_weights = torch.load(POLICY_PATH + exp_name_ + "/trc_medium369.pt")
    ac_.load_state_dict(ac_weights.state_dict(), strict=False)

    replay_buffer_ = ReplayBufferTRC(exp_obs_dim_, exp_act_dim_, replay_size_, device_)

    def unnormalize_action(action):
        return 2 * action - 1

    def normalize_action(action):
        return (action+1)*0.5

    def conjugateGradient(kl: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
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
        rs_old = torch.sum(r * r)
        for i in range(num_conjugate):
            Ap = Hx(kl, p)
            pAp = torch.sum(p * Ap)
            alpha = rs_old / (pAp + EPS)
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.sum(r * r)
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def applyParams(params):
        n = 0
        for p in ac_.pi.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def flatGrad(y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.reshape(-1) for t in g])
        return g

    def getEntropy(obs):

        mu, sigma = ac_.pi._distribution(obs)
        entropy = torch.mean(torch.sum(Normal(mu, sigma).entropy(), dim=-1))

        return entropy


    def getKL(obs, old_mean, old_std):
        
        mean, std = ac_.pi._distribution(obs)

        dist = Normal(mean, std)
        old_dist = Normal(old_mean, old_std)

        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))

        return kl

    def Hx(kl: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
        return H_x + x * damping_coeff



    
    def getObjective(obs, actions, gaes, old_means, old_stds):

        mean, stds = ac_.pi._distribution(obs)
        dist = torch.distributions.Normal(mean, stds+EPS)
        log_probs = torch.sum(dist.log_prob(actions), dim=1)
        entropy = torch.mean(torch.sum(dist.entropy(), dim=1))

        old_dist = torch.distributions.Normal(old_means, old_stds +EPS)
        old_log_probs = torch.sum(old_dist.log_prob(actions), dim=1)

        objective = torch.mean(torch.exp(log_probs-old_log_probs)*gaes)
        return objective, entropy


    def getCostSurrogate(obs, actions, old_means, old_stds, cost_gaes, cost_square_gaes, cost_mean, cost_var_mean):

        

        mean, stds = ac_.pi._distribution(obs)
        dist = torch.distributions.Normal(mean, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(actions), dim=1)

        approx_cost_mean = cost_mean + (1.0/(1.0 - gamma_))*(torch.mean(torch.exp(log_probs - old_log_probs)*cost_gaes))
        approx_cost_var = cost_var_mean + (1.0/(1.0 - gamma_**2))*(torch.mean(torch.exp(log_probs - old_log_probs)*cost_square_gaes))
        cost_surrogate = approx_cost_mean + sigma_unit * torch.sqrt(torch.clamp(approx_cost_var, EPS, np.inf))

        return cost_surrogate



    def getGaesTargets(rewards, values, dones, terminates, next_values):


        deltas = rewards + (1.0 - terminates) * gamma_ * next_values - values
        #import ipdb;ipdb.set_trace()

        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t]= gaes[t] + (1.0 - dones[t]) * gamma_ * lambda_ * gaes[t+1]
        
        targets = values + gaes
        return gaes , targets
        

    def getVarGaesTargets(rewards, values, var_values, dones, terminates, next_values, next_var_values):
       
        delta = torch.square(rewards + (1.0 - terminates)*gamma_ * next_values) - torch.square(values) + \
                          (1.0 - terminates) *(gamma_ ** 2) * next_var_values - var_values

        gaes = deepcopy(delta)

        for t in reversed(range(len(gaes))):

            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t]) *(gamma_**2 * lambda_)*gaes[t+1]
   
        targets = torch.clamp(var_values +gaes, 0.0, np.inf)


        return gaes, targets

        
    
    
    #define update function
    def update(data, t):

        o = data['obs']
        o2 = data['obs2']
        action = data['act']
        reward = data['rew']
        cost = data['cost']
        done = data['done']
        terminate = data['terminate']
        action = normalize_action(action)

        
        with torch.no_grad():


            #For reward
            values = ac_.v(o)
            next_values = ac_.v(o2)

            gaes, targets = getGaesTargets(reward, values, done, terminate, next_values)

            #For cost
            cost_values = ac_.cost_v(o)
            next_cost_values = ac_.cost_v(o2)
            cost_gaes, cost_targets = getGaesTargets(cost, cost_values, done, terminate, next_cost_values)

            #For cost var
            cost_var_values = torch.square(ac_.cost_v_std(o))
            next_cost_var_values = torch.square(ac_.cost_v_std(o2))
            cost_square_gaes, cost_var_targets = getVarGaesTargets(cost, cost_values, cost_var_values, done, terminate, next_cost_values, next_cost_var_values)


            gaes = (gaes - torch.mean(gaes)) / (torch.std(gaes)+EPS)
            cost_gaes -= torch.mean(cost_gaes)
            cost_square_gaes -= torch.mean(cost_square_gaes)
            cost_mean = torch.mean(cost/(1-gamma_))
            cost_var_mean = torch.mean(cost_var_targets)
            cost_std_targets = torch.sqrt(cost_var_targets)


        # ================================================ #


        # ================= Policy Update ================= #

        means, stds = ac_.pi._distribution(o)
        old_means, old_stds = means.clone().detach(), stds.clone().detach()

        #get objective & KL & cost surrogate
        objective, entropy = getObjective(o, action, gaes, old_means, old_stds)
        cost_surrogate = getCostSurrogate(o, action, old_means, old_stds, cost_gaes, cost_square_gaes, cost_mean, cost_var_mean)

        kl = getKL(o, old_means, old_stds)


        # get gradient
        grad_g = flatGrad(objective, ac_.pi.parameters(), retain_graph = True)
        grad_b = flatGrad(-cost_surrogate, ac_.pi.parameters(), retain_graph = True)
        x_value = conjugateGradient(kl, grad_g)
        approx_g = Hx(kl, x_value)
        cost_d = cost_d_per_step / (1.0 - gamma_)
        c_value = cost_surrogate - cost_d



        # solve Lagrangian problem
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_b, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, x_value)
            optim_case = 4
        else:
            H_inv_b = conjugateGradient(kl, grad_b)
            approx_b = Hx(kl, H_inv_b)
            scalar_q = torch.dot(approx_g, x_value)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r ** 2 / scalar_s
            B_value = 2 * max_kl - c_value ** 2 / scalar_s
            if c_value < 0 and B_value < 0: # doesn't need cost constraint
                optim_case = 3
            elif c_value < 0 and B_value >= 0: 
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else: # infeasible
                optim_case = 0
        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q / (2 * max_kl))
            nu = 0
        elif optim_case in [1, 2]:
            LA, LB = [0, scalar_r / c_value], [scalar_r / c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value / B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q / (2 * max_kl)), LB)
            f_a = lambda lam: -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r * c_value / (scalar_s + EPS)
            f_b = lambda lam: -0.5 * (scalar_q / (lam + EPS) + 2 * max_kl * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2 * max_kl / (scalar_s + EPS))
        # line search
        delta_theta = (1. / (lam + EPS)) * (x_value + nu * H_inv_b) if optim_case > 0 else nu * H_inv_b
        beta = 1.0
        init_theta = torch.cat([t.reshape(-1) for t in ac_.pi.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        init_cost_surrogate = cost_surrogate.clone().detach()
        cnt = 0
        while True:
            cnt += 1
            theta = beta * delta_theta + init_theta
            applyParams(theta)
            objective, entropy = getObjective(o, action, gaes, old_means, old_stds)
            cost_surrogate = getCostSurrogate(o, action, old_means, old_stds, cost_gaes, cost_square_gaes, cost_mean, cost_var_mean)

            kl = getKL(o, old_means, old_stds)
            if kl <= max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            
            beta *= line_decay
        # ================================================= #

        #import ipdb;ipdb.set_trace()


        for _ in range(100):

            
            value_loss = torch.mean(0.5 *torch.square(ac_.v(o) - targets))
            
    
            
            cost_value_loss = torch.mean(0.5 * torch.square(ac_.cost_v(o) - cost_targets))
            
            
       
            
            cost_var_value_loss = torch.mean(0.5 * torch.square(ac_.cost_v_std(o) - cost_std_targets))
            
            total_loss = value_loss + cost_value_loss + cost_var_value_loss

            ac_.optimizer.zero_grad()
            total_loss.backward()
            ac_.optimizer.step()

            
    

            




            




        scalar = lambda x:x.item()
        value_loss = scalar(value_loss)
        cost_value_loss = scalar(cost_value_loss)
        cost_var_value_loss = scalar(cost_var_value_loss)
        objective = scalar(objective)
        cost_surrogate = scalar(cost_surrogate)
        kl = scalar(kl)
        entropy = scalar(entropy)
        return {'value loss' :value_loss, 'cost value loss' : cost_value_loss, 'cost var value loss': cost_var_value_loss, 'objective' : objective, 'cost surrogage' : cost_surrogate, 'kl' : kl, 'entropy' : entropy, 'optim_case' : optim_case}


        



        











        
        
    def get_action(o, remove_grad=True, train=True):
        
        a = ac_.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_), train=train)
        if remove_grad:
            return a.detach().cpu().numpy()[0]
        return a[0]

   

    total_steps = steps_per_epoch_ * epochs_
    start_time = time.time()
    o, ep_ret, ep_len, ep_cost = env_.reset(), 0, 0, 0

    ts_axis = []
    rt_axis = []
    max_avg_rt = -1000.0

    score_logger = []
    cost_logger = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t >= 0:

            a = get_action(o, train=True)
     

        a = unnormalize_action(a)
        
        o2, r, d, info = env_.step(a)
     
        ep_ret += r
        ep_cost += info['cost']['total']
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        #d = False if ep_len == exp_epi_len_ else d
        d = True if ep_len == exp_epi_len_ else d
        terminate = True if ep_len < exp_epi_len_ else False

        # Store experience to replay buffer
        #replay_buffer_.store(o, a, r, info['cost']['total'],o2, d, terminate)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == exp_epi_len_):
            score_logger.append(ep_ret)
            cost_logger.append(ep_cost)
            o, ep_ret, ep_len, ep_cost = env_.reset(), 0, 0, 0
        
        
        # # Update handling
        # if (t+1) >= update_after_ and (t+1) % update_every_ == 0:
        #     print('updating')
        #     for j in range(1):
        #         batch = replay_buffer_.sample_batch()
        #         log_infos = update(data=batch, t = t)
        #     log_infos['time steps']  = t
        #     log_infos['reward'] = np.mean(score_logger[-3:])
        #     log_infos['cost'] = np.mean(cost_logger[-3:])
        #     print('updated')
        #     wandb.log(log_infos)
        
         
        
        # if (t+1) % steps_per_epoch_ == 0:
        #     epoch = (t+1) // steps_per_epoch_
        #     torch.save(ac_, POLICY_PATH + exp_name_ + "/trc_medium_fintune" + str(epoch).zfill(3)+".pt")
        
