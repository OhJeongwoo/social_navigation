#!/usr/bin/env python
from __future__ import print_function

##### add python path #####
import sys
import os

from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions.normal import Normal

from sklearn.utils import shuffle

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
EPS = 1e-4
LOG_STD_MIN = -4
LOG_STD_MAX = 2


def initWeights(m, init_value=0.0):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_value, 0.01)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(GaussianActor, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        self.act_limit = act_limit
 
        
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.mu_layer = nn.Linear(self.hidden_layers[self.H - 1], self.act_dim)
        self.mu_activ = torch.sigmoid
        self.log_std_layer = nn.Linear(self.hidden_layers[self.H - 1], self.act_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x, act=None):
        
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        mu = self.mu_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
 
        if act is None:
            pi_distribution = Normal(mu, std)
            pi_action = pi_distribution.rsample()
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action
            return  pi_action, logp_pi
        pi = Normal(mu, std)
        return pi, pi.log_prob(act).sum(axis=-1)
        
        
    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx != 2:
                module.apply(initWeights)
            else:
                module.apply(lambda m: initWeights(m=m, init_value=-1))


class GaussianActorV2(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(GaussianActorV2, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        self.act_limit = act_limit
        
        
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.mu_layer = nn.Linear(self.hidden_layers[self.H - 1], self.act_dim)
        self.mu_activ = torch.sigmoid
        self.log_std_layer = nn.Linear(self.hidden_layers[self.H - 1], self.act_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x, train=True, act=None):
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        mu = self.mu_activ(self.mu_layer(x))
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if train is False:
            pi_action = mu
            return pi_action
        
        if act is None:
            pi_distribution = Normal(mu, std)
            pi_action = pi_distribution.rsample()
            return pi_action
     
    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx != 2:
                module.apply(initWeights)
            else:
                module.apply(lambda m: initWeights(m=m, init_value=-1))
            

        

    def _distribution(self, x):

        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        mu = self.mu_activ(self.mu_layer(x))

        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option, positive = False):
        super(QFunction, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim + self.act_dim, self.hidden_layers[0]))
        self.lr = learning_rate

        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], 1))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.positive = positive
        if positive:
            self.positive_output = nn.ReLU()
        

    def forward(self, state, act):
        x = torch.cat([state, act], dim = -1).to(self.device)
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        q = self.fc[self.H](x)
        if self.positive:
            q = self.positive_output(q)
            q = torch.clamp(q, min=1e-8, max=1e8)
        return torch.squeeze(q, -1)
    def initialize(self):
        self.apply(initWeights)

class VFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(VFunction, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        
        
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], 1))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        v = self.fc[self.H](x)
        return torch.squeeze(v, -1)
        
    def initialize(self):
        self.apply(initWeights)

class V2Function(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(V2Function, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate
    
        
        
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.output_activ = torch.nn.Softplus()
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], 1))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        v = self.fc[self.H](x)
        v = self.output_activ(v)
        return torch.squeeze(v, -1)
        
    def initialize(self):
        self.apply(initWeights)




class SACCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(SACCore, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.act_limit = act_limit
        self.pi = GaussianActor(obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option)
        self.q1 = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.q2 = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        
        self.q1_tar = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.q2_tar = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.q_cost = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option, True)
        self.q_cost_tar = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option, True)
        self.q_std = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option, True)
        self.q_std_tar = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option, True)

        self.pi.initialize()
        self.q1.initialize()
        self.q2.initialize()
        self.q_cost.initialize()
        self.q_std.initialize()
        
        
        
    def act(self, obs, train=True):
        a, log_prob = self.pi(obs)
        return a, log_prob



class TRCCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(TRCCore, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.act_limit = act_limit
        self.pi = GaussianActorV2(obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option)
        self.v = VFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.cost_v = VFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.cost_v_std = V2Function(obs_dim, act_dim, hidden_layers, learning_rate, device, option)


        self.pi.initialize()
        self.v.initialize()
        self.cost_v.initialize()
        self.cost_v_std.initialize()

        self.optimizer = optim.Adam(list(self.v.parameters()) +list(self.cost_v.parameters())+list(self.cost_v_std.parameters()), lr = self.lr)
        
        
        
    def act(self, obs, train=True):
        if train==True:
            a = self.pi(obs)
        else:
            a = self.pi(obs, train=False)
        return a


