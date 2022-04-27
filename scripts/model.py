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
EPS = 1e-6
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, learning_rate, device, option):
        super(MLP, self).__init__()
        self.device = device
        self.option = option
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            if self.option[0] == 'batch-norm':
                self.fc.append(nn.BatchNorm1d(self.hidden_layers[i-1]))
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        if self.option[0] == 'batch-norm':
                self.fc.append(nn.BatchNorm1d(self.hidden_layers[self.H - 1]))
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], self.output_dim))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        # forward network and return
        for i in range(0,self.H):
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            elif self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            elif self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            else:
                x = F.relu(self.fc[i](x))
        x = self.fc[self.H](x)
        if self.option[2] == 'sigmoid':
            x = F.sigmoid(x)
        if self.option[2] == 'tanh':
            x = F.tanh(x)
        return x


    def mse_loss(self, z, action):
        return self.loss(z, action)

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
        self.log_std_layer = nn.Linear(self.hidden_layers[self.H - 1], self.act_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, obs, act=None):
        x = obs
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

        

    def _distribution(self, obs):
        x = obs
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
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(DeterministicActor, self).__init__()
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
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], self.act_dim))
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
        x = self.fc[self.H](x)
        x = torch.tanh(x)
        x = self.act_limit * x
        return x

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
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

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1).to(self.device)
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
        return torch.squeeze(q, -1)

class VFunction(nn.Module):
    def __init__(self, obs_dim, hidden_layers, learning_rate, device, option):
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

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(Discriminator, self).__init__()
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

    def forward(self, obs, act=None):
        if act is not None:
            x = torch.cat([obs, act], dim = -1).to(self.device)
        else:
            x = torch.cat([obs], dim=-1).to(self.device)
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
        q = torch.clamp(torch.sigmoid(q), 1e-6, 1-1e-6)
        return torch.squeeze(q, -1)

    def get_reward(self, obs, act):
        with torch.no_grad():
            return -torch.log(self.forward(obs, act)).cpu().numpy()

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
        
    def act(self, obs):
        a, _ = self.pi(obs)
        return a
        



class DDPGCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(DDPGCore, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.act_limit = act_limit
        self.pi = GaussianActor(obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option)
        self.q = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)

    def act(self, obs):
        a, _ = self.pi(obs)
        return a

class PPOCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(PPOCore, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.act_limit = act_limit
        self.pi = GaussianActor(obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option)
        self.v = VFunction(obs_dim, hidden_layers, learning_rate, device, option)

    def act(self, obs):
        a, _ = self.pi(obs)
        return a

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

class ValueDiceCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(ValueDiceCore, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.act_limit = act_limit
        self.pi = GaussianActor(obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option)
        self.v = QFunction(obs_dim, act_dim, hidden_layers, learning_rate, device, option)

    def act(self, obs):
        a, _ = self.pi(obs)
        return a

class GAILCore(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, act_limit, device, option):
        super(GAILCore, self).__init__()
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
        self.D = Discriminator(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        
    def act(self, obs):
        a, _ = self.pi(obs)
        return a
