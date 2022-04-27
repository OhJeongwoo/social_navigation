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
from torch.distributions import Categorical

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


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(DiscreteActor, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate

        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], act_dim))
        
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        

    def forward(self, obs):
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
        x = self.fc[self.H](x)
        pi = self.softmax(x)
        return pi



class QFunctionDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(QFunctionDiscrete, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.obs_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], act_dim))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, obs):
        x = obs.to(self.device)
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
        return q




    

class SACCoreDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers, learning_rate, device, option):
        super(SACCoreDiscrete, self).__init__()
        self.device = device
        self.option = option
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.lr = learning_rate
        self.pi = DiscreteActor(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.q1 = QFunctionDiscrete(obs_dim, act_dim, hidden_layers, learning_rate, device, option)
        self.q2 = QFunctionDiscrete(obs_dim, act_dim, hidden_layers, learning_rate, device, option)

    def forward(self, obs, train=True):
        if train == False:
            actions = torch.argmax(self.pi(obs), dim=-1, keepdim=True)
            return actions
        prob = Categorical(self.pi(obs))
        
        actions = prob.sample().unsqueeze(-1)
        return actions

