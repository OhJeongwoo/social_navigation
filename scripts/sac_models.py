import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

def initWeights(m, init_bias=0.0):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_bias, 0.01)

class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.state_dim = args.obs_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        self.linear1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.linear2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.linear3 = nn.Linear(self.hidden2_units, 1)        
        self.act_fn = eval(f'torch.nn.{self.activation}()')


    def forward(self, state):
        x = self.act_fn(self.linear1(state))
        x = self.act_fn(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def initialize(self):
        self.apply(initWeights)


class QNetwork(nn.Module):
    def __init__(self, args, var=False):
        super(QNetwork, self).__init__()

        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        # Q1 architecture
        self.linear1 = nn.Linear(self.state_dim + self.action_dim, self.hidden1_units)
        self.linear2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.linear3 = nn.Linear(self.hidden2_units, 1)
        self.act_fn = eval(f'torch.nn.{self.activation}()')
        
        #Is Variance Network?
        self.var = var
        if var:
           self.var_output = nn.Softplus()


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)        
        x1 = self.act_fn(self.linear1(sa))
        x1 = self.act_fn(self.linear2(x1))
        x1 = self.linear3(x1)
        if self.var:
            x1 = self.var_output(x1)
            x1 = torch.clamp(x1, min=1e-8, max=1e8)
        return x1
    
    def initialize(self):
        self.apply(initWeights)
    

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation
        self.log_std_init = args.log_std_init
        
        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = eval(f'torch.nn.{self.activation}()')
        
        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.fc_log_std = nn.Linear(self.hidden2_units, self.action_dim)


    def forward(self, state):
        x = self.act_fn(self.fc1(state))
        x = self.act_fn(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def sample(self, state):
        assert len(state.shape) == 2

        mean, log_std, std = self.forward(state)
        normal = Normal(mean, std)
        noise_action = normal.rsample()

        mu = torch.tanh(mean)
        pi = torch.tanh(noise_action)        
        logp_pi = torch.sum(normal.log_prob(noise_action), dim=1)
        logp_pi -= torch.sum(2.0*(np.log(2.0) - noise_action - nn.Softplus()(-2.0*noise_action)), dim=1)
        return mu, pi, logp_pi

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx == 4:
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)