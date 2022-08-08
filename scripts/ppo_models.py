from torch.distributions import Normal
from torch.nn import functional
from torch import jit, nn
import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m, init_value=0.0):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        m.bias.data.normal_(init_value, 0.01)

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.log_std_init = args.log_std_init
        self.activation = args.activation

        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.activ = eval(f'torch.nn.{self.activation}()')
        self.output_activ = torch.sigmoid

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.log_std = torch.tensor(
            [self.log_std_init]*self.action_dim, dtype=torch.float32, 
            requires_grad=True, device=args.device
        )
        self.log_std = nn.Parameter(self.log_std)
        self.register_parameter(name="my_log_std", param=self.log_std)


    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        mean = self.output_activ(self.fc_mean(x))

        log_std = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.ones_like(mean)*torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            module.apply(initWeights)


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.activ = eval(f'torch.nn.{self.activation}()')


    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)