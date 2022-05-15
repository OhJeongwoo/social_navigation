import numpy as np
import torch
import scipy.signal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics_scalar(x):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    return mean, std


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs_grid_buf = np.zeros((size, 15, 40, 40), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_grid_buf = np.zeros((size, 15, 40, 40), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.timeout_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done, timeout, cost):
        self.obs_buf[self.ptr] = obs['flat']
        self.obs_grid_buf[self.ptr] = obs['grid']
        self.obs2_buf[self.ptr] = next_obs['flat']
        self.obs2_grid_buf[self.ptr] = next_obs['grid']
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.timeout_buf[self.ptr] = timeout
        self.cost_buf[self.ptr] = cost
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs={'flat': torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(device=self.device),
                        'grid': torch.as_tensor(self.obs_grid_buf[idxs], dtype=torch.float32).to(device=self.device)},
                     obs2={'flat': torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32).to(device=self.device),
                        'grid': torch.as_tensor(self.obs2_grid_buf[idxs], dtype=torch.float32).to(device=self.device)},
                     act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).to(device=self.device),
                     rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device=self.device),
                     done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device=self.device),
                     timeout = torch.as_tensor(self.timeout_buf[idxs], dtype=torch.float32).to(device=self.device),
                     cost = torch.as_tensor(self.cost_buf[idxs], dtype=torch.float32).to(device=self.device))
        return batch
        #return {k: torch.as_tensor(v, dtype=torch.float32).to(device=self.device) for k,v in batch.items()}

    def sample_all(self):

        batch = dict(obs={'flat': torch.as_tensor(self.obs_buf, dtype=torch.float32).to(device=self.device),
                        'grid': torch.as_tensor(self.obs_grid_buf, dtype=torch.float32).to(device=self.device)},
                     obs2={'flat': torch.as_tensor(self.obs2_buf, dtype=torch.float32).to(device=self.device),
                        'grid': torch.as_tensor(self.obs2_grid_buf, dtype=torch.float32).to(device=self.device)},
                     act=torch.as_tensor(self.act_buf, dtype=torch.float32).to(device=self.device),
                     rew=torch.as_tensor(self.rew_buf, dtype=torch.float32).to(device=self.device),
                     done=torch.as_tensor(self.done_buf, dtype=torch.float32).to(device=self.device),
                     timeout = torch.as_tensor(self.timeout_buf, dtype=torch.float32).to(device=self.device),
                     cost = torch.as_tensor(self.cost_buf, dtype=torch.float32).to(device=self.device))
        return batch


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
