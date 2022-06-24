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
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs= torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(device=self.device),
                     obs2= torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32).to(device=self.device),
                     act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).to(device=self.device),
                     rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device=self.device),
                     done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device=self.device))
        return batch



class ReplayBufferTRC:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.terminate_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, cost, next_obs, done, terminate):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.terminate_buf[self.ptr] = terminate
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):

        start_idx = 0
        end_idx = self.max_size
        batch = dict(obs= torch.as_tensor(self.obs_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     obs2= torch.as_tensor(self.obs2_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     act=torch.as_tensor(self.act_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     rew=torch.as_tensor(self.rew_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     cost = torch.as_tensor(self.cost_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     done=torch.as_tensor(self.done_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     terminate = torch.as_tensor(self.terminate_buf[start_idx:end_idx], dtype=torch.float32).to(device=self.device),
                     )
        return batch
