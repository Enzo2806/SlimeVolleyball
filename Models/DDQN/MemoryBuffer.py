import collections
from torch import nn
import torch
import random
import numpy as np

# Implementation from: https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py
class MemoryBuffer:

    def __init__(self, buffer_size):
        # Create 4 buffers for the rewards, states, actions, and is_done flags
        self.rewards = collections.deque(maxlen=buffer_size)
        self.states = collections.deque(maxlen=buffer_size)
        self.actions = collections.deque(maxlen=buffer_size)
        self.done_flags = collections.deque(maxlen=buffer_size)

    def update(self, state, action, reward, done):
        if not done:
            self.states.append(state) # Since the next state does not matter if we are done
        self.actions.append(action)
        self.rewards.append(reward)
        self.done_flags.append(done)

    def sample(self, batch_size):
        n = len(self.done_flags)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.states)[idx], torch.LongTensor(self.actions)[idx], \
               torch.Tensor(self.states)[1+np.array(idx)], torch.Tensor(self.rewards)[idx], \
               torch.Tensor(self.done_flags)[idx]

    def reset(self):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.done_flags.clear()