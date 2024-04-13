import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from utils import SumTree

class PrioritizedReplayBuffer(object):
	
	def __init__(self, buffer_size, state_dim, alpha, beta_init, device):
		self.ptr = 0
		self.size = 0
		max_size = int(buffer_size)
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dw = np.zeros((max_size, 1))
		self.max_size = max_size

		self.sum_tree = SumTree(max_size)
		self.alpha = alpha
		self.beta = beta_init
		self.device = device

	def save_to_disk(self, path):
		torch.save(self.state, f"{path}/state.pt")
		torch.save(self.action, f"{path}/action.pt")
		torch.save(self.reward, f"{path}/reward.pt")
		torch.save(self.dw, f"{path}/dw.pt")
		torch.save(self.ptr, f"{path}/ptr.pt")
		torch.save(self.size, f"{path}/size.pt")
		torch.save(self.beta, f"{path}/beta.pt")

	def load_from_disk(self, path):
		self.state = torch.load(f"{path}/state.pt")
		self.action = torch.load(f"{path}/action.pt")
		self.reward = torch.load(f"{path}/reward.pt")
		self.dw = torch.load(f"{path}/dw.pt")
		self.ptr = torch.load(f"{path}/ptr.pt")
		self.size = torch.load(f"{path}/size.pt")
		self.beta = torch.load(f"{path}/beta.pt")

	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw.cpu()  #0,0,0，...，1

		# 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
		priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
		self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  # 更新当前经验在sum_tree中的优先级

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)

		return (
			torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
			torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
			ind,
			Normed_IS_weight.to(self.device) # shape：(batch_size,)
		)
	
	def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
		priorities = (np.abs(td_errors) + 0.01) ** self.alpha
		for index, priority in zip(batch_index, priorities):
			self.sum_tree.update_priority(data_index=index, priority=priority)