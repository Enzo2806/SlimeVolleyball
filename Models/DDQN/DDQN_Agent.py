from Models.DDQN.QNetwork import QNetwork
import numpy as np
from Models.DDQN.MemoryBuffer import MemoryBuffer
import torch.nn as nn
import torch
import torch.nn.functional as F

# Inspired from https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py
class DDQN_Agent:
    
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, min_epsilon, epsilon_decay, buffer_size, batch_size, DEVICE):
        
        # Initialize the Q-Network
        self.q_network = QNetwork(state_size, action_size).to(DEVICE)
        self.target_network = QNetwork(state_size, action_size).to(DEVICE)

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha)

        # Store the hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.DEVICE = DEVICE
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize the replay buffer
        self.memory = MemoryBuffer(buffer_size)

    # Select the next action using an epsilon-greedy policy
    def select_action(self, s, greedy=False):

        # Move the state to GPU
        s = s.to(self.DEVICE)

        # Calculate the values without gradients
        with torch.no_grad():
            values = self.target_network(s)

        # Select the action with epsilon-greedy policy
        if greedy:
            return np.argmax(values.cpu().numpy())
        else:
            if np.random.rand() > self.epsilon:
                return self.select_action(s, greedy=True)
            else:
                return np.random.choice(np.arange(self.action_size))

    def training_iteration(self):
        
        s_batch, a_batch, s_prime_batch, r_batch, done_batch = self.memory.sample(self.batch_size)

        # Move the data to the GPU
        s_batch = s_batch.to(self.DEVICE)
        a_batch = a_batch.to(self.DEVICE)
        s_prime_batch = s_prime_batch.to(self.DEVICE)

        # Calculate the current states' q-values using the q-network as well as the next states' q-values using both networks
        q_values = self.q_network(s_batch)
        next_q_values = self.q_network(s_prime_batch)
        next_q_values_target = self.target_network(s_prime_batch)

        # Calculate the expected q-values for the batch
        q_value = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        q_value = q_value.to('cpu')

        next_q_value = next_q_values_target.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        next_q_value = next_q_value.to('cpu')

        expected_q_value = r_batch + self.gamma * next_q_value * (1 - done_batch)

        # Calculate the loss and backpropagate
        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def remember(self, s, a, r, done):
        self.memory.update(s, a, r, done)

    def save_models(self, folder_path, agent_number, n):
        self.q_network.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-q_network.pt")
        self.target_network.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-target_network.pt")

    def save_models(self, folder_path, agent_number, n):
        self.q_network.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-q_network.pt")
        self.target_network.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-target_network.pt")
    