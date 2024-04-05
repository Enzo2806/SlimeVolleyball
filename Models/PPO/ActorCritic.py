import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, alpha, device, fc1_dims=256, fc2_dims=256):
        """
        Initialize the network (used for both actor and critic) with the given parameters
        :param state_dim: dimensions of the state space
        :param action_dim: dimensions of the action space
        :param alpha: learning rate
        :param chkpt_dir: directory to save the model
        TODO: device
        :param fc1_dims: number of neurons in the first fully connected layer
        :param fc2_dims: number of neurons in the second fully connected layer
        """
        # Call the parent class constructor
        super(Actor, self).__init__()

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(*state_dim, fc1_dims), # *state_dim is the same as state_dim[0], state_dim[1], etc.
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, action_dim), # Output layer has action_dim neurons      
            nn.Softmax(dim = -1)      
        )

        # Optimizer is adam
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)
        

    def forward(self, state):
        """
        Forward pass through the network, uses Boltzman
        :param state: state of the environment
        :return: mean of the action distribution
        """
        # Pass the state through the network
        return self.network(state)

    def save_checkpoint(self, path):
        """
        Save the network
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Load the network
        """
        self.load_state_dict(torch.load(path))

class Critic(nn.Module):
    def __init__(self, state_dim, alpha, device, fc1_dims=256, fc2_dims=256):
        """
        Initialize the network (used for both actor and critic) with the given parameters
        :param state_dim: dimensions of the state space
        :param alpha: learning rate
        :param chkpt_dir: directory to save the model
        TODO: device
        :param fc1_dims: number of neurons in the first fully connected layer
        :param fc2_dims: number of neurons in the second fully connected layer
        """
        # Call the parent class constructor
        super(Critic, self).__init__()

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(*state_dim, fc1_dims), # *state_dim is the same as state_dim[0], state_dim[1], etc.
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1) # Output layer has action_dim neurons      
        )

        # Optimizer is adam
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)
        

    def forward(self, state):
        """
        Forward pass through the network, uses Boltzman
        :param state: state of the environment
        :return: mean of the action distribution
        """
        # Pass the state through the network
        return self.network(state)

    def save_checkpoint(self, path):
        """
        Save the network
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Load the network
        """
        self.load_state_dict(torch.load(path))