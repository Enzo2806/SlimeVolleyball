
import torch.nn as nn
import torch.nn.functional as F
import torch

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
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