import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import *
from torch.distributions import Categorical
from .PPOMemory import PPO_Memory
from .ActorCritic import Actor, Critic 

class PPO_Agent:

    def __init__(self, state_size, action_size, alpha, lamda, gamma, batch_size, horizon, num_epoch, clip, device):
        """
        Initialize the PPO agent with the given parameters
        :param state_size: Number of states
        :param action_size: Number of actions
        :param alpha: Learning rate
        :param lamda: GAE parameter
        :param gamma: Discount factor
        :param horizon: Number of steps to look ahead before updating the policy
        :param batch_size: Number of experiences to sample from the memory
        :param num_epoch: Number of epochs to train the policy
        :param clip: Clipping parameter
        TODO: device
        """
        self.state_size = state_size # Number of states
        self.action_size = action_size # Number of actions
        self.alpha = alpha # Learning rate
        self.lamda = lamda # GAE parameter
        self.gamma = gamma # Discount factor
        self.batch_size = batch_size # Number of experiences to sample from the memory
        self.num_epoch = num_epoch # Number of epochs to train the policy
        self.horizon = horizon # Number of epochs before we perform an update
        self.clip = clip # Clipping parameter
        self.device = device # Device

        self.actor = Actor(state_size, action_size, alpha, device)
        self.critic = Critic(state_size, alpha, device)

        self.memory = PPO_Memory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        """
        Store experiences in memory
        :param state: current state
        :param action: action taken
        :param probs: probabilities of actions
        :param vals: values of states
        :param reward: reward received
        :param done: done flag
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, folder_path, agent_number, n):
        """
        Save the models
        """
        self.actor.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt")  

    def load_models(self, folder_path, agent_number, n):
        """
        Load the models
        """
        self.actor.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt")

    def select_action(self, observation, greedy = False):
        
        # Convert the observation to a tensor 
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        dist = self.actor(state)
        value = self.critic(state)
        # If greedy, we are testing so we return the action with max probability
        if greedy:
            return torch.argmax(dist), None, None
        
        # Categorical distribution for the action
        dist = Categorical(dist)

        # Get action by sampling from the distribution
        action = dist.sample()

        # Get rid of batch dimension by squeezing 
        # return the log probability of the action we took
        probs = torch.squeeze(dist.log_prob(action)).item() # item gives int
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        # return the action and the log probability of the action
        return action, probs, value
    
    def learn(self):

        # Iterate over the number of epochs
        for _ in range(self.num_epoch):

            # Get the batches of experiences from the memory
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            # Start calculating the advantages
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # For each step in the episode
            for t in range(len(reward_arr) - 1):
                # Calculate the discounted sum of rewards
                discount = 1
                a_t = 0
                # Calculate the advantage for each step
                for k in range(t, len(reward_arr) - 1):
                    # Advantage = sum from t to T-1 of (r_t + gamma * V(s_{t+1}) - V(s_t)) where T is the end of the episode
                    # If the episode ends, the advantage is just the reward -> value of terminal state is 0 by definition (this is not present in the formula of the paper but is a common assumption in RL)
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.lamda
                # End of every k steps, store the advantage
                advantage[t] = a_t
            
            # Turn the advantage into a tensor
            advantage = torch.tensor(advantage).to(self.device)
            
            # TODO: clean this up
            values = torch.tensor(values, dtype=torch.float32).to(self.device)

            # Iterate over the batches
            for batch in batches:
                # Get the states, actions, old probabilities, advantages and values for the current batch
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                # Compute the new probabilities pi and the values for the current batch
                # Get the distributions and values for the current batch
                dist = self.actor(states)
                # Categorical distribution for the action
                dist = Categorical(dist)
                
                critic_value = torch.squeeze(self.critic(states))

                # Calculate the probabilities of the actions taken
                new_probs = dist.log_prob(actions)

                # Calculate the ratio of the new probabilities to the old probabilities
                # We use the exponential of the difference to get the ratio 
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Calculate the unclipped surrogate loss 
                weighted_probs = prob_ratio * advantage[batch]

                # Calculate the clipped surrogate loss
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]

                # Calculate the actor loss: the minimum of the clipped and unclipped surrogate losses
                # We take the negative because we want to maximize the objective
                # We take the mean because we are averaging over the batch size 
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate the returns
                returns = advantage[batch] + values[batch]

                # Calculate the critic loss
                critic_loss = F.mse_loss(critic_value, returns)

                # Calculate the total loss
                total_loss = actor_loss + 0.5 * critic_loss

                # Zero the gradients
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # Backpropagate the total loss
                total_loss.backward()

                # Update the weights
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # Clear the memory at the end of each epoch 
        self.memory.clear_memory()