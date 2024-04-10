import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from Models.A2C.MLP import MLP
from utils import convert_to_vector
 


class A2C_Agent:
    def __init__(self, obs_dim, act_dim, DEVICE, lr, eps, gamma, lam, ent_coef, max_grad_norm, mlp_layers, render, timesteps_per_actor=5, n_actors=16):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.DEVICE = DEVICE
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.mlp_layers = mlp_layers
        self.render = render
        self.timesteps_per_actor = timesteps_per_actor
        self.n_actors = n_actors

        self.actor = MLP(self.obs_dim, self.act_dim, is_actor=True, DEVICE=DEVICE, fc1_dims=mlp_layers[0], fc2_dims=mlp_layers[1])
        self.critic = MLP(self.obs_dim, 1, is_actor=False, DEVICE=DEVICE, fc1_dims=mlp_layers[0], fc2_dims=mlp_layers[1])

        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr, eps=eps)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr, eps=eps)
    
 
    def save_models(self, folder_path, agent_number, n):
        self.actor.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt") 
    
    def load_models(self, folder_path, agent_number, n):

        self.actor.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt")

    def copy_models(self, agent):
        self.actor.load_state_dict(agent.actor.state_dict())
        self.critic.load_state_dict(agent.critic.state_dict())
    
    def evaluation_mode(self):
        self.actor.eval()
    
    def training_mode(self):
        self.actor.train()
    

    def disable_gradients(self):
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False

    def learn(self, batch_obs, batch_acts, batch_log_probs, batch_rews, batch_vals, batch_dones):
        #print(batch_rews.shape)
        #print(batch_vals.shape)
        #print(batch_dones.shape)
       
        A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)

        V = self.critic(batch_obs).squeeze()
        batch_rtgs = A_k + V.detach()

        #Normalize Adv fxn
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        
        V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

        # policy grad loss
        logratios = curr_log_probs - batch_log_probs
        actor_loss = -(logratios * A_k.detach()).mean()

        # entropy loss
        actor_loss -= self.ent_coef * entropy.mean()


        critic_loss = F.mse_loss(V, batch_rtgs.detach())

    
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  
        #print(rewards.shape, "is the shape of rewards")
        
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  
            last_advantage = 0  
            #print(ep_rews.shape, "is the shape of ep_rews")
            
            for t in reversed(range(len(ep_rews))): 
                if t + 1 < len(ep_rews):
                    
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  
                advantages.insert(0, advantage)  

            batch_advantages.extend(advantages)

    
        return torch.tensor(batch_advantages, dtype=torch.float).to(self.DEVICE)

    def select_action(self, obs, greedy=False):

        obs = torch.tensor(obs,dtype=torch.float).to(self.DEVICE)
        probs = self.actor(obs.to(self.DEVICE))
        

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        if greedy:
            return torch.argmax(probs).item(), 1

        return action.detach().item(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):

        V = self.critic(batch_obs).squeeze()

        probs = self.actor(batch_obs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def gather_data(self, env, otherAgent):
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.DEVICE)
            action, log_prob = self.select_action(state_tensor)
            
            other_action, _ = otherAgent.select_action(torch.tensor(state, dtype=torch.float).to(self.DEVICE), greedy=True)
            
            value = self.critic(state_tensor)

           
            converted_action = convert_to_vector(action)
            converted_other_action = convert_to_vector(other_action)

            next_state, reward, done, _ = env.step(converted_action, otherAction=converted_other_action)

            states.append(state.tolist())  
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item()) 
            log_probs.append(log_prob.item())  

            state = next_state if not done else env.reset()

        
        batch_obs = torch.tensor(states, dtype=torch.float).to(self.DEVICE)
        batch_acts = torch.tensor(actions, dtype=torch.long).to(self.DEVICE)
        #batch_rews = torch.tensor(rewards, dtype=torch.float).to(self.DEVICE)
        batch_rews = [rewards]
        #batch_dones = torch.tensor(dones, dtype=torch.float).to(self.DEVICE)
        batch_dones = [dones]
        #batch_vals = torch.tensor(values, dtype=torch.float).to(self.DEVICE)
        batch_vals = [values]
        batch_log_probs = torch.tensor(log_probs, dtype=torch.float).to(self.DEVICE)

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_vals, batch_dones


