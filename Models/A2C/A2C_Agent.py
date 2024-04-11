import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from Models.A2C.MLP import MLP
from utils import convert_to_vector
 


class A2C_Agent:
    def __init__(self, obs_dim, act_dim, DEVICE, lr, eps, gamma, lam, ent_coef, max_grad_norm, mlp_layers, render, timesteps_per_actor=5, n_actors=16, timesteps_per_batch=4096):
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
        self.timesteps_per_batch = timesteps_per_batch

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

    def learn(self, batch_obs, batch_acts, batch_log_probs, batch_rews, batch_vals, batch_dones, writer, n_steps_so_far):
        A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)

        V = self.critic(batch_obs).squeeze()
        batch_rtgs = A_k + V.detach()

    
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        
   
        V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

     
        logratios = curr_log_probs - batch_log_probs
        actor_loss = -(logratios * A_k.detach()).mean()

   
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

        writer.add_scalar('Actor loss - Training step', actor_loss.item(), n_steps_so_far)
        

        current_lr = self.actor_optim.param_groups[0]['lr']
        writer.add_scalar('Learning rate value - Training step', current_lr, n_steps_so_far)


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
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            
            # Reset the environment. Note that obs is short for observation. 
            obs1 = env.reset()
            obs2 = obs1
            
            # Initially, the game is not done
            done = False
            ep_t = 0

            # Run an episode
            while not done:

                # If render is specified, render the environment
                if self.render:
                    self.env.render()

                # Track done flag of the current state
                ep_dones.append(done)

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs1)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                obs1 = torch.tensor(obs1, dtype=torch.float)
                action1, log_prob1 = self.select_action(obs1)
                action2, _ = otherAgent.select_action(obs2, greedy=True) # The opponent agent is always greedy
                val = self.critic(obs1.to(self.DEVICE))

                obs1, rew, done, info = env.step(convert_to_vector(action1), otherAction=convert_to_vector(action2))
                obs2 = info['otherObs']

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action1)
                batch_log_probs.append(log_prob1)

                # Increment the episode length
                ep_t += 1

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.DEVICE)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.DEVICE)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.DEVICE)
        #print(type(batch_obs), type(batch_acts), type(batch_log_probs), type(batch_rews), type(batch_lens), type(batch_vals), type(batch_dones))
        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
