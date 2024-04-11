import torch
import numpy as np

class LightPriorReplayBuffer():
    """
    Implements a memory-efficient priority experience replay buffer that avoids explicitly storing the next state (s_next) 
    for each state-action pair. This design is particularly beneficial for environments with large state spaces, such as those involving images.
    
    The buffer includes mechanisms to handle terminal and truncated states correctly during training, ensuring the integrity of state transitions 
    and adjusting loss calculations appropriately.
    
    Attributes:
        device: The device on which the tensors will be allocated (e.g., CPU or CUDA).
        ptr: The current index for adding new experiences.
        size: The current size of the buffer.
        state: Tensor storing the states.
        action: Tensor storing the actions.
        reward: Tensor storing the rewards.
        dw: Tensor indicating whether a state is terminal (done).
        priorities: Tensor storing the priorities of experiences based on their TD-error.
        buffer_size: The maximum size of the buffer.
        alpha: The exponent determining the impact of TD-error on priority.
        beta: The exponent for importance-sampling weights, adjusts how much prioritization is used.
        replacement: Boolean indicating whether sampling with replacement is allowed.

    Usage:
        - Add experiences using the add method during environment interaction.
        - Sample batches of experiences using the sample method for training.
    """
    def __init__(self, buffer_size, state_dim, alpha, beta_init, replacement, device):
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory for the buffer's components. For image states, consider using uint8 for the state tensor to save space.
        self.state = torch.zeros((buffer_size, state_dim), device=device)
        self.action = torch.zeros((buffer_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((buffer_size, 1), device=device)
        self.dw = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device)  # 0/1 indicating done states
        self.priorities = torch.zeros(buffer_size, dtype=torch.float32, device=device)  # Priorities based on TD-error
        self.buffer_size = buffer_size

        self.alpha = alpha
        self.beta = beta_init
        self.replacement = replacement

    def add(self, state, action, reward, dw, priority):
        """
        Adds a new experience to the buffer.
        
        Arguments:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            dw: Boolean indicating if the next state is terminal.
            priority: The priority of this experience.
        """
        # Convert inputs to appropriate tensor formats and store them in the buffer.
        self.state[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.dw[self.ptr] = dw
        self.priorities[self.ptr] = (abs(priority) + 0.01) ** self.alpha  # Calculate priority

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        
        Arguments:
            batch_size: The number of experiences to sample.
        
        Returns:
            A tuple containing batches of states, actions, rewards, next states, done flags, indices of sampled experiences, 
            and normalized importance-sampling weights.
        """
        # Calculate the probabilities for sampling based on priorities, avoiding the edge case at the buffer's current pointer.
        Prob_torch_gpu = self.priorities[0:self.size].clone()
        if self.ptr < self.size: Prob_torch_gpu[self.ptr - 1] = 0  # Exclude the edge case

        # Sample indices based on calculated probabilities.
        ind = torch.multinomial(Prob_torch_gpu, num_samples=batch_size, replacement=self.replacement)

        # Calculate importance-sampling weights for the sampled experiences, normalizing them.
        IS_weight = (1. / (self.size * Prob_torch_gpu[ind])) ** self.beta
        Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1)  # Normalize weights

        # Adjust indices for next state to handle edge cases and terminal states correctly.
        next_state_indices = (ind + 1) % self.size

        return self.state[ind], self.action[ind], self.reward[ind], self.state[next_state_indices], self.dw[ind], ind, Normed_IS_weight