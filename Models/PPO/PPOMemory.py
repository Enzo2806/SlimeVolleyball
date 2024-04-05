import numpy as np

class PPO_Memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size
    
    def generate_batches(self):
        """
        Generate batches of experiences for training
        Ex: if we have 20 states and batch size is 5, we will have 4 batches
        :return: list of batched experiences
        """
        # Get number of experiences stored in memory
        n_states = len(self.states)
        # Generate random indices for each batch
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        
        # Take all starting points of each batch and create a list of batches
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.logprobs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, logprob, value, reward, done):
        """
        Store experiences in memory
        :param state: current state
        :param action: action taken
        :param logprob: log probability of action
        :param value: value of state
        :param reward: reward received
        :param done: done flag
        """
        # Append experiences to memory
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    
    def clear_memory(self):
        """
        Clear memory
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []