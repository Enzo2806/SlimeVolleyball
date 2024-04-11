import slimevolleygym
import gym

# To convert the action value (0-5) to the action vector
def convert_to_vector(action_val):
    if action_val == 0:
        return [0, 0, 0]
    elif action_val == 1:
        return [0, 0, 1]
    elif action_val == 2:
        return [0, 1, 0]
    elif action_val == 3:
        return [0, 1, 1]
    elif action_val == 4:
        return [1, 0, 0]
    elif action_val == 5:
        return [1, 0, 1]
    elif action_val == 6:
        return [1, 1, 0]
    elif action_val == 7:
        return [1, 1, 1]
    else:
        raise ValueError("Invalid action value")
    
# To convert the action vector to the action value (0-5)
def convert_to_value(action_vector):
    return action_vector[0]*4 + action_vector[1]*2 + action_vector[2]



class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

