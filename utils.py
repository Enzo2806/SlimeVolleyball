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