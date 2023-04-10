import numpy as np

def intuitive_policy(observation, action_space, saisons = True):
    """
    Simple rule based policy based on day or night time
    """
    month = observation[0]
    electricty_price = observation[24]
    carbon_density = observation[19]
    battery_storage = observation[22]
    
    action = 0.0 # Default value
    # Si l'électricité ne coûte pas cher et est décarbonnée
    if electricty_price < 0.27 and carbon_density < 0.16:
        if month >= 4 and month <= 9 and saisons:
            action = 0.2
        else :
            action = 0.1
    else:
        if month >= 4 and month <= 9 and saisons:
            action = -0.1
        else:
            action = -0.08
    

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action

class IntuitiveRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return intuitive_policy(observation, self.action_space[agent_id], saisons = False)

class IntuitiveSeasonRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return intuitive_policy(observation, self.action_space[agent_id], saisons = True)